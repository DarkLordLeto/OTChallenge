/**
 * Cedar Kids Therapy — Referral Inbox Triage Agent
 *
 * Selected tools (5 of 8):
 *   1. verify_insurance — result directly forks the workflow:
 *        in_network  → create intake task, proceed toward scheduling
 *        out_of_network / expired → create billing task, block slot hold
 *        Used for every item that includes payer information (items 1, 3, 4, 7).
 *
 *   2. escalate — required by policy for P0 / P1 items:
 *        P0: any safeguarding signal (harm / abuse / neglect) → clinical_lead, same hour
 *        P1: same-day cancellation or reschedule → front_desk, immediate
 *        Used for items 2 (safeguarding) and 8 (same-day reschedule).
 *
 *   3. create_task — assigns concrete follow-up work to the right staff member;
 *        always driven by verify_insurance result or classification:
 *        billing      → out-of-network insurance
 *        clinical_lead → safeguarding escalations
 *        intake       → new referrals ready to proceed
 *        front_desk   → scheduling changes, missing-info follow-ups
 *        Used for all 8 items — the assignee and notes reflect the triage decision.
 *
 *   4. search_patient — looks up the existing patient record by name + DOB.
 *        Called FIRST for scheduling / reschedule items so the patient_id
 *        returned is used as patient_ref in hold_slot, tying the hold to
 *        the verified record rather than a raw name string.
 *
 *   5. find_slots — searches available appointment slots for a given discipline.
 *        Called for scheduling / reschedule items after search_patient,
 *        so staff can see concrete options when they review the hold.
 *
 *   6. hold_slot — places the earliest suitable slot in pending_review status.
 *        Called after find_slots when a matching slot exists. patient_ref is
 *        the patient_id from search_patient. Always followed by create_task.
 *        Result is always pending_review — this is NOT a confirmed appointment.
 *
 *   7. draft_message — composes outbound replies as drafts; NEVER auto-sent.
 *        Called for every item that warrants a reply to the family or referrer.
 *        The `body` arg is captured and surfaced as `draft_reply` in the output.
 *        Channel is chosen from context (email/phone/portal); language "es" for
 *        Spanish-speaking families. Safeguarding replies are neutral ack only.
 *
 * Architecture:
 *   Phase 1  — Claude (claude-haiku-4-5-20251001) runs an agentic loop per item,
 *              calling the 3 tools above inside withItemContext() so every call
 *              lands in the audit trace.
 *   Review   — Claude validates the output against clinic policies and returns
 *              structured feedback when rules are violated.
 *   Phase 2  — If rejected, Claude revises the output JSON using existing tool
 *              results as context. No new tool calls are made, so the trace
 *              stays clean: getToolCallsForItem() returns the same entries
 *              both before and after revision.
 */

import Anthropic from "@anthropic-ai/sdk";
import {
  withItemContext,
  getToolCallsForItem,
  search_patient,
  verify_insurance,
  escalate,
  create_task,
  find_slots,
  hold_slot,
  draft_message,
} from "./tools.js";
import type { Assignee, Discipline, InboxItem, ItemOutput } from "./types.js";

// ─── Clients ─────────────────────────────────────────────────────────────────

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

const CLAUDE_MODEL = "claude-haiku-4-5-20251001";
const REVIEW_MODEL = "claude-haiku-4-5-20251001";
const MAX_TOOL_ROUNDS = 10;
const MAX_REVISIONS = 2;

// ─── Tool definitions exposed to Claude ──────────────────────────────────────
// 5 tools: verify_insurance, escalate, create_task, find_slots, hold_slot.

const TOOL_DEFINITIONS: Anthropic.Tool[] = [
  {
    name: "verify_insurance",
    description: [
      "Verify a patient's insurance coverage against Cedar Kids Therapy's network.",
      "Returns: in_network | out_of_network | expired | unknown.",
      "ALWAYS call this for any item that contains payer or member_id information.",
      "The result determines the follow-up path:",
      "  in_network  → create an intake task for scheduling workflow",
      "  out_of_network / expired → create a billing task; do NOT recommend hold_slot",
      "  unknown     → create a billing task to verify manually",
    ].join(" "),
    input_schema: {
      type: "object",
      properties: {
        payer: { type: "string", description: "Insurance payer name from the item" },
        member_id: { type: "string", description: "Member ID from the item" },
      },
    },
  },
  {
    name: "escalate",
    description: [
      "Flag an item for immediate human oversight.",
      "REQUIRED for:",
      "  P0 — any mention of harm, abuse, neglect, or unsafe caregiving (safeguarding)",
      "  P1 — same-day cancellation or reschedule request",
      "Always follow escalate() with a create_task() assigned to the appropriate staff.",
    ].join(" "),
    input_schema: {
      type: "object",
      required: ["item_id", "reason", "severity"],
      properties: {
        item_id: { type: "string", description: "Inbox item ID" },
        reason: { type: "string", description: "Concise reason for escalation" },
        severity: {
          type: "string",
          enum: ["P0", "P1"],
          description: "P0 = safeguarding/immediate risk; P1 = same-day operational",
        },
      },
    },
  },
  {
    name: "create_task",
    description: [
      "Create a staff follow-up task. Always call this to assign concrete next steps.",
      "Choose assignee based on situation:",
      "  clinical_lead → safeguarding escalations",
      "  billing       → out-of-network or unknown insurance, benefits conversations",
      "  intake        → new referrals ready to proceed, missing-info follow-ups",
      "  front_desk    → scheduling changes, hold confirmations, general parent communication",
    ].join(" "),
    input_schema: {
      type: "object",
      required: ["assignee", "title", "due", "notes"],
      properties: {
        assignee: {
          type: "string",
          enum: ["front_desk", "intake", "billing", "clinical_lead"],
        },
        title: { type: "string", description: "Short task title (1 line)" },
        due: { type: "string", description: "Due date YYYY-MM-DD" },
        notes: {
          type: "string",
          description: "Detailed notes including patient name, key findings, and action required",
        },
      },
    },
  },
  {
    name: "search_patient",
    description: [
      "Look up an existing patient record by name and/or date of birth.",
      "Call this FIRST for any scheduling or reschedule item before find_slots or hold_slot.",
      "The patient_id returned by this tool must be used as patient_ref in hold_slot,",
      "tying the slot hold to the verified patient record.",
      "If no match is found, use the patient's full name as patient_ref instead.",
    ].join(" "),
    input_schema: {
      type: "object",
      properties: {
        name: {
          type: "string",
          description: "Patient full name — infer from item body",
        },
        dob: {
          type: "string",
          description: "Date of birth in YYYY-MM-DD format — use when present in the item",
        },
      },
    },
  },
  {
    name: "find_slots",
    description: [
      "Search available appointment slots for a discipline.",
      "Use for scheduling / reschedule items AFTER search_patient.",
      "Returns up to 5 slots with provider name, start time, and slot_id.",
      "If no slots are found, do not call hold_slot; note availability in recommended_next_action.",
    ].join(" "),
    input_schema: {
      type: "object",
      properties: {
        discipline: {
          type: "string",
          enum: ["SLP", "OT", "PT"],
          description: "Therapy discipline to search — infer from the existing appointment context",
        },
        preferences: {
          type: "string",
          description: "Optional scheduling preferences from the family (e.g. mornings)",
        },
        language: {
          type: "string",
          description: "Language preference code: 'en' or 'es'",
        },
      },
    },
  },
  {
    name: "hold_slot",
    description: [
      "Place a slot in pending_review status for staff to confirm.",
      "Only call after find_slots returns at least one slot.",
      "Always use the earliest slot_id from find_slots results.",
      "Result status is always pending_review — this is NOT a confirmed appointment.",
      "Always follow this call with create_task so front_desk knows to review and confirm the hold.",
    ].join(" "),
    input_schema: {
      type: "object",
      required: ["slot_id", "patient_ref"],
      properties: {
        slot_id: {
          type: "string",
          description: "slot_id from find_slots result",
        },
        patient_ref: {
          type: "string",
          description: "patient_id from search_patient result, or full name if no record found",
        },
      },
    },
  },
  {
    name: "draft_message",
    description: [
      "Compose an outbound reply as a draft. The message is NEVER sent automatically.",
      "Call this for every item that warrants a reply to the family or referring provider.",
      "The body you write becomes the draft_reply in the output — do NOT write draft_reply in your JSON.",
      "Channel rules: use 'email' when an email address is present, 'phone' when only a phone number exists, 'portal' for portal_message items.",
      "Language rules: use language='es' and write the body in Spanish for Spanish-speaking families.",
      "Safeguarding items (P0): body must be a neutral acknowledgement only — no clinical or investigative content.",
      "Clinical questions: body must route to clinician review — never include clinical advice.",
    ].join(" "),
    input_schema: {
      type: "object",
      required: ["recipient", "channel", "body"],
      properties: {
        recipient: {
          type: "string",
          description: "Recipient name or email address",
        },
        channel: {
          type: "string",
          enum: ["portal", "email", "phone"],
          description: "Communication channel — infer from item context",
        },
        body: {
          type: "string",
          description: "Full message body. This text becomes the draft_reply field.",
        },
        language: {
          type: "string",
          enum: ["en", "es"],
          description: "Message language — use 'es' for Spanish-speaking families",
        },
      },
    },
  },
];

// ─── System prompt ────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are a medical intake triage agent for Cedar Kids Therapy, a pediatric therapy practice for children ages 0–18.

## Your task
Triage each referral inbox item. Use the available tools to gather information and create follow-up tasks, then output a structured JSON triage record.

## Tool usage rules
- verify_insurance: call for EVERY item that includes payer or member_id. Use the result to decide the next action (in_network → intake; out_of_network/expired → billing; no slot hold for OON).
- escalate: REQUIRED for safeguarding signals (P0) and same-day cancellations/reschedules (P1). Must be called before other tools for those items.
- search_patient: call for scheduling/reschedule items to verify the patient record. Use the returned patient_id as patient_ref in hold_slot.
- find_slots: call after search_patient for scheduling/reschedule items to surface available slots for the patient's discipline.
- hold_slot: call after find_slots when slots are available. Use the earliest slot_id and the patient_id from search_patient as patient_ref. Result is always pending_review — NOT a confirmed appointment.
- create_task: call for EVERY item to assign concrete staff follow-up. Must follow hold_slot with a task for front_desk to confirm the hold. Assignee must match the situation.
- draft_message: call for EVERY item that warrants an outbound reply. The body you write becomes the draft_reply field — leave draft_reply null in your JSON. Message is never auto-sent.

## Reschedule workflow (e.g. same-day cancellation)
1. escalate(item_id, reason, "P1")
2. search_patient(name, dob)          ← verify the patient record; capture patient_id
3. find_slots(discipline)             ← discipline from the patient's existing appointment
4. hold_slot(slot_id, patient_ref)    ← earliest slot; patient_ref = patient_id from step 2
5. create_task(front_desk, "Confirm reschedule hold for <patient>", due=today, notes with patient_id + hold_id)
6. draft_message(recipient, channel, body mentioning the pending hold and next steps)

## Classification guide
- new_referral: complete or partial referral for a new evaluation
- existing_patient_request: request from a known patient family
- scheduling: cancellation, reschedule, appointment change
- clinical_question: parent asking for clinical advice or developmental info
- billing_question: insurance or payment inquiry
- missing_paperwork: referral with missing required fields
- safeguarding: any disclosure of harm, abuse, neglect, or unsafe caregiving
- other: does not fit the above

## Urgency guide
- P0: safeguarding / immediate risk → escalate required
- P1: same-day operational issue → escalate required
- P2: standard new referral or follow-up (1–2 business days)
- P3: non-urgent inquiry or routine request

## Critical policies
1. requires_human_review must always be true.
2. Out-of-network insurance (Kaiser, Cigna Select, Beacon): do NOT recommend hold_slot; billing must discuss benefits first.
3. Clinical questions: draft_message body must NOT contain clinical advice; acknowledge and route to clinician.
4. Safeguarding: draft_message body must be a neutral acknowledgement only — no investigative content.
5. Spanish-speaking families: call draft_message with language="es" and write the body in Spanish.
6. FORBIDDEN: never call schedule_appointment or send_message.

## Output format
After finishing tool calls, respond with ONLY a raw JSON object (no markdown fences):
{
  "item_id": "<from item>",
  "classification": "<see guide>",
  "urgency": "P0|P1|P2|P3",
  "requires_human_review": true,
  "extracted_intake": {
    "child_name": "<string or null>",
    "dob_or_age": "<string or null>",
    "parent_contact": "<string or null>",
    "discipline": ["SLP"|"OT"|"PT"] or null,
    "diagnosis_or_concern": "<string or null>",
    "payer": "<string or null>",
    "member_id": "<string or null>"
  },
  "missing_info": ["<field name>", ...],
  "tools_called": [],
  "recommended_next_action": "<one clear sentence>",
  "draft_reply": null,
  "task_ids": [],
  "escalation": {"reason": "<string>", "severity": "P0|P1"} or null,
  "decision_rationale": "<2–3 sentences explaining classification, urgency, and key findings>"
}
Leave tools_called, task_ids, and draft_reply as empty/null — they are populated from the execution trace.`;

// ─── Tool dispatcher ──────────────────────────────────────────────────────────

type ToolInput = Record<string, unknown>;

interface DispatchResult {
  data: unknown;
  result_summary: string;
}

async function dispatchTool(
  name: string,
  input: ToolInput,
): Promise<DispatchResult> {
  switch (name) {
    case "search_patient":
      return search_patient(
        input as { name?: string; dob?: string },
      );
    case "verify_insurance":
      return verify_insurance(
        input as { payer?: string; member_id?: string },
      );
    case "escalate":
      return escalate(
        input as { item_id: string; reason: string; severity: "P0" | "P1" },
      );
    case "create_task":
      return create_task(
        input as { assignee: Assignee; title: string; due: string; notes: string },
      );
    case "find_slots":
      return find_slots(
        input as { discipline?: Discipline; preferences?: string; language?: string },
      );
    case "hold_slot":
      return hold_slot(
        input as { slot_id: string; patient_ref: string },
      );
    case "draft_message":
      return draft_message(
        input as {
          recipient: string;
          channel: "portal" | "email" | "phone";
          body: string;
          language?: "en" | "es";
        },
      );
    default:
      throw new Error(`Tool not available in this agent: ${name}`);
  }
}

// ─── Phase 1: Claude agentic triage loop ──────────────────────────────────────

interface TriageDraft {
  output: ItemOutput;
  taskIds: string[];
  holdIds: string[];
  /** Body text from the draft_message tool call — populates draft_reply in output */
  draftReply: string | null;
  escalationResult: { reason: string; severity: "P0" | "P1" } | null;
  /** Plain-text log of tool calls for use in revision prompts */
  toolLog: string[];
}

async function triageWithClaude(
  item: InboxItem,
  feedbackFromReviewer: string | null,
): Promise<TriageDraft> {
  const taskIds: string[] = [];
  const holdIds: string[] = [];
  let draftReply: string | null = null;
  let escalationResult: { reason: string; severity: "P0" | "P1" } | null = null;
  const toolLog: string[] = [];

  const userText = feedbackFromReviewer
    ? `Triage the inbox item below. A quality reviewer rejected a previous attempt — fix all issues listed.\n\n<reviewer_feedback>\n${feedbackFromReviewer}\n</reviewer_feedback>\n\nInbox item:\n${JSON.stringify(item, null, 2)}`
    : `Triage this inbox item:\n${JSON.stringify(item, null, 2)}`;

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userText },
  ];

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    const response = await anthropic.messages.create({
      model: CLAUDE_MODEL,
      max_tokens: 4096,
      system: SYSTEM_PROMPT,
      tools: TOOL_DEFINITIONS,
      messages,
    });

    messages.push({ role: "assistant", content: response.content });

    if (response.stop_reason === "end_turn") {
      const textBlock = response.content.find((c) => c.type === "text");
      const raw =
        textBlock && textBlock.type === "text" ? textBlock.text.trim() : "{}";
      const parsed = parseItemOutput(raw, item.id);

      // Authoritative overrides from actual execution
      parsed.tools_called = getToolCallsForItem(item.id);
      parsed.task_ids = taskIds;
      parsed.draft_reply = draftReply;
      parsed.escalation = escalationResult;
      parsed.requires_human_review = true;

      return { output: parsed, taskIds, holdIds, draftReply, escalationResult, toolLog };
    }

    if (response.stop_reason === "tool_use") {
      const results: Anthropic.ToolResultBlockParam[] = [];

      for (const block of response.content) {
        if (block.type !== "tool_use") continue;

        let content: string;
        try {
          const result = await dispatchTool(
            block.name,
            block.input as ToolInput,
          );

          if (block.name === "create_task") {
            taskIds.push((result.data as { task_id: string }).task_id);
          }
          if (block.name === "hold_slot") {
            holdIds.push((result.data as { hold_id: string }).hold_id);
          }
          if (block.name === "draft_message") {
            // Capture the message body as the authoritative draft_reply
            draftReply = (block.input as { body: string }).body;
          }
          if (block.name === "escalate") {
            const inp = block.input as {
              reason: string;
              severity: "P0" | "P1";
            };
            escalationResult = {
              reason: inp.reason,
              severity: inp.severity,
            };
          }

          content = JSON.stringify(result.data);
          toolLog.push(
            `${block.name}(${JSON.stringify(block.input)}) → ${result.result_summary}`,
          );
        } catch (err) {
          content = `Error: ${err instanceof Error ? err.message : String(err)}`;
          toolLog.push(`${block.name} → ERROR: ${content}`);
        }

        results.push({
          type: "tool_result",
          tool_use_id: block.id,
          content,
        });
      }

      messages.push({ role: "user", content: results });
    }
  }

  throw new Error(`Max tool rounds exceeded for ${item.id}`);
}

// ─── Phase 2: revision without new tool calls ─────────────────────────────────

async function reviseOutput(
  item: InboxItem,
  previous: ItemOutput,
  toolLog: string[],
  feedback: string,
): Promise<ItemOutput> {
  const toolContext =
    toolLog.length > 0
      ? `Tool calls already made for this item:\n${toolLog.join("\n")}`
      : "No tools were called.";

  // No tools passed — Claude produces text only in this pass
  const response = await anthropic.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 4096,
    system: SYSTEM_PROMPT,
    messages: [
      {
        role: "user",
        content: [
          `Revise the triage output below to fix the reviewer's feedback.`,
          `Do NOT call any tools — use the results already captured.`,
          ``,
          `Inbox item:`,
          JSON.stringify(item, null, 2),
          ``,
          toolContext,
          ``,
          `Previous output:`,
          JSON.stringify(previous, null, 2),
          ``,
          `Reviewer feedback (all issues must be fixed):`,
          feedback,
          ``,
          `Respond with ONLY the corrected JSON object.`,
        ].join("\n"),
      },
    ],
  });

  const textBlock = response.content.find((c) => c.type === "text");
  const raw =
    textBlock && textBlock.type === "text" ? textBlock.text.trim() : "{}";
  const revised = parseItemOutput(raw, item.id);

  // These all come from Phase 1 tool execution and never change in revision
  revised.tools_called = previous.tools_called;
  revised.task_ids = previous.task_ids;
  revised.draft_reply = previous.draft_reply;
  revised.escalation = previous.escalation ?? revised.escalation;
  revised.requires_human_review = true;

  return revised;
}

// ─── Anthropic policy review ──────────────────────────────────────────────────

interface ReviewResult {
  approved: boolean;
  feedback: string;
}

const REVIEW_RULES = `
Rules to enforce (check each one):
1. SAFEGUARDING — if the item body mentions harm, abuse, neglect, or unsafe caregiving:
   classification must be "safeguarding", urgency must be "P0",
   escalation must not be null and severity must be "P0".
2. SAME-DAY RESCHEDULE/CANCELLATION — classification "scheduling", urgency "P1",
   escalation must not be null and severity must be "P1",
   tools_called must include search_patient (to verify the patient record first),
   tools_called must include find_slots (to surface available slots for staff review),
   tools_called must include hold_slot (patient_ref must be the patient_id from search_patient),
   tools_called must include create_task for front_desk to confirm the hold.
3. OUT-OF-NETWORK insurance (Kaiser, Cigna Select, Beacon):
   tools_called must include verify_insurance;
   recommended_next_action must NOT suggest hold_slot or confirmed scheduling.
4. CLINICAL QUESTION — classification "clinical_question";
   draft_reply must NOT contain clinical advice or developmental assessments.
5. INCOMPLETE REFERRAL — classification "missing_paperwork";
   missing_info must list every blank field.
6. requires_human_review must be true.
7. escalation must not be null for P0 or P1 items.
8. tools_called must be non-empty (at least one tool was called).
9. task_ids must be non-empty (at least one task was created per item).
10. decision_rationale must reference the key finding (insurance status, safeguarding signal, slot hold ID, etc.).
11. draft_reply must not be null — a draft_message tool call must have been made for every item.
`.trim();

async function reviewWithClaude(
  item: InboxItem,
  output: ItemOutput,
): Promise<ReviewResult> {
  const response = await anthropic.messages.create({
    model: REVIEW_MODEL,
    max_tokens: 1200,
    system:
      "You are a strict policy reviewer for a medical intake triage system. " +
      "Check every rule. Return only valid JSON.",
    messages: [
      {
        role: "user",
        content: [
          REVIEW_RULES,
          "",
          "## Inbox item",
          JSON.stringify(item, null, 2),
          "",
          "## Triage output",
          JSON.stringify(output, null, 2),
          "",
          "Return a raw JSON object — no markdown fences, no explanation, no extra keys:",
          '{"approved": true, "feedback": ""}',
          "or",
          '{"approved": false, "feedback": "<concise list of violated rules>"}',
        ].join("\n"),
      },
    ],
  });

  const textBlock = response.content.find((block) => block.type === "text");
  const raw =
    textBlock && textBlock.type === "text"
      ? textBlock.text
      : '{"approved":false,"feedback":"No reviewer response"}';

  try {
    return JSON.parse(stripFences(raw)) as ReviewResult;
  } catch {
    return {
      approved: false,
      feedback: `Reviewer response unparseable: ${raw.slice(0, 200)}`,
    };
  }
}

// ─── Per-item orchestration ───────────────────────────────────────────────────

async function processItem(item: InboxItem): Promise<ItemOutput> {
  // Phase 1 — agentic triage with tool calls recorded in trace
  const draft = await withItemContext(item.id, () =>
    triageWithClaude(item, null),
  );

  let output = draft.output;

  // Review + revision loop (revision never makes new tool calls)
  for (let revision = 0; revision < MAX_REVISIONS; revision++) {
    const review = await reviewWithClaude(item, output);
    const verdict = review.approved ? "✓ approved" : "✗ rejected";
    console.log(`  [${item.id}] review ${revision + 1}: ${verdict}`);

    if (review.approved) return output;

    console.log(`  [${item.id}] feedback: ${review.feedback}`);

    output = await reviseOutput(
      item,
      output,
      draft.toolLog,
      review.feedback,
    );
    // Ensure authoritative trace values are preserved after revision
    output.tools_called = getToolCallsForItem(item.id);
    output.task_ids = draft.taskIds;
    output.draft_reply = draft.draftReply;
    output.escalation = draft.escalationResult;
    output.requires_human_review = true;
  }

  // Final check after last revision
  const final = await reviewWithClaude(item, output);
  console.log(
    `  [${item.id}] final: ${final.approved ? "✓ approved" : "✗ using last output"}`,
  );
  return output;
}

// ─── Shared helpers ───────────────────────────────────────────────────────────

/**
 * Remove markdown code fences that Claude sometimes wraps around JSON despite
 * being told not to.  Handles ```json, ```JSON, ``` (plain), and leading/trailing
 * whitespace.  Falls back to the original string if no fence is found.
 */
function stripFences(text: string): string {
  const match = text.match(/^```(?:[a-zA-Z]*)?\s*([\s\S]*?)```\s*$/);
  return match ? match[1].trim() : text.trim();
}

// ─── Output parser ────────────────────────────────────────────────────────────

function parseItemOutput(text: string, itemId: string): ItemOutput {
  const jsonStr = stripFences(text);

  try {
    const p = JSON.parse(jsonStr) as Partial<ItemOutput>;
    return {
      item_id: (p.item_id as string | undefined) ?? itemId,
      classification: p.classification ?? "other",
      urgency: p.urgency ?? "P2",
      requires_human_review: true,
      extracted_intake: p.extracted_intake ?? {
        child_name: null,
        dob_or_age: null,
        parent_contact: null,
        discipline: null,
        diagnosis_or_concern: null,
        payer: null,
        member_id: null,
      },
      missing_info: p.missing_info ?? [],
      tools_called: [],
      recommended_next_action:
        p.recommended_next_action ?? "Requires manual review.",
      draft_reply: p.draft_reply ?? null,
      task_ids: [],
      escalation: p.escalation ?? null,
      decision_rationale:
        p.decision_rationale ?? "See tools_called for decision details.",
    };
  } catch {
    return {
      item_id: itemId,
      classification: "other",
      urgency: "P2",
      requires_human_review: true,
      extracted_intake: {
        child_name: null,
        dob_or_age: null,
        parent_contact: null,
        discipline: null,
        diagnosis_or_concern: null,
        payer: null,
        member_id: null,
      },
      missing_info: ["agent_output_parse_error"],
      tools_called: [],
      recommended_next_action:
        "Manual review required — agent output could not be parsed.",
      draft_reply: null,
      task_ids: [],
      escalation: null,
      decision_rationale: `Parse error. Raw: ${text.slice(0, 300)}`,
    };
  }
}

// ─── Entry point ──────────────────────────────────────────────────────────────

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  console.log(`\nTriaging ${inbox.length} inbox items...\n`);
  const results: ItemOutput[] = [];

  for (const item of inbox) {
    console.log(`[${item.id}] ${item.subject}`);
    const output = await processItem(item);
    console.log(
      `  → ${output.classification} | ${output.urgency} | ` +
        `tools: ${output.tools_called.length} | tasks: ${output.task_ids.length}\n`,
    );
    results.push(output);
  }

  return results;
}
