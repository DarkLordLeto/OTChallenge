/**
 * Cedar Kids Therapy — Referral Inbox Triage Agent
 *
 * Architecture:
 *  1. Claude (claude-haiku-4-5-20251001) runs the agentic triage loop for each item:
 *     - Calls real tool implementations in tools.ts (recorded in trace automatically)
 *     - Produces a structured ItemOutput JSON
 *  2. OpenAI (gpt-4o-mini) reviews the output against clinic policies
 *  3. If OpenAI rejects the output, it sends feedback back to Claude for a revision pass
 *     - Revision pass shares the original tool results as context (no new tool calls)
 *     - This keeps the trace clean: tools are only called once per item
 *  4. After MAX_REVIEW_RETRIES, the last produced output is used regardless
 */

import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import {
  withItemContext,
  getToolCallsForItem,
  search_patient,
  verify_insurance,
  lookup_policy,
  find_slots,
  hold_slot,
  create_task,
  draft_message,
  escalate,
} from "./tools.js";
import type {
  Assignee,
  Discipline,
  InboxItem,
  ItemOutput,
  PolicyTopic,
} from "./types.js";

// ─── Clients ─────────────────────────────────────────────────────────────────

const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const CLAUDE_MODEL = "claude-haiku-4-5-20251001";
const OPENAI_MODEL = "gpt-4o-mini";
const MAX_REVIEW_RETRIES = 2; // OpenAI can reject and re-prompt Claude up to this many times
const MAX_TOOL_ROUNDS = 12;   // safety cap on the agentic loop per item

// ─── Tool definitions for Claude API ─────────────────────────────────────────

const TOOL_DEFINITIONS: Anthropic.Tool[] = [
  {
    name: "search_patient",
    description:
      "Search for existing patients by name and/or date of birth. Use for any item that mentions a patient by name.",
    input_schema: {
      type: "object",
      properties: {
        name: { type: "string", description: "Full or partial patient name" },
        dob: {
          type: "string",
          description: "Date of birth in YYYY-MM-DD format",
        },
      },
    },
  },
  {
    name: "verify_insurance",
    description:
      "Verify a patient's insurance coverage. Returns in_network, out_of_network, expired, or unknown.",
    input_schema: {
      type: "object",
      properties: {
        payer: { type: "string", description: "Insurance payer name" },
        member_id: { type: "string", description: "Member ID from insurance card" },
      },
    },
  },
  {
    name: "lookup_policy",
    description: "Retrieve clinic policy snippets by topic.",
    input_schema: {
      type: "object",
      required: ["topic"],
      properties: {
        topic: {
          type: "string",
          enum: [
            "service_lines",
            "insurance",
            "safeguarding",
            "clinical_advice",
            "scheduling",
            "cancellation",
            "language_access",
          ],
          description: "Policy topic to look up",
        },
      },
    },
  },
  {
    name: "find_slots",
    description:
      "Search available appointment slots. Only use when insurance is in-network and intake data is complete.",
    input_schema: {
      type: "object",
      properties: {
        discipline: {
          type: "string",
          enum: ["SLP", "OT", "PT"],
          description: "Therapy discipline",
        },
        preferences: {
          type: "string",
          description: "Scheduling preferences (e.g. mornings, after school)",
        },
        language: {
          type: "string",
          description: "Language preference code (e.g. 'es' for Spanish)",
        },
      },
    },
  },
  {
    name: "hold_slot",
    description:
      "Reserve a slot in pending_review status. ONLY call when insurance is confirmed in-network AND intake is complete. This does NOT confirm an appointment.",
    input_schema: {
      type: "object",
      required: ["slot_id", "patient_ref"],
      properties: {
        slot_id: {
          type: "string",
          description: "Slot ID returned by find_slots",
        },
        patient_ref: {
          type: "string",
          description: "Patient name or ID for reference",
        },
      },
    },
  },
  {
    name: "create_task",
    description: "Create a follow-up task for clinic staff.",
    input_schema: {
      type: "object",
      required: ["assignee", "title", "due", "notes"],
      properties: {
        assignee: {
          type: "string",
          enum: ["front_desk", "intake", "billing", "clinical_lead"],
        },
        title: { type: "string" },
        due: { type: "string", description: "Due date in YYYY-MM-DD format" },
        notes: { type: "string", description: "Detailed task notes" },
      },
    },
  },
  {
    name: "draft_message",
    description:
      "Compose an outbound message draft. Message stays in draft state — it is never sent automatically.",
    input_schema: {
      type: "object",
      required: ["recipient", "channel", "body"],
      properties: {
        recipient: { type: "string" },
        channel: { type: "string", enum: ["portal", "email", "phone"] },
        body: { type: "string" },
        language: { type: "string", enum: ["en", "es"] },
      },
    },
  },
  {
    name: "escalate",
    description:
      "Flag an item for immediate human oversight. Required for P0 (safeguarding, immediate risk) and P1 (same-day operational issues).",
    input_schema: {
      type: "object",
      required: ["item_id", "reason", "severity"],
      properties: {
        item_id: { type: "string" },
        reason: { type: "string" },
        severity: { type: "string", enum: ["P0", "P1"] },
      },
    },
  },
];

// ─── System prompt for Claude ─────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are a medical intake triage agent for Cedar Kids Therapy, a pediatric therapy practice serving children ages 0–18.

## Your task
For each inbox item, use the provided tools to gather relevant information, then produce a single JSON object (ItemOutput) that captures your triage decision.

## Critical policies
1. SAFEGUARDING — any mention of harm, abuse, neglect, or unsafe caregiving:
   - classification: "safeguarding", urgency: "P0"
   - MUST call escalate() with severity "P0"
   - draft_reply must be a neutral acknowledgement only — no investigative advice
2. SAME-DAY CANCELLATION / RESCHEDULE:
   - classification: "scheduling", urgency: "P1"
   - MUST call escalate() with severity "P1"
3. OUT-OF-NETWORK insurance (Kaiser, Cigna Select, Beacon):
   - call verify_insurance() + lookup_policy({topic:"insurance"})
   - create_task() for billing team to discuss benefits
   - DO NOT call hold_slot() — slot holds require a benefits conversation first
4. IN-NETWORK payers: Aetna, Blue Cross Blue Shield, UnitedHealthcare, Medicaid
5. CLINICAL QUESTIONS:
   - classification: "clinical_question"
   - NEVER provide clinical advice in draft_reply; route to clinician or screening
6. INCOMPLETE REFERRALS:
   - classification: "missing_paperwork"
   - list every missing field in missing_info[]
7. SPANISH-SPEAKING FAMILIES:
   - call find_slots() with language:"es"
   - draft_reply in Spanish
8. requires_human_review MUST always be true
9. FORBIDDEN: never call schedule_appointment or send_message

## Tool guidelines
- search_patient() — use for existing/reschedule patients when name+DOB available
- verify_insurance() — use for any item with payer info
- lookup_policy() — use when policy context needed (safeguarding, insurance, clinical_advice, etc.)
- find_slots() — use for new referrals after confirming in-network insurance
- hold_slot() — only for in-network + complete intake; always pending_review
- create_task() — assign follow-ups to appropriate staff
- draft_message() — compose outbound replies (stays draft)
- escalate() — required for P0 and P1

## Output format
After using tools, respond with ONLY a raw JSON object (no markdown fences, no explanation):
{
  "item_id": "<from input>",
  "classification": "<see enum>",
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
  "missing_info": [],
  "tools_called": [],
  "recommended_next_action": "<string>",
  "draft_reply": "<string or null>",
  "task_ids": [],
  "escalation": {"reason":"<string>","severity":"P0|P1"} or null,
  "decision_rationale": "<string>"
}
Leave tools_called and task_ids as empty arrays — they will be populated from the execution trace.`;

// ─── Tool dispatcher ──────────────────────────────────────────────────────────

type ToolInput = Record<string, unknown>;

async function executeTool(
  name: string,
  input: ToolInput,
): Promise<{ data: unknown; result_summary: string }> {
  switch (name) {
    case "search_patient":
      return search_patient(input as { name?: string; dob?: string });
    case "verify_insurance":
      return verify_insurance(input as { payer?: string; member_id?: string });
    case "lookup_policy":
      return lookup_policy(input as { topic: PolicyTopic });
    case "find_slots":
      return find_slots(
        input as { discipline?: Discipline; preferences?: string; language?: string },
      );
    case "hold_slot":
      return hold_slot(input as { slot_id: string; patient_ref: string });
    case "create_task":
      return create_task(
        input as { assignee: Assignee; title: string; due: string; notes: string },
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
    case "escalate":
      return escalate(
        input as { item_id: string; reason: string; severity: "P0" | "P1" },
      );
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// ─── Phase 1: Claude agentic triage loop ──────────────────────────────────────

interface TriageRun {
  output: ItemOutput;
  taskIds: string[];
  escalationResult: { reason: string; severity: "P0" | "P1" } | null;
  toolSummaryLines: string[]; // human-readable tool call log for revision context
}

async function triageWithClaude(
  item: InboxItem,
  priorFeedback: string | null,
): Promise<TriageRun> {
  const taskIds: string[] = [];
  let escalationResult: { reason: string; severity: "P0" | "P1" } | null = null;
  const toolSummaryLines: string[] = [];

  const userMessage = priorFeedback
    ? `Triage this inbox item. A previous attempt was rejected by the quality reviewer:\n<feedback>\n${priorFeedback}\n</feedback>\n\nPlease fix these issues.\n\nInbox item:\n${JSON.stringify(item, null, 2)}`
    : `Triage this inbox item:\n${JSON.stringify(item, null, 2)}`;

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: userMessage },
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

    // Claude finished — parse its JSON output
    if (response.stop_reason === "end_turn") {
      const textBlock = response.content.find((c) => c.type === "text");
      const rawText =
        textBlock && textBlock.type === "text" ? textBlock.text.trim() : "{}";

      const parsed = parseItemOutput(rawText, item.id);

      // Authoritative sources override Claude's self-reported values
      parsed.tools_called = getToolCallsForItem(item.id);
      parsed.task_ids = taskIds;
      parsed.escalation = escalationResult;
      parsed.requires_human_review = true;

      return { output: parsed, taskIds, escalationResult, toolSummaryLines };
    }

    // Claude made tool calls — execute them and feed results back
    if (response.stop_reason === "tool_use") {
      const toolResults: Anthropic.ToolResultBlockParam[] = [];

      for (const block of response.content) {
        if (block.type !== "tool_use") continue;

        let resultContent: string;
        try {
          const result = await executeTool(
            block.name,
            block.input as ToolInput,
          );

          // Track task IDs and escalations from tool results
          if (block.name === "create_task") {
            const d = result.data as { task_id: string };
            taskIds.push(d.task_id);
          }
          if (block.name === "escalate") {
            const inp = block.input as { reason: string; severity: "P0" | "P1" };
            escalationResult = { reason: inp.reason, severity: inp.severity };
          }

          resultContent = JSON.stringify(result.data);
          toolSummaryLines.push(
            `${block.name}(${JSON.stringify(block.input)}) → ${result.result_summary}`,
          );
        } catch (err) {
          resultContent = `Error: ${err instanceof Error ? err.message : String(err)}`;
          toolSummaryLines.push(`${block.name} → ERROR: ${resultContent}`);
        }

        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: resultContent,
        });
      }

      messages.push({ role: "user", content: toolResults });
    }
  }

  throw new Error(`Max tool rounds (${MAX_TOOL_ROUNDS}) exceeded for ${item.id}`);
}

// ─── Phase 2: Claude revision (no new tool calls) ─────────────────────────────

async function reviseWithClaude(
  item: InboxItem,
  previousOutput: ItemOutput,
  toolSummaryLines: string[],
  feedback: string,
): Promise<ItemOutput> {
  const toolContext =
    toolSummaryLines.length > 0
      ? `Tool calls already executed for this item:\n${toolSummaryLines.join("\n")}`
      : "No tools were called in the previous attempt.";

  const messages: Anthropic.MessageParam[] = [
    {
      role: "user",
      content: `Revise your triage output based on reviewer feedback. Do NOT call any tools — use the results already gathered.

Inbox item:
${JSON.stringify(item, null, 2)}

${toolContext}

Your previous output:
${JSON.stringify(previousOutput, null, 2)}

Reviewer feedback (issues to correct):
${feedback}

Respond with ONLY the corrected JSON object.`,
    },
  ];

  // No tools passed — Claude cannot call any tools in this pass
  const response = await anthropic.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 4096,
    system: SYSTEM_PROMPT,
    messages,
  });

  const textBlock = response.content.find((c) => c.type === "text");
  const rawText =
    textBlock && textBlock.type === "text" ? textBlock.text.trim() : "{}";

  const revised = parseItemOutput(rawText, item.id);

  // Keep authoritative values from the original triage run
  revised.tools_called = previousOutput.tools_called;
  revised.task_ids = previousOutput.task_ids;
  revised.escalation = previousOutput.escalation ?? revised.escalation;
  revised.requires_human_review = true;

  return revised;
}

// ─── OpenAI review ────────────────────────────────────────────────────────────

interface ReviewResult {
  approved: boolean;
  feedback: string;
}

async function reviewWithOpenAI(
  item: InboxItem,
  output: ItemOutput,
): Promise<ReviewResult> {
  const prompt = `You are a strict quality reviewer for a medical intake triage system at Cedar Kids Therapy.

Check the triage output against these rules and return JSON only.

## Rules
1. SAFEGUARDING: if the item body mentions harm, abuse, neglect, or unsafe caregiving →
   - classification must be "safeguarding", urgency must be "P0"
   - escalation must not be null and severity must be "P0"
2. SAME-DAY RESCHEDULE/CANCELLATION → classification "scheduling", urgency "P1", escalation severity "P1"
3. OUT-OF-NETWORK insurance (Kaiser, Cigna Select, Beacon) →
   - tools_called must include verify_insurance AND lookup_policy
   - tools_called must NOT include hold_slot
   - task_ids must be non-empty (billing task required)
4. CLINICAL QUESTION → classification "clinical_question"; draft_reply must NOT give clinical advice
5. INCOMPLETE REFERRAL → classification "missing_paperwork"; missing_info must list all missing fields
6. SPANISH-SPEAKING → draft_reply should be in Spanish
7. requires_human_review must be true
8. escalation must not be null for P0 or P1 items
9. At least one tool must appear in tools_called
10. task_ids must contain IDs that look like real IDs (starting with "task_"), not empty for referral items

## Inbox item
${JSON.stringify(item, null, 2)}

## Triage output
${JSON.stringify(output, null, 2)}

Respond ONLY with this JSON (no other text):
{"approved": true|false, "feedback": "<if not approved: list specific violations; if approved: empty string>"}`;

  const response = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    temperature: 0,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content:
          "You are a strict quality reviewer. Respond only with valid JSON matching the requested format.",
      },
      { role: "user", content: prompt },
    ],
  });

  const content =
    response.choices[0]?.message?.content ??
    '{"approved":false,"feedback":"No response from reviewer"}';

  try {
    return JSON.parse(content) as ReviewResult;
  } catch {
    return {
      approved: false,
      feedback: `Reviewer response could not be parsed: ${content.slice(0, 200)}`,
    };
  }
}

// ─── Orchestration: triage + review + optional revision ───────────────────────

async function processItemWithReview(item: InboxItem): Promise<ItemOutput> {
  // Phase 1 — full agentic run with tool calls (inside withItemContext for trace)
  let triageRun = await withItemContext(item.id, () =>
    triageWithClaude(item, null),
  );

  let output = triageRun.output;

  for (let attempt = 0; attempt < MAX_REVIEW_RETRIES; attempt++) {
    const review = await reviewWithOpenAI(item, output);
    const status = review.approved ? "✓ approved" : "✗ rejected";
    console.log(`  [${item.id}] review attempt ${attempt + 1}: ${status}`);

    if (review.approved) return output;

    console.log(`  [${item.id}] feedback: ${review.feedback}`);

    if (attempt === 0 && !review.approved) {
      // First rejection: re-run triage with feedback so Claude can also re-use tools
      // (subsequent rejections use revision without new tool calls)
      triageRun = await withItemContext(item.id, () =>
        triageWithClaude(item, review.feedback),
      );
      output = triageRun.output;
    } else {
      // Subsequent rejections: revision only — no new tool calls, keeps trace clean
      output = await reviseWithClaude(
        item,
        output,
        triageRun.toolSummaryLines,
        review.feedback,
      );
    }
  }

  // Final review after last revision
  const finalReview = await reviewWithOpenAI(item, output);
  console.log(
    `  [${item.id}] final review: ${finalReview.approved ? "✓ approved" : "✗ using last output anyway"}`,
  );

  return output;
}

// ─── JSON parser ──────────────────────────────────────────────────────────────

function parseItemOutput(text: string, itemId: string): ItemOutput {
  let jsonStr = text;

  // Strip markdown code fences if present
  const fenceMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) jsonStr = fenceMatch[1].trim();

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
        p.recommended_next_action ?? "Manual review required.",
      draft_reply: p.draft_reply ?? null,
      task_ids: p.task_ids ?? [],
      escalation: p.escalation ?? null,
      decision_rationale:
        p.decision_rationale ?? "Agent output — see tools_called for details.",
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
      recommended_next_action: "Manual review required — could not parse agent output.",
      draft_reply: null,
      task_ids: [],
      escalation: null,
      decision_rationale: `Parse error. Raw output: ${text.slice(0, 300)}`,
    };
  }
}

// ─── Entry point ──────────────────────────────────────────────────────────────

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  console.log(`\nStarting triage for ${inbox.length} inbox items...\n`);
  const results: ItemOutput[] = [];

  for (const item of inbox) {
    console.log(`Processing [${item.id}]: ${item.subject}`);
    const output = await processItemWithReview(item);
    results.push(output);
    console.log(
      `  → classification: ${output.classification} | urgency: ${output.urgency} | tools: ${output.tools_called.length}\n`,
    );
  }

  console.log("Triage complete.\n");
  return results;
}
