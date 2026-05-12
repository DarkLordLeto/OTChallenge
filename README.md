# OTChallenge

Cedar Kids Therapy referral inbox triage agent.

This project implements a single-agent triage workflow for 8 inbound referral and scheduling items. The agent reads each item, calls mock clinic tools, produces structured output, and then runs a second-pass policy review before writing `output.json`.

## Question 1. How do I run the project?

Prerequisites:
- Node.js LTS
- npm
- `ANTHROPIC_API_KEY` in the environment

Install dependencies:

```bash
npm install
```

Set the API key:

```bash
set ANTHROPIC_API_KEY=sk-ant-...
```

Run triage:

```bash
npm run triage -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
```

Validate the output:

```bash
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
```

Type-check:

```bash
npm run typecheck
```

Notes:
- In this Windows environment, `npm.cmd` may be required instead of `npm`.
- The trace file is part of the contract. Validation checks that `output.json` and `.trace/tool-calls.jsonl` match.

## Question 2. What stack and runtime does this use?

- Language: TypeScript
- Runtime: Node.js with ESM
- Execution: `tsx` for direct TS execution
- LLM SDK: `@anthropic-ai/sdk`
- Validation: `ajv` + `ajv-formats`
- IDs: `ulid`

Current runtime model setup:
- Action model: Claude `claude-haiku-4-5-20251001`
- Review model: Claude `claude-haiku-4-5-20251001`

The repo still contains the `openai` dependency from an earlier iteration, but the current implementation path is Anthropic-only for both action and review.

## Question 3. What is the architecture, including the action-review design idea and its concerns?

High-level flow:

```text
data/inbox.json
  -> src/index.ts
  -> runAgent(inbox)
  -> processItem(item)
     -> triageWithClaude()
        -> tool calls inside withItemContext(item.id, ...)
     -> reviewWithClaude()
     -> optional reviseOutput()
  -> buildBatchOutput()
  -> output.json
  -> .trace/tool-calls.jsonl
```

Core files:
- [src/index.ts](/E:/Work/OTChallenge/src/index.ts): CLI entry point
- [src/agent.ts](/E:/Work/OTChallenge/src/agent.ts): action loop, review loop, revision loop
- [src/tools.ts](/E:/Work/OTChallenge/src/tools.ts): mock tools and trace recording
- [src/validate.ts](/E:/Work/OTChallenge/src/validate.ts): schema and business-rule validation
- [schema/output.schema.json](/E:/Work/OTChallenge/schema/output.schema.json): output schema

Action-review design:
- Pass 1 is the action pass. Claude reads an inbox item, calls tools, and emits JSON.
- Pass 2 is the review pass. A second Claude call checks the item output against policy rules.
- If review rejects the output, the agent performs a revision pass using existing tool results only.

Why this design:
- It separates "do the work" from "criticize the work".
- It gives the system a chance to catch policy misses without re-running the whole batch validator.
- It keeps tool usage auditable, because revision does not introduce new tool calls.

Concerns with this design:
- The reviewer can contradict itself across attempts.
- The reviewer can overfit to prompt wording instead of intended policy logic.
- The system is slower and more expensive than a single-pass design.
- Action-review only helps if the reviewer is more stable than the actor. Right now that is not consistently true.

## Question 4. What are the main failure modes, and how would I evaluate this in production?

Observed and likely failure modes:
- Reviewer contradiction: the reviewer sometimes flags P2 items for missing escalation even though escalation is only required for P0/P1.
- Reviewer formatting drift: fenced JSON or verbose explanations can still break the strict parse path.
- Prompt-rule mismatch: reviewer language can drift from actual business intent.
- Over-tooling: the agent may call more tools than necessary, especially when policy lookup is overused.
- Under-extraction: missing intake fields can be incompletely enumerated.
- Communication quality drift: `draft_message` output may be polite but weakly grounded in the actual situation.
- Scheduling flow mistakes: patient lookup, slot hold, and task notes must stay internally consistent.

What I would evaluate in production:
- Per-item accuracy by classification, urgency, and escalation correctness
- Tool precision: whether the right tools were called, not just enough tools
- Trace/output consistency rate
- Reviewer agreement rate with human auditors
- False positive and false negative rates for safeguarding and same-day scheduling
- Message quality review for neutrality, no clinical advice, and language correctness
- Latency and token cost per item

Production eval plan:
- Create a fixed regression set of inbox items with expected outputs and expected tool traces
- Add adversarial items for borderline safeguarding, incomplete referrals, and Spanish-language cases
- Score reviewer disagreement separately from action-model mistakes
- Track "approved by reviewer but failed validator" and "rejected by reviewer but passed validator" as first-class metrics

## Question 5. What did I choose not to build, and why?

- I did not build a real persistence layer.
  The assignment already defines deterministic JSON inputs and outputs, so a database would add surface area without helping correctness much.

- I did not build a real external scheduler or messaging integration.
  The assignment explicitly forbids autonomous scheduling and sending. Mock `hold_slot` and `draft_message` are the right boundary.

- I did not build a UI.
  The critical work here is policy-constrained agent behavior and traceability, not operator workflow polish.

- I did not build a separate human-review queue service.
  `create_task`, `escalate`, and the structured output already encode the handoff points.

- I did not fully normalize prompt and policy definitions into a single shared source.
  Right now some rules live in prompts, some in review rules, and some in validator logic. That was acceptable for iteration speed, but it is also one cause of contradictions.

## Question 6. What would I do with another 4 hours?

- Resolve reviewer contradiction feedback first.
  I would move reviewer rules into a stricter, shorter rubric that mirrors `src/validate.ts` exactly and remove ambiguous language around escalation for P2/P3 items.

- Add a shared rule source.
  I would define policy rules once and generate both reviewer instructions and test assertions from that source.

- Add structured reviewer output validation.
  I would require a machine-friendly format like `{approved, violations: [{rule_id, message}]}` and reject freeform prose.

- Add more tool post-processing guards.
  Examples:
  `assertNoHoldForOON()`, `assertEscalationForP0P1()`, `assertDraftForCommunications()`, `assertPatientRefUsesPatientIdWhenFound()`.

- Add new helper functions around the existing tools.
  Not new external tools first, but orchestration helpers such as:
  `pickPolicyTopic(item)`, `pickDraftChannel(item)`, `extractDraftReplyFromTrace(itemId)`, `buildTaskNotesFromPolicy(...)`, `buildRecommendedNextAction(...)`.

- If time remained, add one or two new tools only where they reduce prompt ambiguity.
  The strongest candidates would be a dedicated `extract_intake` helper or a `classify_item` helper, because those are repeated reasoning tasks and currently depend entirely on prompt behavior.

- Add a reproducible evaluation harness.
  I would save a gold set of expected outputs and compare runs across prompt changes, not just rely on pass/fail validation.

## Question 7. What tool call choices did I make, and why?

The current implementation uses 8 active tools. 
1. `verify_insurance`
   This is the main workflow fork for referral items. In-network can proceed toward intake; out-of-network must route to billing and avoid slot holds.

2. `lookup_policy`
   This makes policy grounding explicit before writing tasks and family-facing drafts. It reduces the chance that the agent invents clinic policy from prompt memory alone.

3. `escalate`
   This is required for the high-risk paths: safeguarding and same-day scheduling issues. It creates an explicit human-oversight event instead of burying urgency in prose.

4. `create_task`
   This is the concrete operational handoff. The output is not useful in practice unless someone specific owns the next action.

5. `search_patient`
   This is used in the scheduling path to anchor the hold to a real patient record. When a patient match exists, `hold_slot.patient_ref` uses `patient_id` instead of a raw name string.

6. `find_slots`
   This gives the scheduling path a real next step instead of only abstract advice. It makes the reschedule recommendation actionable for front desk staff.

7. `hold_slot`
   This lets the agent reserve an option without violating the "do not schedule autonomously" constraint. `pending_review` is the right compromise.

8. `draft_message`
   This captures the outbound communication artifact without sending it. Its `body` becomes the authoritative `draft_reply` in the structured output.

What I intentionally did not prioritize:
- `schedule_appointment`
  Forbidden by the assignment because autonomous booking is out of scope.

- `send_message`
  Also forbidden. Drafting is allowed; sending is not.

## Current State

- The batch currently runs end to end.
- `triage`, `validate`, and `typecheck` all run successfully in the current setup.
- The validator is more trustworthy than the reviewer right now.
- The biggest remaining engineering problem is reviewer stability, not raw batch generation.

## Additional Note

I used both Claude Code and Codex during this project.

- Claude Code was useful for iterative agent design, prompt/tool workflow changes, and fast experimentation.
- Codex was useful for implementation cleanup, documentation updates, verification runs, and catching consistency issues between code and docs. (I mainly used it for testing and also draft this README file for better illustrating my ideas)

For more implementation detail, see [Agents.md](/E:/Work/OTChallenge/Agents.md).
