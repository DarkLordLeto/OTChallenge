# Agents.md

> This document describes the AI agent architecture, components, and design decisions for the OTChallenge project.
> It is a living document - update it whenever the agent structure changes.

---

## Project Overview

**Project Name:** OTChallenge - Cedar Kids Therapy Referral Inbox Triage Agent  
**Status:** In development  
**Last Updated:** 2026-05-12  
**Default Branch:** `main`

Build a single AI agent that reads 8 referral inbox items for a pediatric therapy practice (Cedar Kids Therapy) and produces structured triage output: classification, urgency, intake extraction, tool calls, draft replies, and escalation flags.

---

## Agent Architecture

```text
data/inbox.json
  -> src/index.ts
     CLI entry point; configures trace; calls runAgent()

  -> src/agent.ts
     runAgent() processes all 8 items sequentially

  -> per item: processItem(item)
     -> Phase 1: withItemContext(item.id, triageWithClaude)
        -> Claude `claude-haiku-4-5-20251001`
           Runs the agentic loop, calls tools, and emits ItemOutput JSON
        -> Tool dispatcher -> src/tools.ts
           Tool calls are recorded in the trace automatically

     -> Review: reviewWithClaude(item, output)
        -> Claude `claude-haiku-4-5-20251001`
           Checks output against clinic policies and returns
           `{approved, feedback}`

     -> If rejected: reviseOutput(item, previous, toolLog, feedback)
        Uses existing tool results only; no new tool calls in revision

     -> After `MAX_REVISIONS`: use the last output regardless

  -> buildBatchOutput()
     Assembles BatchOutput with summary stats

  -> output.json + .trace/tool-calls.jsonl

  -> src/validate.ts
     Validates output against schema and business rules
```

### Agent Type

Single-provider dual-pass pipeline. Claude performs the triage pass with tool use, then a second Claude pass reviews the output against policy and can send corrective feedback back to the revision loop.

---

## Tech Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | TypeScript | Node.js LTS, ESM modules |
| Triage LLM | Claude `claude-haiku-4-5-20251001` | Via `@anthropic-ai/sdk`; agentic loop with tool use |
| Review LLM | Claude `claude-haiku-4-5-20251001` | Temporary test configuration; second pass for policy validation + feedback |
| Tool Calling | Anthropic tool use API | 8 mock tools in `src/tools.ts` |
| Validation | AJV + ajv-formats | Schema at `schema/output.schema.json` |
| ID generation | ulid | For call_ids, task_ids, hold_ids |
| Runtime | tsx | Direct TS execution, no build step needed |

---

## Tools & Capabilities

8 tools are available in `src/tools.ts`. The current implementation uses 5 and leaves 3 unused because they are not required for the present triage strategy.

| Tool | Used? | Role in decision process | Items |
|------|-------|--------------------------|-------|
| `verify_insurance` | Yes | Result forks the workflow: in-network to intake task; out-of-network to billing task, no slot hold | 1, 3, 4, 7 |
| `escalate` | Yes | Required by policy for P0 safeguarding and P1 same-day changes | 2, 8 |
| `create_task` | Yes | Creates concrete staff follow-up; assignee reflects triage decision | 1-8 |
| `find_slots` | Yes | Called for reschedule flow after `escalate`; surfaces available options | 8 |
| `hold_slot` | Yes | Called after `find_slots`; places earliest slot in `pending_review` | 8 |
| `draft_message` | Yes | Composes outbound reply as a draft (never auto-sent); `body` arg becomes `draft_reply` in output | 1–8 |
| `search_patient` | No | Not selected for the current implementation | - |
| `lookup_policy` | No | Not selected; policy logic is encoded in prompts/review rules | - |

### Reschedule Workflow

For item 8 (`Noah Patel`, same-day OT cancellation):

```text
1. escalate(item_id="item_8", reason="Same-day cancellation request", severity="P1")
2. find_slots(discipline="OT")
3. hold_slot(slot_id=<earliest>, patient_ref="Noah Patel")
4. create_task(assignee="front_desk", notes include hold_id and contact info)
5. draft_message(recipient, channel, body mentioning hold and next steps)
```

`hold_slot` is always `pending_review`; the agent does not confirm appointments autonomously.
`draft_message` is always `status: "draft"`; the agent never sends messages automatically.

**Forbidden tools:** `schedule_appointment`, `send_message`

---

## Inbox Items

| ID | Patient | Channel | Key Signal |
|----|---------|---------|-----------|
| item_1 | Emma Lee | fax_referral | SLP referral, complete data, BCBS in-network |
| item_2 | Leo Gomez | voicemail | SLP eval request plus safeguarding signal ("dad getting rough") -> P0 |
| item_3 | Owen Brooks | fax_referral | OT referral, Kaiser HMO out-of-network |
| item_4 | Mateo Ramirez | email | PT referral, Aetna in-network |
| item_5 | Ava Kim | portal_message | Clinical question; must not provide clinical advice |
| item_6 | Sam Taylor | fax_referral | Incomplete SLP referral |
| item_7 | Isabella Lopez | voicemail | SLP referral, Spanish preference, Medicaid in-network |
| item_8 | Noah Patel | email | Same-day reschedule request -> P1 |

---

## Output Requirements

Every `ItemOutput` must include:
- `item_id`, `classification`, `urgency` (`P0`-`P3`), `requires_human_review: true`
- `extracted_intake`: child_name, dob_or_age, parent_contact, discipline, diagnosis_or_concern, payer, member_id
- `missing_info`
- `tools_called`
- `recommended_next_action`
- `draft_reply` or `null`
- `task_ids`
- `escalation` for P0/P1, else `null`
- `decision_rationale`

Batch-level constraints:
- All 8 items must have exactly one output entry
- All items must have `requires_human_review: true`
- At least 3 distinct tool names must be used across the batch
- Summary stats must match actual item counts

---

## Clinic Policies

- Service lines: SLP, OT, PT for ages 0-8
- In-network: Aetna, Blue Cross Blue Shield, UnitedHealthcare, Medicaid
- Out-of-network: Kaiser, Cigna Select, Beacon
- Safeguarding: any harm/abuse/neglect signal -> P0, escalate immediately
- Clinical advice: intake/agent must not give clinical advice
- Scheduling: agent may hold slots but may not schedule independently
- Same-day cancellations: P1 operational issue
- Language access: Spanish-speaking staff available

---

## Providers

| Name | Discipline | Languages | Caseload |
|------|------------|-----------|----------|
| Maya Chen, MS CCC-SLP | SLP | en | accepting |
| Lucia Morales, MA CCC-SLP | SLP | en, es | accepting |
| James Owens, OTR/L | OT | en | limited |
| Priya Shah, PT, DPT | PT | en | accepting |
| Sofia Reyes, OTR/L | OT | en, es | full |

---

## Evaluation & Testing

```bash
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run typecheck
```

Latest local run on 2026-05-12:
- `npm.cmd run typecheck` passed
- `npm.cmd run triage -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl` completed successfully
- `npm.cmd run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl` passed
- Generated batch stats: `total_items=8`, `p0_count=1`, `p1_count=1`, `requires_human_review_count=8`

`src/validate.ts` enforces:
1. All 8 input items have exactly one output entry
2. Summary stats match calculated counts
3. All items have `requires_human_review: true`
4. At least 3 distinct tools used across the batch
5. No forbidden tools (`schedule_appointment`, `send_message`) in output or trace
6. Every tool call in output matches the trace by `call_id`, `name`, `args`, and `result_summary`

---

## Known Constraints & Design Decisions

- Git default branch is `main`
- All tool calls must be made inside `withItemContext(item.id, fn)` or `recordTool` throws
- `hold_slot` always returns `pending_review`; slot confirmation requires human action
- The current implementation uses Anthropic for both triage and review during testing
- Out-of-network items currently rely on `verify_insurance` plus billing follow-up; `lookup_policy` is not used in the current implementation
- Safeguarding items (item 2) must call `escalate` with `P0`
- Same-day reschedule items (item 8) must call `escalate`, `find_slots`, `hold_slot`, and `create_task`

Known problem from the latest live run:
- The review parser is too brittle for Anthropic text output
- The reviewer often returns fenced ```json blocks
- Those fenced responses are currently treated as parse failures in the review loop
- Result: logs show false rejections such as `Reviewer response unparseable` even when the underlying JSON is valid
- Despite that issue, the generated `output.json` and `.trace/tool-calls.jsonl` still passed `src/validate.ts`
- This means the current bug is in review-response parsing, not in schema generation, trace matching, or batch validation

---

## Setup & Running

```bash
npm install

# Set API key in environment
set ANTHROPIC_API_KEY=sk-ant-...

# Run triage
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl

# Validate output
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl

# Type check
npm run typecheck
```

Required environment variables:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```
