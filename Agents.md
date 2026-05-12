# Agents.md

> This document describes the AI agent architecture, components, and design decisions for the OTChallenge project.
> It is a living document — update it whenever the agent structure changes.

---

## Project Overview

**Project Name:** OTChallenge — Cedar Kids Therapy Referral Inbox Triage Agent  
**Status:** In development  
**Last Updated:** 2026-05-12  
**Default Branch:** `main`

Build a single AI agent that reads 8 referral inbox items for a pediatric therapy practice (Cedar Kids Therapy) and produces structured triage output: classification, urgency, intake extraction, tool calls, draft replies, and escalation flags.

---

## Agent Architecture

```
data/inbox.json
      │
      ▼
 src/index.ts              ← CLI entry point; configures trace; calls runAgent()
      │
      ▼
 src/agent.ts — runAgent() ← processes all 8 items sequentially
      │
      └── per item: processItemWithReview(item)
            │
            ├─ Phase 1 ── withItemContext(item.id, triageWithClaude)
            │               │
            │               ├── Claude (claude-haiku-4-5-20251001)
            │               │     Agentic loop: reasons about item,
            │               │     calls tools until satisfied, then
            │               │     emits final ItemOutput JSON
            │               │
            │               └── Tool dispatcher → src/tools.ts
            │                     (calls are recorded in trace automatically)
            │
            ├─ Review ─── OpenAI (gpt-4o-mini)
            │               Checks output against clinic policies.
            │               Returns {approved, feedback}.
            │
            ├─ If rejected (attempt 1): re-run triageWithClaude with feedback
            │               Claude has another chance to call tools and fix decisions
            │
            ├─ If rejected (attempt 2+): reviseWithClaude (no new tool calls)
            │               Claude corrects its JSON using existing tool results
            │               Trace stays clean — tools only called in Phase 1
            │
            └─ After MAX_REVIEW_RETRIES: use last output regardless
      │
      ▼
 buildBatchOutput()        ← assembles BatchOutput with summary stats
      │
      ▼
 output.json + .trace/tool-calls.jsonl
      │
      ▼
 src/validate.ts           ← validates output against schema and business rules
```

### Agent Type

Dual-LLM single-agent pipeline. Claude performs triage with tool use; OpenAI acts as a policy-aware reviewer that can send corrective feedback back to Claude.

---

## Tech Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | TypeScript | Node.js LTS, ESM modules |
| Triage LLM | Claude `claude-haiku-4-5-20251001` | Via `@anthropic-ai/sdk`; agentic loop with tool use |
| Review LLM | OpenAI `gpt-4o-mini` | Via `openai` SDK; policy validation + feedback |
| Tool Calling | Anthropic tool use API | 8 mock tools in `src/tools.ts` |
| Validation | AJV + ajv-formats | Schema at `schema/output.schema.json` |
| ID generation | ulid | For call_ids, task_ids, hold_ids |
| Runtime | tsx | Direct TS execution, no build step needed |

---

## Tools & Capabilities

8 tools are available in `src/tools.ts`. The agent uses **3** — chosen because each is a real decision input, not a threshold filler.

| Tool | Used? | Role in decision process | Items |
|------|-------|--------------------------|-------|
| `verify_insurance` | **Yes** | Result forks the workflow: in-network → intake task; out-of-network → billing task, no slot hold | 1, 3, 4, 7 |
| `escalate` | **Yes** | Required by policy for P0 (safeguarding) and P1 (same-day changes); produces escalation_id | 2, 8 |
| `create_task` | **Yes** | Creates concrete staff follow-up; assignee (billing/intake/clinical_lead/front_desk) reflects triage decision | 1–8 |
| `search_patient` | No | Not selected — patient lookup not needed for first-contact triage of these 8 items | — |
| `lookup_policy` | No | Not selected — policies are encoded in the system prompt; runtime lookups would be performative | — |
| `find_slots` | No | Not selected — out-of-network items block slot holds; no item is fully cleared for hold in this batch | — |
| `hold_slot` | No | Not selected — slot reservation requires billing clearance not yet done for any item | — |
| `draft_message` | No | Not selected — outbound messages are represented in `draft_reply` field; extra tool call would duplicate that | — |

**Forbidden tools (must never appear in output or trace):** `schedule_appointment`, `send_message`

---

## Inbox Items

| ID | Patient | Channel | Key Signal |
|----|---------|---------|-----------|
| item_1 | Emma Lee | fax_referral | SLP referral, complete data, BCBS in-network |
| item_2 | Leo Gomez | voicemail | SLP eval request + **safeguarding signal** (dad "getting rough") → P0 |
| item_3 | Owen Brooks | fax_referral | OT referral, Kaiser HMO (out-of-network) → benefits conversation required |
| item_4 | Mateo Ramirez | email | PT referral, Aetna in-network, existing patient match |
| item_5 | Ava Kim | portal_message | Clinical question about R sounds → must NOT give clinical advice |
| item_6 | Sam Taylor | fax_referral | SLP referral, **incomplete** (missing DOB, guardian, insurance) |
| item_7 | Isabella Lopez | voicemail | SLP referral, Spanish-language preference, Medicaid in-network |
| item_8 | Noah Patel | email | Same-day reschedule request → P1 operational issue |

---

## Output Requirements

Every `ItemOutput` must include:
- `item_id`, `classification`, `urgency` (P0–P3), `requires_human_review: true`
- `extracted_intake`: child_name, dob_or_age, parent_contact, discipline, diagnosis_or_concern, payer, member_id
- `missing_info`: list of absent fields
- `tools_called`: array of tool invocations with call_id, name, args, result_summary
- `recommended_next_action`, `draft_reply` (or null), `task_ids`
- `escalation`: `{reason, severity}` for P0/P1, else null
- `decision_rationale`

**Batch-level constraints:**
- All 8 items must have exactly one output entry
- All items must have `requires_human_review: true`
- At least 3 distinct tool names used across the batch
- Summary stats (p0_count, p1_count, etc.) must match actual items

---

## Clinic Policies (summary)

- **Service lines:** SLP, OT, PT for ages 0–18
- **Insurance in-network:** Aetna, Blue Cross Blue Shield, UnitedHealthcare, Medicaid
- **Insurance out-of-network:** Kaiser, Cigna Select, Beacon → benefits conversation before any slot hold
- **Safeguarding:** Any harm/abuse/neglect signal → P0, escalate to clinical lead immediately
- **Clinical advice:** Front desk / agents must NOT provide clinical advice; route to clinician
- **Scheduling:** Agents may hold slots but must NOT schedule independently
- **Same-day cancellations:** P1 operational issue
- **Language access:** Spanish-speaking staff available (Lucia Morales, Sofia Reyes)

---

## Providers

| Name | Discipline | Languages | Caseload |
|------|-----------|-----------|---------|
| Maya Chen, MS CCC-SLP | SLP | en | accepting |
| Lucia Morales, MA CCC-SLP | SLP | en, es | accepting |
| James Owens, OTR/L | OT | en | limited |
| Priya Shah, PT, DPT | PT | en | accepting |
| Sofia Reyes, OTR/L | OT | en, es | **full** |

---

## Evaluation & Testing

```bash
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run typecheck
```

`src/validate.ts` enforces:
1. All 8 input items have exactly one output entry
2. Summary stats match calculated counts
3. All items have `requires_human_review: true`
4. At least 3 distinct tools used across the batch
5. No forbidden tools (`schedule_appointment`, `send_message`) in output or trace
6. Every tool call in output matches the trace by call_id, name, args, result_summary

---

## Known Constraints & Design Decisions

- Git default branch is `main`
- All tool calls must be made inside `withItemContext(item.id, fn)` — otherwise `recordTool` throws
- `hold_slot` always returns `pending_review` — slot confirmation requires human action
- `draft_message` never sends — output is always `{status: "draft"}`
- Out-of-network payers require `lookup_policy({topic: "insurance"})` + billing task before any `hold_slot`
- Safeguarding items (item_2) must call `escalate` with P0 severity

---

## Setup & Running

```bash
npm install

# Copy and fill in API keys
cp .env.example .env

# Run triage
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl

# Validate output
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl

# Type check
npm run typecheck
```

Required environment variables (set in `.env`):
```
ANTHROPIC_API_KEY=sk-ant-...   # Claude haiku for triage
OPENAI_API_KEY=sk-...          # GPT-4o-mini for review
```
