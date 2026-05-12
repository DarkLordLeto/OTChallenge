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
 src/index.ts        ← CLI entry point, configures trace, calls runAgent()
      │
      ▼
 src/agent.ts        ← YOUR IMPLEMENTATION: runAgent(inbox) → ItemOutput[]
      │
      ├── withItemContext(item.id, ...)   ← wraps each item for trace association
      │
      └── Tool calls (src/tools.ts)
            ├── search_patient
            ├── verify_insurance
            ├── lookup_policy
            ├── find_slots
            ├── hold_slot
            ├── create_task
            ├── draft_message
            └── escalate
      │
      ▼
 buildBatchOutput()  ← wraps items into BatchOutput with summary stats
      │
      ▼
 output.json + .trace/tool-calls.jsonl
      │
      ▼
 src/validate.ts     ← validates output against schema and business rules
```

### Agent Type

Single agent — no multi-agent orchestration. The agent processes all 8 inbox items sequentially (or in parallel), calling tools per item inside `withItemContext()` to maintain trace association.

---

## Tech Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Language | TypeScript | Node.js LTS, ESM modules |
| LLM | Claude (Anthropic) | Via `@anthropic-ai/sdk` |
| Tool Calling | Anthropic tool use API | 8 mock tools in `src/tools.ts` |
| Validation | AJV + ajv-formats | Schema at `schema/output.schema.json` |
| ID generation | ulid | For call_ids, task_ids, hold_ids |
| Runtime | tsx | Direct TS execution, no build step needed |

---

## Tools & Capabilities

| Tool | Purpose | Assignee / Notes |
|------|---------|-----------------|
| `search_patient` | Look up existing patients by name/DOB | Returns active/inactive status |
| `verify_insurance` | Check payer network status | in_network / out_of_network / expired / unknown |
| `lookup_policy` | Retrieve clinic policy snippets by topic | 7 topics available |
| `find_slots` | Search available appointment slots | Filters by discipline, language |
| `hold_slot` | Reserve a slot pending human review | Status always `pending_review` — NOT a confirmed booking |
| `create_task` | Log action items for staff | Assignees: front_desk, intake, billing, clinical_lead |
| `draft_message` | Compose outbound messages | Stays in draft state — never sent automatically |
| `escalate` | Flag items for immediate human oversight | P0 or P1 severity |

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
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
```

Requires `ANTHROPIC_API_KEY` environment variable set for the agent LLM calls.
