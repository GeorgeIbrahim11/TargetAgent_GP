# DECISIONS.md - Architecture and Design Decisions

## Model Choice

**Selected Model:** `openai/gpt-oss-120b` through Groq

**Configuration:** The model is loaded from `LLM_MODEL` in `.env`.

**Reasons:**
- Free Groq API tier for this project.
- Fast response times for iterative red-team testing.
- Good tool-calling support with structured JSON output.
- Sufficient instruction-following to respect the banking persona and rules.

**Date:** 2026-09-28

---

## Environment Configuration

- `GROQ_API_KEY` stores the Groq API key.
- `LLM_MODEL` selects the Groq model.
- `TARGET_CANARY` sets the synthetic canary string.
- `LLM_COST_LOG` controls cost logging. Groq cost is recorded as zero.

Copy `.env.example` to `.env` and fill in the API key.

---

## Canary String Design

**Canary:** Loaded from `TARGET_CANARY`.

**Rationale:** Looks like a plausible internal operations code and is unique enough to detect if leaked. It is embedded in the system prompt as an internal operations code with instructions not to share.

---

## Guardrail Architecture

**Decision:** Guardrails are enforced in Python code, NOT only in the prompt.

**Rationale:** Prompt-only guardrails can be bypassed by prompt injection. Code guardrails cannot be bypassed by the LLM because the validation runs before and after the tool call, outside the model's control.

**Three layers:**
1. **Input validator** - scans user message for injection patterns before sending to the LLM.
2. **Pre-tool validator** - checks transfer amount (<= $500) and recipient whitelist before executing.
3. **Output validator** - scans model response for the canary string before returning to the user.

---

## Tool Design

**Decision:** 4 tools implemented as plain Python functions with synthetic data.

- `get_balance(account_id)` - read-only, low risk
- `transfer_money(to, amount)` - write operation, guarded
- `request_refund(tx_id)` - write operation, moderate risk
- `get_transactions(account_id)` - read-only, low risk

All tools log their calls (function name + full arguments + result) for the Judge to verify.

---

## Conversation Manager

**Decision:** Simple list-based message history. One session per TargetAgent instance. Call `reset_conversation()` to start a new session.

**Rationale:** Multi-turn attacks require conversation context. Keeping it simple with a Python list of message dictionaries.
