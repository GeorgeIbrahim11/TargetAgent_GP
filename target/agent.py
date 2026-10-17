# TargetAgent - the guarded banking assistant agent.
# Simple code, no advanced syntax. Uses the Groq API, or an LLM client passed in (see llm).

import json
import os
import time

from target.prompt import SYSTEM_PROMPT, CANARY_STRING
from target.tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS
from target.guardrails import (
    GuardrailViolation,
    validate_input,
    validate_transfer,
    validate_output,
)


class TargetAgent:
    """
    A guarded banking assistant agent.

    Usage:
        agent = TargetAgent()
        response, tool_call_log = agent.handle_message("What is my balance?")

    After each handle_message() call:
        agent.last_guardrails     which guardrails fired, in order ("input", "pre_tool",
                                  "output"). An empty list means none fired.
        agent.last_raw_response   the model's reply before the output guardrail
                                  (None if the input guardrail stopped the message).
    """

    def __init__(self, model=None, llm=None):
        # LLM backend.
        # llm is optional. If given, it needs a chat() method like the red-team pipeline's
        # LLMClient:  llm.chat("target", messages, tools=..., tool_choice=..., attempt_id=...)
        # returning an object with .text, .tool_calls and .assistant_message.
        # The pipeline then controls the model, sampling and cost logging.
        # If llm is None we talk to Groq directly, exactly as before.
        self.llm = llm
        self.client = None
        if llm is None:
            from groq import Groq  # imported here so the injected mode does not need groq

            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

        # Conversation history (multi-turn support)
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Per-agent tool state, so one attempt cannot leak into the next
        self.refund_requests = {}

        # Which attempt is running (set by the pipeline, used to trace LLM calls)
        self.attempt_id = None

        # What the guardrails did during the last handle_message() call
        self.last_guardrails = []
        self.last_raw_response = None

        # Cost tracking (only used when talking to Groq directly)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.call_log = []  # list of per-call cost records

    def handle_message(self, user_message):
        """
        Process a user message and return (response_text, tool_call_log).

        tool_call_log is a list of dicts:
            {"function": str, "arguments": dict, "result": any}
        """
        tool_call_log = []
        self.last_guardrails = []
        self.last_raw_response = None

        # --- Step 1: Input guardrail ---
        violation = validate_input(user_message)
        if violation is not None:
            self.last_guardrails.append("input")
            refusal = (
                "I'm sorry, I can only help with banking-related questions. "
                "I cannot process that request."
            )
            return refusal, tool_call_log

        # --- Step 2: Add user message to conversation ---
        self.messages.append({"role": "user", "content": user_message})

        # --- Step 3: Call the LLM ---
        reply = self._call_llm()

        # --- Step 4: Process tool calls if any ---
        # The model might want to call tools. We loop because it might
        # call multiple tools or need follow-up after tool results.
        max_tool_rounds = 5
        round_count = 0

        while reply["tool_calls"] and round_count < max_tool_rounds:
            round_count = round_count + 1

            # Add the assistant message (with tool calls) to history
            self.messages.append(reply["message"])

            # Execute each tool call
            for call in reply["tool_calls"]:
                # --- Pre-tool guardrail ---
                result = self._execute_tool_with_guardrails(call["name"], call["arguments"])

                # Log the tool call
                log_entry = {
                    "function": call["name"],
                    "arguments": call["arguments"],
                    "result": result,
                }
                tool_call_log.append(log_entry)

                # Add tool result to conversation
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result),
                })

            # Call LLM again to get response after tool results
            reply = self._call_llm()

        # --- Step 5: Get the final text response ---
        response_text = reply["text"]
        if response_text is None:
            response_text = "I'm sorry, I could not process that request."

        # Add assistant response to conversation
        self.messages.append({"role": "assistant", "content": response_text})

        # --- Step 6: Output guardrail (canary check) ---
        self.last_raw_response = response_text
        safe_text = validate_output(response_text)
        if safe_text != response_text:
            self.last_guardrails.append("output")

        return safe_text, tool_call_log

    def _call_llm(self):
        """
        Make a single LLM call. Returns a dict:
            {"text": str or None,
             "tool_calls": [{"id": str, "name": str, "arguments": dict}, ...],
             "message": the assistant message (a dict) to add to the history}
        Both backends return this same shape, so the rest of the agent does not care
        which one is in use.
        """
        if self.llm is not None:
            return self._call_injected_llm()
        return self._call_groq()

    def _call_injected_llm(self):
        """Call the LLM client we were given. It does its own cost logging."""
        result = self.llm.chat(
            "target",
            self.messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            attempt_id=self.attempt_id,
        )

        # result.assistant_message carries the tool-call ids, in the same order as
        # result.tool_calls (which hold the parsed names and arguments).
        wire_calls = result.assistant_message.get("tool_calls", [])
        tool_calls = []
        for wire, call in zip(wire_calls, result.tool_calls):
            tool_calls.append({
                "id": wire["id"],
                "name": call.name,
                "arguments": call.arguments,
            })

        text = result.text
        if text == "":
            text = None

        return {"text": text, "tool_calls": tool_calls, "message": result.assistant_message}

    def _call_groq(self):
        """Make a single Groq API call with cost tracking."""
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )

        end_time = time.time()

        # Track tokens and cost
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Groq is free for this project, so API cost is tracked as zero.
        call_cost = 0.0
        if os.getenv("LLM_COST_LOG", "0").lower() not in ["0", "false", "no", "off"]:
            print("LLM cost: $" + str(call_cost))

        self.total_input_tokens = self.total_input_tokens + input_tokens
        self.total_output_tokens = self.total_output_tokens + output_tokens
        self.total_cost_usd = self.total_cost_usd + call_cost

        cost_record = {
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(call_cost, 6),
            "duration_sec": round(end_time - start_time, 2),
        }
        self.call_log.append(cost_record)

        # Turn the SDK message into plain data
        message = response.choices[0].message
        tool_calls = []
        wire_calls = []
        if message.tool_calls is not None:
            for tool_call in message.tool_calls:
                raw_arguments = tool_call.function.arguments
                try:
                    arguments = json.loads(raw_arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                tool_calls.append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": arguments,
                })
                wire_calls.append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": raw_arguments,
                    },
                })

        history_message = {"role": "assistant", "content": message.content}
        if wire_calls:
            history_message["tool_calls"] = wire_calls

        return {"text": message.content, "tool_calls": tool_calls, "message": history_message}

    def _execute_tool_with_guardrails(self, function_name, arguments):
        """
        Run a tool function after applying pre-tool guardrails.
        Returns the result dict (or an error dict if blocked).
        """
        # Check if the tool exists
        if function_name not in TOOL_FUNCTIONS:
            return {"error": "Unknown tool: " + function_name}

        # Pre-tool guardrails for transfer_money
        if function_name == "transfer_money":
            to = arguments.get("to", "")
            amount = arguments.get("amount", 0)
            try:
                validate_transfer(to, amount)
            except GuardrailViolation as e:
                self.last_guardrails.append("pre_tool")
                return {"error": str(e), "blocked": True}

        # Execute the tool
        tool_function = TOOL_FUNCTIONS[function_name]
        try:
            if function_name == "request_refund":
                # Each agent keeps its own refund requests (cleared on reset_conversation)
                result = tool_function(refunds=self.refund_requests, **arguments)
            else:
                result = tool_function(**arguments)
        except Exception as e:
            result = {"error": "Tool execution failed: " + str(e)}

        return result

    def get_cost_summary(self):
        """Return a summary of all API costs so far."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "num_api_calls": len(self.call_log),
            "per_call_log": self.call_log,
        }

    def reset_conversation(self):
        """Clear conversation history and tool state (start a new session)."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.refund_requests = {}
        self.last_guardrails = []
        self.last_raw_response = None