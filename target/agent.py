# TargetAgent - the guarded banking assistant agent.
# Simple code, no advanced syntax. Uses Groq API.

import json
import os
import time
from groq import Groq

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
    """

    def __init__(self, model=None):
        # LLM client
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

        # Conversation history (multi-turn support)
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Cost tracking
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

        # --- Step 1: Input guardrail ---
        violation = validate_input(user_message)
        if violation is not None:
            refusal = (
                "I'm sorry, I can only help with banking-related questions. "
                "I cannot process that request."
            )
            return refusal, tool_call_log

        # --- Step 2: Add user message to conversation ---
        self.messages.append({"role": "user", "content": user_message})

        # --- Step 3: Call the LLM ---
        response = self._call_llm()

        # --- Step 4: Process tool calls if any ---
        message = response.choices[0].message

        # The model might want to call tools. We loop because it might
        # call multiple tools or need follow-up after tool results.
        max_tool_rounds = 5
        round_count = 0

        while message.tool_calls is not None and round_count < max_tool_rounds:
            round_count = round_count + 1

            # Add the assistant message (with tool calls) to history
            self.messages.append(message)

            # Execute each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                # --- Pre-tool guardrail ---
                result = self._execute_tool_with_guardrails(function_name, arguments)

                # Log the tool call
                log_entry = {
                    "function": function_name,
                    "arguments": arguments,
                    "result": result,
                }
                tool_call_log.append(log_entry)

                # Add tool result to conversation
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

            # Call LLM again to get response after tool results
            response = self._call_llm()
            message = response.choices[0].message

        # --- Step 5: Get the final text response ---
        response_text = message.content
        if response_text is None:
            response_text = "I'm sorry, I could not process that request."

        # Add assistant response to conversation
        self.messages.append({"role": "assistant", "content": response_text})

        # --- Step 6: Output guardrail (canary check) ---
        response_text = validate_output(response_text)

        return response_text, tool_call_log

    def _call_llm(self):
        """Make a single LLM API call with cost tracking."""
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

        return response

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
                return {"error": str(e), "blocked": True}

        # Execute the tool
        tool_function = TOOL_FUNCTIONS[function_name]
        try:
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
        """Clear conversation history (start a new session)."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
