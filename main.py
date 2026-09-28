# Interactive runner for the target agent.
# Usage: python main.py
# Requires GROQ_API_KEY environment variable.

import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from target.agent import TargetAgent


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "replace_with_your_groq_api_key":
        print("ERROR: Set the GROQ_API_KEY environment variable first.")
        print("  Windows:  set GROQ_API_KEY=gsk-...")
        print("  Linux:    export GROQ_API_KEY=gsk-...")
        sys.exit(1)

    print("=" * 50)
    print("  SecureBank Assistant (Target Agent)")
    print("=" * 50)
    print("Type your messages below. Type 'quit' to exit.")
    print("Type 'reset' to start a new conversation.")
    print("Type 'cost' to see the cost summary.")
    print("")

    agent = TargetAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input == "":
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "reset":
            agent.reset_conversation()
            print("Conversation reset.\n")
            continue

        if user_input.lower() == "cost":
            cost = agent.get_cost_summary()
            print("--- Cost Summary ---")
            print("API calls:", cost["num_api_calls"])
            print("Input tokens:", cost["total_input_tokens"])
            print("Output tokens:", cost["total_output_tokens"])
            print("Total cost: $" + str(cost["total_cost_usd"]))
            print("")
            continue

        # Send message to agent
        response, tool_call_log = agent.handle_message(user_input)

        # Show tool calls if any
        if len(tool_call_log) > 0:
            print("\n[Tool Calls]")
            for entry in tool_call_log:
                print("  " + entry["function"] + "(" + json.dumps(entry["arguments"]) + ")")
                print("  -> " + json.dumps(entry["result"]))

        print("\nAssistant:", response)
        print("")


if __name__ == "__main__":
    main()
