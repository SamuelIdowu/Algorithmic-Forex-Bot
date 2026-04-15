#!/usr/bin/env python3
"""
CLI Chat Interface — Phase 11
Direct consultation with the AI Fund Manager via the terminal.
"""
import sys
import os

# Add current dir to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.chat_agent import ChatAgent

def main():
    agent = ChatAgent()
    history = []

    print("\n" + "="*60)
    print("🤖  AI HEDGE FUND — INSTITUTIONAL CONSULTATION (CLI)")
    print("="*60)
    print("Ask about recent trades, market analysis, or trading strategy.")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("👋 Assistant: Goodbye, Fund Manager.")
                break

            print("\n⚙️  Consulting LLM...")
            
            # The ChatAgent handles DB context fetching automatically
            response = agent.ask(user_input, history=history)
            
            print(f"\n🤖 Assistant: {response}\n")
            print("-" * 40)

            # Update session history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\n👋 Assistant: Goodbye.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
