import os
import sys
from dotenv import load_dotenv
from agent_init import get_gemini_agent, get_claude_agent

def test_gemini():
    print("\n--- Testing Gemini Agent ---")
    try:
        agent = get_gemini_agent()
        response = agent.run("What is the current version of the smolagents library?")
        print(f"Gemini Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Gemini Test Failed: {e}")
        return False

def test_claude():
    print("\n--- Testing Claude Agent ---")
    try:
        agent = get_claude_agent()
        response = agent.run("Explain the core philosophy of smolagents in one sentence.")
        print(f"Claude Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Claude Test Failed: {e}")
        return False

def test_skills_integration():
    print("\n--- Testing Skills Integration (Synthetic Prompt) ---")
    try:
        agent = get_gemini_agent()
        # Explicitly ask to use the tool
        response = agent.run("Use the 'generate_synthetic_prompt' tool to create a prompt for 'A futuristic cyberpunk city in the rain'.")
        print(f"Integration Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Skills Integration Test Failed: {e}")
        return False

if __name__ == "__main__":
    load_dotenv()
    
    # Check for keys
    missing = []
    if not os.getenv("GEMINI_API_KEY"): missing.append("GEMINI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY"): missing.append("ANTHROPIC_API_KEY")
    if not os.getenv("HF_TOKEN"): missing.append("HF_TOKEN")
    
    if missing:
        print(f"⚠️  Missing API keys: {', '.join(missing)}")
        print("Please add them to the .env file before running tests.")
        # We don't exit here so individual tests can still be attempted if some keys are present
    
    # Optional: Run tests based on available keys
    if os.getenv("GEMINI_API_KEY"):
        test_gemini()
        if os.getenv("HF_TOKEN"):
             test_skills_integration()
             
    if os.getenv("ANTHROPIC_API_KEY"):
        test_claude()
