import os
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel
from skills_bridge import generate_synthetic_prompt

def get_agent(provider="gemini", tools=None, model_id=None):
    """
    Initialize a CodeAgent using the specified provider.
    
    Args:
        provider: "gemini" or "claude"
        tools: List of tools to add to the agent.
        model_id: Specific model ID for the provider.
    """
    load_dotenv()
    
    # Default tools if none provided
    if tools is None:
        tools = [generate_synthetic_prompt]
    
    if provider == "gemini":
        model_id = model_id or "gemini/gemini-1.5-flash"
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        model = LiteLLMModel(model_id=model_id, api_key=api_key)
        
    elif provider == "claude":
        model_id = model_id or "anthropic/claude-3-5-sonnet-20240620"
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment.")
        model = LiteLLMModel(model_id=model_id, api_key=api_key)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
        
    return CodeAgent(tools=tools, model=model)

# Convenience functions
def get_gemini_agent(tools=None, model_id=None):
    return get_agent(provider="gemini", tools=tools, model_id=model_id)

def get_claude_agent(tools=None, model_id=None):
    return get_agent(provider="claude", tools=tools, model_id=model_id)
