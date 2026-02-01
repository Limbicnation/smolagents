import os
from dotenv import load_dotenv
import google.generativeai as genai
from smolagents import CodeAgent, LiteLLMModel, Model, InferenceClientModel
from smolagents.monitoring import TokenUsage
from smolagents.models import ChatMessage, MessageRole
from skills_bridge import generate_synthetic_prompt, push_to_hf_hub

class GeminiModel(Model):
    # ... (rest of GeminiModel remains same)
    def __init__(self, model_id="gemini-2.0-flash", api_key=None, **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)

    def generate(self, messages, stop_sequences=None, **kwargs):
        # Convert smolagents messages to Google format
        contents = []
        for msg in messages:
            role = "user" if msg.role == MessageRole.USER else "model"
            
            # smolagents content can be a string or a list of parts (OpenAI style)
            parts = []
            if isinstance(msg.content, str):
                parts.append(msg.content)
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
            
            if parts:
                contents.append({"role": role, "parts": parts})
        
        response = self.model.generate_content(contents)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=response.text,
            token_usage=TokenUsage(
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count
            )
        )

def get_agent(provider="gemini", tools=None, model_id=None):
    """
    Initialize a CodeAgent using the specified provider.
    
    Args:
        provider: "gemini", "claude", or "qwen"
        tools: List of tools to add to the agent.
        model_id: Specific model ID for the provider.
    """
    load_dotenv()
    
    # Default tools if none provided
    if tools is None:
        tools = [generate_synthetic_prompt, push_to_hf_hub]
    
    if provider == "gemini":
        model_id = model_id or "gemini-2.0-flash"
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        model = GeminiModel(model_id=model_id, api_key=api_key)
        
    elif provider == "claude":
        model_id = model_id or "anthropic/claude-3-5-sonnet-20240620"
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or "your_claude_key_here" in api_key:
            raise ValueError("ANTHROPIC_API_KEY not found or is a placeholder in .env.")
        os.environ["ANTHROPIC_API_KEY"] = api_key
        model = LiteLLMModel(model_id=model_id)

    elif provider == "qwen":
        # Qwen 72B via Inference Hub (using the yesterday's model preference)
        model_id = model_id or "Qwen/Qwen2.5-72B-Instruct"
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN found in environment.")
        model = InferenceClientModel(model_id=model_id, token=token)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")
        
    return CodeAgent(tools=tools, model=model, add_base_tools=True)

# Convenience functions
def get_gemini_agent(tools=None, model_id=None):
    return get_agent(provider="gemini", tools=tools, model_id=model_id)

def get_claude_agent(tools=None, model_id=None):
    return get_agent(provider="claude", tools=tools, model_id=model_id)

def get_qwen_agent(tools=None, model_id=None):
    return get_agent(provider="qwen", tools=tools, model_id=model_id)
