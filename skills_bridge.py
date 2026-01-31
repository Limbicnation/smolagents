from smolagents import tool
import sys
import os

# Add the skills directory to path to import generation scripts
SKILLS_PATH = "/home/gero/GitHub/DeepLearning_Lab/skills"
sys.path.append(os.path.join(SKILLS_PATH, "scripts"))

@tool
def generate_synthetic_prompt(concept: str) -> str:
    """
    Converts a simple video concept into a high-fidelity LTX-Video formatted prompt.
    Args:
        concept: A simple description of the video (e.g., 'A robot walking').
    """
    from generate_synthetic_video_prompts import generate_prompt, DEFAULT_MODEL
    from huggingface_hub import InferenceClient
    
    # We assume HF_TOKEN is in the environment
    token = os.getenv("HF_TOKEN")
    if not token:
        return "Error: HF_TOKEN environment variable not set. Please add it to your .env file."
        
    client = InferenceClient(token=token)
    result = generate_prompt(client, concept, DEFAULT_MODEL)
    return str(result)
