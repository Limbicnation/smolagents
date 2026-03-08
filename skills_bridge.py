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

@tool
def push_to_hf_hub(data: dict, repo_id: str = "Limbicnation/Video-Diffusion-Prompt-Style") -> str:
    """
    Pushes a dictionary of generated prompt data to a Hugging Face Hub dataset.
    Args:
        data: The prompt data dictionary (style_name, prompt_text, etc.)
        repo_id: The target Hugging Face dataset repository (default: Limbicnation/Video-Diffusion-Prompt-Style)
    """
    from datasets import Dataset, load_dataset, concatenate_datasets
    
    token = os.getenv("HF_TOKEN")
    if not token:
        return "Error: HF_TOKEN environment variable not set."
        
    try:
        # Create a single-row dataset from the dict
        new_row = Dataset.from_list([data])
        
        # 1. Check if the repository exists first to decide if we should expect existing data
        from huggingface_hub import repo_info
        repo_exists = False
        try:
            repo_info(repo_id, repo_type="dataset", token=token)
            repo_exists = True
        except Exception:
            # If we can't even check repo info, we should probably stop to be safe
            pass

        try:
            # Try to load existing and append
            # Use download_mode='force_redownload' to ensure we have the latest
            existing_ds = load_dataset(repo_id, split="train", token=token, download_mode="force_redownload")
            updated_ds = concatenate_datasets([existing_ds, new_row])
            print(f"Appended to existing dataset. New total size: {len(updated_ds)}")
        except Exception as e:
            # SAFETY CHECK: If repo exists but load_dataset failed, it's likely a temporary error.
            # DO NOT overwrite. Raise error instead.
            if repo_exists:
                return f"Error: Dataset exists but failed to load existing data. Aborting to prevent overwrite. Error: {str(e)}"
            
            # If it truly doesn't exist, start fresh
            updated_ds = new_row
            print("Starting new dataset (no existing data found).")
            
        updated_ds.push_to_hub(repo_id, token=token)
        return f"Successfully pushed to {repo_id}"
    except Exception as e:
        return f"Error pushing to Hub: {str(e)}"
