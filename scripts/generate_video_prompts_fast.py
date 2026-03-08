#!/usr/bin/env python3
"""
Optimized Video Prompt Generator (Fast Version)

This script generates high-quality synthetic video prompts using:
- Direct Gemini API calls (1 call per prompt, no agent overhead)
- Batched HF Hub uploads (every N prompts to reduce overhead)
- Overwrite protection with force_redownload

Usage:
    python generate_video_prompts_fast.py --count 50 --batch-size 20
    
Dependencies:
    pip install google-generativeai datasets huggingface_hub python-dotenv
"""

import os
import json
import argparse
import random
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
DATASET_REPO = "Limbicnation/Video-Diffusion-Prompt-Style"

# Seed Concepts (diverse visual ideas for variety)
SEED_CONCEPTS = [
    # Sci-Fi / Cyberpunk
    "Cyberpunk street food vendor in heavy rain",
    "Futuristic hospital clean room with robots",
    "Spaceship crash site in a desert",
    "Holographic advertisement malfunctioning",
    "Android getting repaired in a back alley",
    
    # Nature / Documentary
    "Slow motion hummingbird wings",
    "Time-lapse of a flower blooming",
    "Drone shot of a waterfall in Iceland",
    "Lion hunting in tall grass",
    "Underwater coral reef teeming with life",
    
    # Cinematic / Noir
    "Detective walking down a foggy alley",
    "Jazz musician playing saxophone in smoke",
    "Car chase through a tunnel at night",
    "Dramatic courtroom confrontation",
    "Silhouette of a figure in a doorway",
    
    # Abstract / Experimental
    "Ink dropping into water macro shot",
    "Fractal patterns morphing in 3D",
    "Liquid metal flowing uphill",
    "Explosion of colorful powder in slow motion",
    "Light painting in a dark room",
    
    # Fantasy
    "Dragon flying over a medieval castle",
    "Wizard casting a spell with particles",
    "Enchanted forest with glowing mushrooms",
    "Giant walking through a cloud city",
    "Potion brewing in a witch's cauldron",

    # Daily Life / Realism
    "Barista pouring latte art",
    "Skater landing a trick in a park",
    "Busy Tokyo crossing at night",
    "Chef chopping vegetables rapidly",
    "Painter applying brushstrokes to canvas",
    
    # Horror / Thriller
    "Shadow moving in a dark hallway",
    "Abandoned carnival ride creaking",
    "Hand reaching out from under a bed",
    "Flickering light in a basement",
    "Fog rolling over a graveyard",
    
    # Additional concepts for variety
    "Sunrise over mountain peaks time-lapse",
    "Robot hand learning to write calligraphy",
    "Bioluminescent jellyfish in dark ocean",
    "Steam rising from hot coffee cup macro",
    "Northern lights dancing over frozen lake",
]

SYSTEM_PROMPT = """You are an expert AI Video Prompt Engineer specializing in high-fidelity synthetic data for LoRA fine-tuning.
Your goal is to generate premium, structured training data to train a prompt-generation LoRA model on Qwen/Qwen3-4B-Instruct-2507 or 8B Instruct.
The objective is to produce data that enables these small models to generate complex, high-quality video prompts compatible with ComfyUI and LTX-Video on consumer hardware (RTX 4090).

You will be given a simple concept (e.g., "A car chase").
You must transform it into a sophisticated, highly detailed video prompt following this EXACT JSON structure:

{
  "style_name": "A short, descriptive name for the style (e.g. 'Cinematic Action', 'Macro Nature')",
  "prompt_text": "(Camera/Motion description) The main visual subject description. [Style tags, Lighting, Technical specs] --ar 16:9 --model WanVideo --seed 1234",
  "negative_prompt": "A list of things to avoid (e.g. blurry, static, distorted)",
  "tags": ["tag1", "tag2", "tag3"],
  "compatible_models": ["WanVideo", "Sora", "LTX-Video"]
}

LoRA TRAINING RULES:
1. CONSISTENCY: Ensure a strict follow-through of the (Camera/Motion) ... [Style/Technical] format.
2. HIGH FIDELITY: Descriptions must be rich, vivid, and technically precise (e.g. '4k, highly detailed, photorealistic, 24fps').
3. DIVERSITY: Explore unique angles, lighting (volumetric, rim-lighting), and complex motions (dolly zoom, tilt-shift).
4. TAGGING: Provide accurate tags that help the LoRA model learn stylistic associations.
5. NO MARKDOWN: The output must be valid JSON only. Do not wrap in ```json blocks."""


def generate_prompt_gemini(model, concept: str) -> Dict[str, Any]:
    """Generates a single synthetic example using direct Gemini API call."""
    try:
        response = model.generate_content(
            f"Generate a video prompt for: {concept}",
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 500,
            }
        )
        content = response.text.strip()
        
        # Clean up potential markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
            
        result = json.loads(content)
        
        # Add randomness to seed if not present
        if "--seed" not in result.get("prompt_text", ""):
            result["prompt_text"] += f" --seed {random.randint(1000, 9999)}"
            
        return result
    except Exception as e:
        print(f"Error generating for '{concept}': {e}")
        return None


def push_batch_to_hub(batch: List[Dict], token: str, repo_id: str):
    """Pushes a batch of prompts to HF Hub with overwrite protection."""
    from datasets import Dataset, load_dataset, concatenate_datasets
    from huggingface_hub import repo_info
    
    if not batch:
        return
    
    new_ds = Dataset.from_list(batch)
    
    # Check if repo exists (overwrite protection)
    repo_exists = False
    try:
        repo_info(repo_id, repo_type="dataset", token=token)
        repo_exists = True
    except Exception:
        pass
    
    try:
        # Load existing and append with force_redownload for safety
        existing_ds = load_dataset(repo_id, split="train", token=token, download_mode="force_redownload")
        updated_ds = concatenate_datasets([existing_ds, new_ds])
        print(f"  → Appending batch. New total: {len(updated_ds)}")
    except Exception as e:
        if repo_exists:
            raise RuntimeError(f"Dataset exists but failed to load. Aborting to prevent overwrite. Error: {e}")
        updated_ds = new_ds
        print(f"  → Starting new dataset with {len(updated_ds)} rows")
    
    updated_ds.push_to_hub(repo_id, token=token, private=True)


def main():
    parser = argparse.ArgumentParser(description="Fast video prompt generator with batched uploads")
    parser.add_argument("--count", type=int, default=50, help="Number of prompts to generate")
    parser.add_argument("--batch-size", type=int, default=20, help="Prompts to generate before pushing to Hub")
    parser.add_argument("--dry-run", action="store_true", help="Generate without pushing to Hub")
    args = parser.parse_args()

    # Check API keys
    gemini_key = os.environ.get("GEMINI_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not gemini_key:
        print("❌ Error: GEMINI_API_KEY not set in environment or .env file")
        exit(1)
    if not hf_token and not args.dry_run:
        print("❌ Error: HF_TOKEN not set in environment or .env file")
        exit(1)

    # Initialize Gemini
    import google.generativeai as genai
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT
    )
    
    print(f"🚀 Fast Video Prompt Generator")
    print(f"   Target: {args.count} prompts")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Dry run: {args.dry_run}")
    print(f"   Repository: {DATASET_REPO}")
    print()
    
    current_batch = []
    total_generated = 0
    
    pbar = tqdm(total=args.count, desc="Generating")
    
    while total_generated < args.count:
        concept = random.choice(SEED_CONCEPTS)
        result = generate_prompt_gemini(model, concept)
        
        if result:
            current_batch.append(result)
            total_generated += 1
            pbar.update(1)
            
            # Push batch when reaching batch size
            if len(current_batch) >= args.batch_size:
                if not args.dry_run:
                    print(f"\n📤 Pushing batch of {len(current_batch)} prompts...")
                    push_batch_to_hub(current_batch, hf_token, DATASET_REPO)
                    print(f"✅ Batch pushed successfully!")
                else:
                    print(f"\n[DRY RUN] Would push batch of {len(current_batch)} prompts")
                current_batch = []
        
        time.sleep(0.3)  # Respect rate limits
    
    pbar.close()
    
    # Push any remaining prompts
    if current_batch:
        if not args.dry_run:
            print(f"\n📤 Pushing final batch of {len(current_batch)} prompts...")
            push_batch_to_hub(current_batch, hf_token, DATASET_REPO)
            print(f"✅ Final batch pushed successfully!")
        else:
            print(f"\n[DRY RUN] Would push final batch of {len(current_batch)} prompts")
    
    print(f"\n🎉 Generation complete! Total: {total_generated} prompts")


if __name__ == "__main__":
    main()
