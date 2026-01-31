#!/bin/bash

# setup_smolagents.sh
# Automates the setup of smolagents conda environment and dependencies.

ENV_NAME="smolagents"
PYTHON_VERSION="3.10"

echo "🚀 Starting smolagents setup..."

# 1. Create Conda Environment
if conda env list | grep -q "$ENV_NAME"; then
    echo "⚠️  Environment '$ENV_NAME' already exists. Updating..."
else
    echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

# 2. Activate Environment and Install smolagents
echo "Installing smolagents and core dependencies..."
# We use 'conda run' to ensure we are installing into the right environment without needing to 'source activate' in the script
conda run -n "$ENV_NAME" pip install "smolagents[toolkit,litellm,openai,mcp]"
conda run -n "$ENV_NAME" pip install google-generativeai anthropic python-dotenv datasets

# 3. Create .env template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys (GEMINI_API_KEY, ANTHROPIC_API_KEY, HF_TOKEN)."
fi

echo "✅ Setup complete!"
echo "To activate the environment, run: conda activate $ENV_NAME"
