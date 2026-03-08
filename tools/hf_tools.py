"""
High-Performance Hugging Face Tools for smolagents.

This module provides a suite of modular, optimized tools for use with
CodeAgent and ToolCallingAgent. Each tool is decorated with @tool and
implements precise type hints and docstrings for maximum LLM accuracy.
"""

import os
import subprocess
import json
from typing import Optional

from smolagents import tool


# =============================================================================
# Hub Operations
# =============================================================================

@tool
def hf_download(
    repo_id: str,
    file_pattern: Optional[str] = None,
    local_dir: Optional[str] = None,
    repo_type: str = "model"
) -> str:
    """
    Download files from a Hugging Face Hub repository to local storage.

    Args:
        repo_id: The repository ID in format 'owner/repo-name' (e.g., 'meta-llama/Llama-2-7b-hf', 'HuggingFaceFW/fineweb').
        file_pattern: Glob pattern to filter files. Use '*.safetensors' for model weights, '*.json' for configs, '*.parquet' for datasets.
        local_dir: Absolute path to save files locally (e.g., '/home/user/models/llama'). If None, uses HF cache (~/.cache/huggingface/).
        repo_type: Must be one of: 'model' (default), 'dataset', or 'space'.

    Returns:
        Success message with downloaded file paths, or detailed error message.
    """
    print(f"🔄 Starting download: {repo_id} (type={repo_type}, pattern={file_pattern or 'all'})")
    
    # Validate repo_type
    if repo_type not in ["model", "dataset", "space"]:
        error_msg = f"Invalid repo_type '{repo_type}'. Must be 'model', 'dataset', or 'space'."
        print(f"❌ Validation error: {error_msg}")
        raise ValueError(error_msg)
    
    cmd = ["hf", "download", repo_id, "--repo-type", repo_type]
    
    if file_pattern:
        cmd.extend(["--include", file_pattern])
        print(f"   Filtering files: {file_pattern}")
    if local_dir:
        cmd.extend(["--local-dir", local_dir])
        print(f"   Saving to: {local_dir}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            output = f"✅ Downloaded from {repo_id}:\n{result.stdout}"
            print(output)
            return output
        error_output = f"❌ Download failed (exit code {result.returncode}):\n{result.stderr}"
        print(error_output)
        return error_output
    except subprocess.TimeoutExpired:
        error_msg = "❌ Download timed out after 10 minutes. Try downloading a specific file using file_pattern."
        print(error_msg)
        return error_msg
    except FileNotFoundError:
        error_msg = "❌ 'hf' CLI not found. Install with: pip install 'huggingface_hub[cli]'"
        print(error_msg)
        return error_msg


@tool
def hf_upload(
    repo_id: str,
    local_path: str,
    remote_path: Optional[str] = None,
    repo_type: str = "model",
    commit_message: Optional[str] = None
) -> str:
    """
    Upload files or directories to a Hugging Face Hub repository.

    Args:
        repo_id: The target repository ID (e.g., 'username/my-model').
        local_path: Local file or directory path to upload.
        remote_path: Optional destination path in the repo. Defaults to root.
        repo_type: Type of repository: 'model', 'dataset', or 'space'.
        commit_message: Optional commit message for the upload.

    Returns:
        Upload confirmation or error message.
    """
    remote = remote_path or "."
    cmd = ["hf", "upload", repo_id, local_path, remote, "--repo-type", repo_type]
    
    if commit_message:
        cmd.extend([f"--commit-message={commit_message}"])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            return f"✅ Uploaded to {repo_id}: {result.stdout}"
        return f"❌ Upload failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "❌ Upload timed out (10 min limit)."
    except FileNotFoundError:
        return "❌ 'hf' CLI not found. Install with: pip install huggingface_hub[cli]"


@tool
def hf_repo_create(
    name: str,
    repo_type: str = "model",
    private: bool = False,
    organization: Optional[str] = None
) -> str:
    """
    Create a new repository on Hugging Face Hub.

    Args:
        name: Name for the new repository.
        repo_type: Type of repository: 'model', 'dataset', or 'space'.
        private: Whether the repository should be private.
        organization: Optional organization namespace.

    Returns:
        Repository creation confirmation with URL.
    """
    repo_id = f"{organization}/{name}" if organization else name
    cmd = ["hf", "repo", "create", repo_id, "--repo-type", repo_type]
    
    if private:
        cmd.append("--private")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return f"✅ Created: https://huggingface.co/{repo_type}s/{repo_id}"
        return f"❌ Creation failed: {result.stderr}"
    except FileNotFoundError:
        return "❌ 'hf' CLI not found. Install with: pip install huggingface_hub[cli]"


# =============================================================================
# Model & Dataset Discovery
# =============================================================================

@tool
def hf_search_models(
    query: str,
    author: Optional[str] = None,
    task: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Search for models on Hugging Face Hub.

    Args:
        query: Search query string.
        author: Filter by author/organization (e.g., 'meta-llama').
        task: Filter by task (e.g., 'text-generation', 'image-classification').
        limit: Maximum number of results to return (1-50).

    Returns:
        JSON list of matching models with id, downloads, and likes.
    """
    cmd = ["hf", "models", "ls", "--search", query, "--limit", str(min(limit, 50))]
    
    if author:
        cmd.extend(["--author", author])
    if task:
        cmd.extend(["--filter", task])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return f"🔍 Search results for '{query}':\n{result.stdout}"
        return f"❌ Search failed: {result.stderr}"
    except FileNotFoundError:
        return "❌ 'hf' CLI not found. Install with: pip install huggingface_hub[cli]"


@tool
def hf_model_info(repo_id: str) -> str:
    """
    Get detailed information about a specific model on Hugging Face Hub.

    Args:
        repo_id: The model repository ID (e.g., 'meta-llama/Llama-2-7b').

    Returns:
        Model metadata including downloads, likes, tags, and pipeline info.
    """
    cmd = ["hf", "models", "info", repo_id]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return f"📦 {repo_id} Info:\n{result.stdout}"
        return f"❌ Info retrieval failed: {result.stderr}"
    except FileNotFoundError:
        return "❌ 'hf' CLI not found. Install with: pip install huggingface_hub[cli]"


@tool
def hf_search_datasets(
    query: str,
    author: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Search for datasets on Hugging Face Hub.

    Args:
        query: Search query string.
        author: Filter by author/organization.
        limit: Maximum number of results to return (1-50).

    Returns:
        List of matching datasets.
    """
    cmd = ["hf", "datasets", "ls", "--search", query, "--limit", str(min(limit, 50))]
    
    if author:
        cmd.extend(["--author", author])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return f"🔍 Dataset results for '{query}':\n{result.stdout}"
        return f"❌ Search failed: {result.stderr}"
    except FileNotFoundError:
        return "❌ 'hf' CLI not found. Install with: pip install huggingface_hub[cli]"


# =============================================================================
# Dataset SQL Querying (Advanced)
# =============================================================================

@tool
def hf_query_dataset(
    repo_id: str,
    sql_query: str,
    split: str = "train",
    limit: int = 100
) -> str:
    """
    Query a Hugging Face dataset using DuckDB SQL syntax.

    This enables powerful data exploration without downloading entire datasets.
    Uses the hf:// protocol for direct dataset access.

    Args:
        repo_id: The dataset repository ID (e.g., 'imdb').
        sql_query: SQL query to execute. Use 'data' as the table name.
                   Example: "SELECT * FROM data WHERE label = 1"
        split: Dataset split to query ('train', 'test', 'validation').
        limit: Row limit for safety (max 10000).

    Returns:
        Query results as JSON or error message.
    """
    try:
        import duckdb
    except ImportError:
        return "❌ DuckDB not installed. Run: pip install duckdb"
    
    try:
        # Construct the HF dataset path
        hf_path = f"hf://datasets/{repo_id}/{split}*.parquet"
        
        # Replace 'data' table reference with actual path
        actual_query = sql_query.replace("data", f"read_parquet('{hf_path}')")
        
        # Add limit if not present
        if "LIMIT" not in actual_query.upper():
            actual_query += f" LIMIT {min(limit, 10000)}"
        
        # Set HF token for authentication
        token = os.getenv("HF_TOKEN", "")
        result = duckdb.sql(f"SET hf_token='{token}'; {actual_query}")
        
        return f"📊 Query Results ({repo_id}):\n{result.fetchdf().to_string()}"
    except Exception as e:
        return f"❌ Query failed: {str(e)}"


# =============================================================================
# Cache & Authentication
# =============================================================================

@tool
def hf_cache_info() -> str:
    """
    Display information about the local Hugging Face cache.

    Returns:
        List of cached repositories and their sizes.
    """
    try:
        result = subprocess.run(
            ["hf", "cache", "ls"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return f"📦 HF Cache:\n{result.stdout}"
        return f"❌ Cache info failed: {result.stderr}"
    except FileNotFoundError:
        return "❌ 'hf' CLI not found. Install with: pip install huggingface_hub[cli]"


@tool
def hf_whoami() -> str:
    """
    Check the currently authenticated Hugging Face user.

    Returns:
        Current user information or authentication status.
    """
    try:
        result = subprocess.run(
            ["hf", "auth", "whoami"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return f"👤 Logged in as:\n{result.stdout}"
        return "❌ Not logged in. Run: hf auth login"
    except FileNotFoundError:
        return "❌ 'hf' CLI not found. Install with: pip install huggingface_hub[cli]"


# =============================================================================
# Cloud Jobs
# =============================================================================

@tool
def hf_run_job(
    command: str,
    image: str = "python:3.12",
    flavor: Optional[str] = None,
    use_secrets: bool = True
) -> str:
    """
    Run a job on Hugging Face Cloud Compute.

    Args:
        command: The command to execute (e.g., 'python train.py').
        image: Docker image to use (default: python:3.12).
        flavor: Hardware flavor (e.g., 'a10g-small' for GPU, None for CPU).
        use_secrets: Whether to inject HF_TOKEN as a secret.

    Returns:
        Job ID and status.
    """
    cmd = ["hf", "jobs", "run"]
    
    if flavor:
        cmd.extend(["--flavor", flavor])
    if use_secrets:
        cmd.extend(["--secrets", "HF_TOKEN"])
    
    cmd.extend([image, *command.split()])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            return f"🚀 Job started:\n{result.stdout}"
        return f"❌ Job failed to start: {result.stderr}"
    except FileNotFoundError:
        return "❌ 'hf' CLI not found. Install with: pip install huggingface_hub[cli]"


@tool
def hf_job_status(job_id: str) -> str:
    """
    Check the status and logs of a Hugging Face job.

    Args:
        job_id: The job ID returned from hf_run_job.

    Returns:
        Job status and recent logs.
    """
    try:
        result = subprocess.run(
            ["hf", "jobs", "logs", job_id],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return f"📋 Job {job_id} Logs:\n{result.stdout}"
        return f"❌ Failed to get logs: {result.stderr}"
    except FileNotFoundError:
        return "❌ 'hf' CLI not found. Install with: pip install huggingface_hub[cli]"


# =============================================================================
# Exports for easy importing
# =============================================================================

ALL_HF_TOOLS = [
    hf_download,
    hf_upload,
    hf_repo_create,
    hf_search_models,
    hf_model_info,
    hf_search_datasets,
    hf_query_dataset,
    hf_cache_info,
    hf_whoami,
    hf_run_job,
    hf_job_status,
]


def get_hub_tools():
    """Return all Hub-related tools (download, upload, repo management)."""
    return [hf_download, hf_upload, hf_repo_create]


def get_search_tools():
    """Return all search/discovery tools."""
    return [hf_search_models, hf_model_info, hf_search_datasets, hf_query_dataset]


def get_compute_tools():
    """Return cloud compute tools."""
    return [hf_run_job, hf_job_status]
