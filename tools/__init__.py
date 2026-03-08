"""
smolagents Tools Package

High-performance, modular tools for CodeAgent and ToolCallingAgent.
"""

from .hf_tools import (
    # Individual tools
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
    # Collections
    ALL_HF_TOOLS,
    get_hub_tools,
    get_search_tools,
    get_compute_tools,
)

__all__ = [
    # Hub Operations
    "hf_download",
    "hf_upload",
    "hf_repo_create",
    # Discovery
    "hf_search_models",
    "hf_model_info",
    "hf_search_datasets",
    "hf_query_dataset",
    # Cache & Auth
    "hf_cache_info",
    "hf_whoami",
    # Cloud Compute
    "hf_run_job",
    "hf_job_status",
    # Collections
    "ALL_HF_TOOLS",
    "get_hub_tools",
    "get_search_tools",
    "get_compute_tools",
]
