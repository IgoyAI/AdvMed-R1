#!/usr/bin/env python3
"""
Automatic Model Checkpoint Downloader for AdvMed-R1

This script provides functionality to automatically download model checkpoints
from Hugging Face Hub, including:
- Qwen2.5-VL-3B-Instruct (base model)
- Med-R1 checkpoints (trained models)
- Other vision-language models

Usage:
    # Download Qwen2.5-VL-3B-Instruct
    python download_model.py --model Qwen/Qwen2.5-VL-3B-Instruct --output ./models/Qwen2.5-VL-3B-Instruct
    
    # Download Med-R1 checkpoint
    python download_model.py --model yuxianglai117/Med-R1 --output ./checkpoints/Med-R1
    
    # With authentication token
    python download_model.py --model Qwen/Qwen2.5-VL-3B-Instruct --output ./models --token YOUR_HF_TOKEN
"""

import argparse
import os
import sys
from typing import Optional
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, login
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub not installed. Please run: pip install huggingface-hub")
    sys.exit(1)


# Pre-defined model configurations
MODEL_CONFIGS = {
    "qwen2.5-vl-3b": {
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "description": "Qwen2.5-VL-3B-Instruct base model"
    },
    "qwen2-vl-7b": {
        "repo_id": "Qwen/Qwen2-VL-7B-Instruct",
        "description": "Qwen2-VL-7B-Instruct base model"
    },
    "med-r1": {
        "repo_id": "yuxianglai117/Med-R1",
        "description": "Med-R1 trained checkpoint"
    },
}


def download_model(
    model_name: str,
    output_dir: str,
    token: Optional[str] = None,
    resume: bool = True,
    max_workers: int = 4
) -> str:
    """
    Download a model checkpoint from Hugging Face Hub.
    
    Args:
        model_name: Name/ID of the model on Hugging Face Hub (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
        output_dir: Local directory to save the model
        token: Hugging Face API token (optional, for private models)
        resume: Resume interrupted downloads (default: True)
        max_workers: Number of concurrent download threads (default: 4)
    
    Returns:
        Path to the downloaded model directory
    """
    # Login if token is provided
    if token:
        print("Logging in with provided token...")
        login(token=token)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model: {model_name}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    try:
        # Download the model
        local_dir = snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            resume_download=resume,
            max_workers=max_workers,
            local_dir_use_symlinks=False,  # Use actual files instead of symlinks
        )
        
        print("-" * 60)
        print(f"✓ Model downloaded successfully to: {local_dir}")
        return local_dir
        
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print(f"✗ Error: Authentication failed. The model '{model_name}' may be private.")
            print("  Please provide a valid Hugging Face token using --token option.")
            print("  You can get your token from: https://huggingface.co/settings/tokens")
        elif "404" in str(e):
            print(f"✗ Error: Model '{model_name}' not found on Hugging Face Hub.")
            print("  Please check the model name/ID and try again.")
        else:
            print(f"✗ Error downloading model: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        sys.exit(1)


def list_available_models():
    """List pre-configured models available for download."""
    print("\nPre-configured Models:")
    print("=" * 70)
    for key, config in MODEL_CONFIGS.items():
        print(f"\nShortcut: {key}")
        print(f"  Repo ID: {config['repo_id']}")
        print(f"  Description: {config['description']}")
    print("\n" + "=" * 70)
    print("\nYou can also download any model from Hugging Face Hub by providing its full repo ID.")
    print("Example: Qwen/Qwen2.5-VL-3B-Instruct, yuxianglai117/Med-R1, etc.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Automatically download model checkpoints from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download using shortcut name
  python download_model.py --model qwen2.5-vl-3b --output ./models/Qwen2.5-VL-3B-Instruct
  
  # Download using full repo ID
  python download_model.py --model Qwen/Qwen2.5-VL-3B-Instruct --output ./models
  
  # Download Med-R1 checkpoint
  python download_model.py --model med-r1 --output ./checkpoints/Med-R1
  
  # Download with authentication token
  python download_model.py --model Qwen/Qwen2.5-VL-3B-Instruct --output ./models --token YOUR_HF_TOKEN
  
  # List available pre-configured models
  python download_model.py --list
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model name/ID or shortcut (e.g., 'qwen2.5-vl-3b', 'Qwen/Qwen2.5-VL-3B-Instruct', 'med-r1')"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory to save the model (e.g., './models/Qwen2.5-VL-3B-Instruct')"
    )
    
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="Hugging Face API token (optional, for private models)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume interrupted downloads (default: resume enabled)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent download threads (default: 4)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available pre-configured models"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list:
        list_available_models()
        return
    
    # Validate required arguments
    if not args.model or not args.output:
        parser.print_help()
        print("\n✗ Error: Both --model and --output arguments are required (unless using --list)")
        sys.exit(1)
    
    # Resolve model name (check if it's a shortcut)
    model_name = args.model.lower()
    if model_name in MODEL_CONFIGS:
        repo_id = MODEL_CONFIGS[model_name]["repo_id"]
        print(f"Using pre-configured model: {MODEL_CONFIGS[model_name]['description']}")
        print(f"Repo ID: {repo_id}")
    else:
        repo_id = args.model
    
    # Download the model
    download_model(
        model_name=repo_id,
        output_dir=args.output,
        token=args.token,
        resume=not args.no_resume,
        max_workers=args.max_workers
    )


if __name__ == "__main__":
    main()
