#!/usr/bin/env python3
"""
Example: Programmatic Usage of Model Download Utility

This script demonstrates how to use the download utility in your own Python scripts.
"""

import sys
import os

# Add the src directory to the path so we can import the utility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.download_model import download_model, MODEL_CONFIGS


def example_basic_download():
    """Example 1: Basic model download"""
    print("=" * 70)
    print("Example 1: Basic Model Download")
    print("=" * 70)
    
    model_path = download_model(
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        output_dir="./models/Qwen2.5-VL-3B-Instruct"
    )
    print(f"\nModel downloaded to: {model_path}")


def example_with_shortcut():
    """Example 2: Download using shortcut"""
    print("\n" + "=" * 70)
    print("Example 2: Download Using Shortcut")
    print("=" * 70)
    
    # Resolve shortcut to full repo ID
    shortcut = "qwen2.5-vl-3b"
    repo_id = MODEL_CONFIGS[shortcut]["repo_id"]
    
    model_path = download_model(
        model_name=repo_id,
        output_dir="./models/Qwen2.5-VL-3B-Instruct"
    )
    print(f"\nModel downloaded to: {model_path}")


def example_with_options():
    """Example 3: Download with advanced options"""
    print("\n" + "=" * 70)
    print("Example 3: Download with Advanced Options")
    print("=" * 70)
    
    model_path = download_model(
        model_name="yuxianglai117/Med-R1",
        output_dir="./checkpoints/Med-R1",
        token=None,  # Set to your HF token if needed
        resume=True,  # Resume interrupted downloads
        max_workers=8  # Use 8 parallel download threads
    )
    print(f"\nModel downloaded to: {model_path}")


def example_error_handling():
    """Example 4: Error handling"""
    print("\n" + "=" * 70)
    print("Example 4: Error Handling")
    print("=" * 70)
    
    try:
        model_path = download_model(
            model_name="non-existent-model/does-not-exist",
            output_dir="./models/test"
        )
        print(f"\nModel downloaded to: {model_path}")
    except Exception as e:
        print(f"\nCaught expected error: {e}")
        print("This demonstrates proper error handling")


def list_available_models():
    """Example 5: List available pre-configured models"""
    print("\n" + "=" * 70)
    print("Example 5: List Available Models")
    print("=" * 70)
    
    print("\nPre-configured models:")
    for key, config in MODEL_CONFIGS.items():
        print(f"  - {key}: {config['description']}")
        print(f"    Repo: {config['repo_id']}")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("Model Download Utility - Programmatic Usage Examples")
    print("=" * 70)
    print("\nNote: These examples show the code structure.")
    print("Actual downloads are commented out to avoid unnecessary downloads.\n")
    
    # List available models (no download required)
    list_available_models()
    
    # Show code for other examples without executing
    print("\n" + "=" * 70)
    print("Code Examples (not executed)")
    print("=" * 70)
    
    print("\n# Example 1: Basic download")
    print('model_path = download_model(')
    print('    model_name="Qwen/Qwen2.5-VL-3B-Instruct",')
    print('    output_dir="./models/Qwen2.5-VL-3B-Instruct"')
    print(')')
    
    print("\n# Example 2: With authentication")
    print('model_path = download_model(')
    print('    model_name="yuxianglai117/Med-R1",')
    print('    output_dir="./checkpoints/Med-R1",')
    print('    token="YOUR_HF_TOKEN"')
    print(')')
    
    print("\n# Example 3: With advanced options")
    print('model_path = download_model(')
    print('    model_name="Qwen/Qwen2.5-VL-3B-Instruct",')
    print('    output_dir="./models",')
    print('    resume=True,')
    print('    max_workers=8')
    print(')')
    
    print("\n# Example 4: Error handling")
    print('try:')
    print('    model_path = download_model(...)')
    print('except Exception as e:')
    print('    print(f"Error downloading model: {e}")')
    
    print("\n" + "=" * 70)
    print("To actually download a model, uncomment the examples above")
    print("or use the command-line interface:")
    print("  python src/utils/download_model.py --model qwen2.5-vl-3b --output ./models")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
