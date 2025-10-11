#!/usr/bin/env python3
"""
Demo script to show how to use the sample image saving feature.

This script demonstrates the new --save_sample_images feature that allows
you to save visual examples of adversarial perturbations.

Usage:
    python demo_sample_images.py --model_path /path/to/model
    
    Or with default settings:
    python demo_sample_images.py (requires MODEL_PATH environment variable)
"""

import argparse
import os
import sys


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def demo_basic_usage():
    """Demo 1: Basic usage with sample images"""
    print_section("Demo 1: Basic FGSM Attack with Sample Images")
    
    cmd = """
python eval_qwen2_5vl_adversarial.py \\
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \\
    --test_data ../../Splits/modality/test/X-Ray_test.json \\
    --dataset_root ../.. \\
    --output_path demo_results/xray_fgsm.json \\
    --batch_size 4 \\
    --attack_type fgsm \\
    --epsilon 0.03 \\
    --save_sample_images 3
"""
    
    print("This command will:")
    print("  ✓ Run FGSM attack on X-Ray dataset")
    print("  ✓ Save 3 sample comparison images in demo_results/sample_images/")
    print("  ✓ Each image shows original vs. perturbed side-by-side")
    print("\nCommand:")
    print(cmd)


def demo_pgd_usage():
    """Demo 2: PGD attack with sample images"""
    print_section("Demo 2: PGD Attack with Sample Images")
    
    cmd = r"""
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root ../.. \
    --output_path demo_results/ct_pgd.json \
    --batch_size 4 \
    --attack_type pgd \
    --epsilon 0.03 \
    --pgd_alpha 0.01 \
    --pgd_iters 10 \
    --save_sample_images 5
"""
    
    print("This command will:")
    print("  ✓ Run PGD attack (10 iterations) on CT dataset")
    print("  ✓ Save 5 sample comparison images")
    print("  ✓ Images will show attack parameters in title")
    print("\nCommand:")
    print(cmd)


def demo_batch_usage():
    """Demo 3: Batch evaluation with sample images"""
    print_section("Demo 3: Batch Evaluation with Sample Images")
    
    cmd = """
python batch_eval_adversarial.py \\
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \\
    --splits_dir ../../Splits \\
    --dataset_root ../.. \\
    --output_dir demo_results/batch_eval \\
    --batch_size 4 \\
    --attacks fgsm pgd \\
    --epsilon 0.03 \\
    --save_sample_images 2
"""
    
    print("This command will:")
    print("  ✓ Run FGSM and PGD attacks on all modalities")
    print("  ✓ Save 2 sample images for EACH evaluation")
    print("  ✓ Organize results by modality and attack type")
    print("  ✓ Generate summary report with all results")
    print("\nCommand:")
    print(cmd)


def demo_shell_script():
    """Demo 4: Using shell scripts"""
    print_section("Demo 4: Using Shell Scripts with Sample Images")
    
    print("Set environment variable to enable sample image saving:")
    print("\n  export SAVE_SAMPLES=3")
    print("  export MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct")
    
    print("\nThen run evaluation:")
    print("\n  ./run_adversarial_eval.sh batch")
    
    print("\nThis will save 3 sample images for each evaluation.")


def demo_output_structure():
    """Demo 5: Output structure"""
    print_section("Demo 5: Output Structure")
    
    print("After running with --save_sample_images 3, you'll see:")
    print("""
demo_results/
├── xray_fgsm.json                  # Evaluation results
└── sample_images/                  # Sample images directory
    ├── sample_0_fgsm_eps0.03.png   # First sample (original vs perturbed)
    ├── sample_15_fgsm_eps0.03.png  # Second sample
    └── sample_42_fgsm_eps0.03.png  # Third sample
""")
    
    print("Each PNG file contains:")
    print("  • Left side: Original image")
    print("  • Right side: Perturbed image")
    print("  • Title: Attack type and parameters")


def run_actual_demo(args):
    """Run an actual demo if model path is provided"""
    print_section("Running Actual Demo")
    
    if not args.model_path:
        print("❌ No model path provided. Skipping actual demo.")
        print("   Set --model_path or MODEL_PATH environment variable to run.")
        return
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        return
    
    test_file = "../../Splits/modality/test/X-Ray_test.json"
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        print("   Make sure you're in the src/eval_vqa directory")
        return
    
    print(f"✓ Model path: {args.model_path}")
    print(f"✓ Test file: {test_file}")
    print("\nRunning demo with 2 sample images...")
    
    cmd = [
        "python", "eval_qwen2_5vl_adversarial.py",
        "--model_path", args.model_path,
        "--test_data", test_file,
        "--dataset_root", "../..",
        "--output_path", "demo_results/xray_fgsm_demo.json",
        "--batch_size", "2",
        "--attack_type", "fgsm",
        "--epsilon", "0.03",
        "--save_sample_images", "2"
    ]
    
    print(f"\nCommand: {' '.join(cmd)}")
    print("\nNote: This demo is skipped in CI environment.")
    print("      Run it manually with your model to see sample images.")


def main():
    parser = argparse.ArgumentParser(
        description="Demo script for sample image saving feature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all demos (no actual execution)
  python demo_sample_images.py
  
  # Run actual demo (requires model)
  python demo_sample_images.py --model_path /path/to/model --run
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.environ.get("MODEL_PATH"),
        help="Path to model (or set MODEL_PATH env var)"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run a demo (requires model and data)"
    )
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║  Sample Perturbed Image Saving - Demo & Usage Examples                   ║
║                                                                           ║
║  This feature allows you to visualize adversarial perturbations by       ║
║  saving side-by-side comparisons of original and perturbed images.       ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Show all demos
    demo_basic_usage()
    demo_pgd_usage()
    demo_batch_usage()
    demo_shell_script()
    demo_output_structure()
    
    if args.run:
        run_actual_demo(args)
    else:
        print_section("To Run Actual Demo")
        print("Add --run flag and --model_path:")
        print("  python demo_sample_images.py --model_path /path/to/model --run")
    
    print("\n" + "="*80)
    print("For more information, see:")
    print("  • README_ADVERSARIAL.md")
    print("  • IMPLEMENTATION_SUMMARY.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
