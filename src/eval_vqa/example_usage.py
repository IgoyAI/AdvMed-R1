#!/usr/bin/env python3
"""
Example usage script demonstrating how to run evaluations
This script shows various ways to use the evaluation tools
"""

import os
import sys
import argparse


def print_section(title):
    """Print a section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def example_clean_evaluation():
    """Example: Clean zero-shot evaluation"""
    print_section("Example 1: Clean Zero-Shot Evaluation")
    
    cmd = """
python eval_qwen2_5vl_zeroshot.py \\
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \\
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \\
    --dataset_root ../.. \\
    --output_path ../../results/ct_clean.json \\
    --batch_size 8
"""
    print("This evaluates the model on CT images without any adversarial attacks.")
    print("\nCommand:")
    print(cmd)


def example_fgsm_evaluation():
    """Example: FGSM adversarial evaluation"""
    print_section("Example 2: FGSM Adversarial Evaluation")
    
    cmd = """
python eval_qwen2_5vl_adversarial.py \\
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \\
    --test_data ../../Splits/modality/test/X-Ray_test.json \\
    --dataset_root ../.. \\
    --output_path ../../results/xray_fgsm.json \\
    --batch_size 4 \\
    --attack_type fgsm \\
    --epsilon 0.03
"""
    print("This evaluates robustness against FGSM attacks on X-Ray images.")
    print("FGSM is a fast single-step attack - good for quick robustness testing.")
    print("\nCommand:")
    print(cmd)


def example_pgd_evaluation():
    """Example: PGD adversarial evaluation"""
    print_section("Example 3: PGD Adversarial Evaluation")
    
    cmd = """
python eval_qwen2_5vl_adversarial.py \\
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \\
    --test_data ../../Splits/modality/test/MR\ \(Mag-netic\ Resonance\ Imaging\)_test.json \\
    --dataset_root ../.. \\
    --output_path ../../results/mri_pgd.json \\
    --batch_size 4 \\
    --attack_type pgd \\
    --epsilon 0.03 \\
    --pgd_alpha 0.01 \\
    --pgd_iters 10
"""
    print("This evaluates robustness against PGD attacks on MRI images.")
    print("PGD is a stronger iterative attack - more comprehensive robustness test.")
    print("\nCommand:")
    print(cmd)


def example_batch_evaluation():
    """Example: Batch evaluation across all modalities"""
    print_section("Example 4: Batch Evaluation (All Modalities)")
    
    cmd = """
python batch_eval_adversarial.py \\
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \\
    --splits_dir ../../Splits \\
    --dataset_root ../.. \\
    --output_dir ../../results/adversarial_eval \\
    --batch_size 8 \\
    --attacks clean fgsm pgd \\
    --epsilon 0.03 \\
    --evaluation_type modality
"""
    print("This runs comprehensive evaluation across all modalities and attack types.")
    print("Results are saved in a structured directory with a summary report.")
    print("\nCommand:")
    print(cmd)


def example_specific_modalities():
    """Example: Evaluate specific modalities only"""
    print_section("Example 5: Evaluate Specific Modalities")
    
    cmd = """
python batch_eval_adversarial.py \\
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \\
    --splits_dir ../../Splits \\
    --dataset_root ../.. \\
    --output_dir ../../results/adversarial_eval \\
    --batch_size 8 \\
    --modalities "CT(Computed Tomography)" "X-Ray" \\
    --attacks clean fgsm
"""
    print("This evaluates only specific modalities (CT and X-Ray) with selected attacks.")
    print("\nCommand:")
    print(cmd)


def example_shell_script():
    """Example: Using the shell script"""
    print_section("Example 6: Using Shell Script for Quick Evaluation")
    
    print("Set environment variables:")
    env_cmd = """
export MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
export BATCH_SIZE=8
export EPSILON=0.03
"""
    print(env_cmd)
    
    print("\nThen run evaluations:")
    
    print("\n1. Batch evaluation (all modalities):")
    print("   ./run_adversarial_eval.sh batch")
    
    print("\n2. CT modality only:")
    print("   ./run_adversarial_eval.sh ct")
    
    print("\n3. X-Ray modality only:")
    print("   ./run_adversarial_eval.sh xray")
    
    print("\n4. Custom test file:")
    print("   ./run_adversarial_eval.sh single path/to/test.json output_name")


def example_comparing_epsilons():
    """Example: Comparing different epsilon values"""
    print_section("Example 7: Comparing Different Attack Strengths")
    
    print("Evaluate with different epsilon values to see robustness degradation:")
    
    for eps in [0.01, 0.03, 0.05]:
        cmd = f"""
python eval_qwen2_5vl_adversarial.py \\
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \\
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \\
    --dataset_root ../.. \\
    --output_path ../../results/ct_fgsm_eps{eps}.json \\
    --batch_size 4 \\
    --attack_type fgsm \\
    --epsilon {eps}
"""
        print(f"\nEpsilon = {eps}:")
        print(cmd)


def example_workflow():
    """Example: Complete workflow"""
    print_section("Example 8: Complete Evaluation Workflow")
    
    print("""
A complete workflow for evaluating your model:

Step 1: Download the model
------------------------
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/Qwen2.5-VL-3B

Step 2: Place the OmniMedVQA dataset
---------------------------------
Place the OmniMedVQA folder in the repository root:
  AdvMed-R1/
  ├── OmniMedVQA/
  │   └── Images/
  ├── Splits/
  └── ...

Step 3: Navigate to evaluation directory
-------------------------------------
cd src/eval_vqa

Step 4: Run clean evaluation
--------------------------
python eval_qwen2_5vl_zeroshot.py \\
    --model_path ../../models/Qwen2.5-VL-3B \\
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \\
    --dataset_root ../.. \\
    --output_path ../../results/ct_clean.json \\
    --batch_size 8

Step 5: Run adversarial evaluations
--------------------------------
python eval_qwen2_5vl_adversarial.py \\
    --model_path ../../models/Qwen2.5-VL-3B \\
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \\
    --dataset_root ../.. \\
    --output_path ../../results/ct_fgsm.json \\
    --batch_size 4 \\
    --attack_type fgsm \\
    --epsilon 0.03

Step 6: Run batch evaluation (all modalities)
------------------------------------------
python batch_eval_adversarial.py \\
    --model_path ../../models/Qwen2.5-VL-3B \\
    --splits_dir ../../Splits \\
    --dataset_root ../.. \\
    --output_dir ../../results/adversarial_eval \\
    --batch_size 8 \\
    --evaluation_type modality

Step 7: Analyze results
--------------------
Results are saved as JSON files with:
- Overall accuracy
- Per-sample predictions
- Correct/incorrect labels

Check the summary:
cat ../../results/adversarial_eval/evaluation_summary.json
""")


def main():
    parser = argparse.ArgumentParser(
        description="Show example usage for adversarial evaluation scripts"
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=range(1, 9),
        help="Show specific example (1-8), or show all if not specified"
    )
    
    args = parser.parse_args()
    
    examples = [
        example_clean_evaluation,
        example_fgsm_evaluation,
        example_pgd_evaluation,
        example_batch_evaluation,
        example_specific_modalities,
        example_shell_script,
        example_comparing_epsilons,
        example_workflow,
    ]
    
    if args.example:
        examples[args.example - 1]()
    else:
        print("\n" + "="*80)
        print("  ADVERSARIAL EVALUATION EXAMPLES")
        print("  Qwen 2.5 VL 3B on OmniMedVQA")
        print("="*80)
        
        for example_fn in examples:
            example_fn()
        
        print("\n" + "="*80)
        print("  For more information, see README_ADVERSARIAL.md")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
