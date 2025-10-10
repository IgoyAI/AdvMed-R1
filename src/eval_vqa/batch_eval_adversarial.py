"""
Batch evaluation script for running evaluations across multiple modalities and attack types
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
import subprocess
import sys


def get_test_files(splits_dir: str, evaluation_type: str = "modality") -> List[Dict]:
    """
    Get all test files from the Splits directory
    
    Args:
        splits_dir: Path to Splits directory
        evaluation_type: Type of evaluation ('modality' or 'question_type')
        
    Returns:
        List of dictionaries with test file information
    """
    test_dir = os.path.join(splits_dir, evaluation_type, "test")
    test_files = []
    
    if os.path.exists(test_dir):
        for file in sorted(os.listdir(test_dir)):
            if file.endswith('.json'):
                # Extract modality/task name
                name = file.replace('_test.json', '')
                test_files.append({
                    'name': name,
                    'path': os.path.join(test_dir, file),
                    'type': evaluation_type
                })
    
    return test_files


def run_single_evaluation(model_path: str, test_data_path: str, dataset_root: str,
                         output_path: str, batch_size: int, attack_type: str,
                         epsilon: float, pgd_alpha: float, pgd_iters: int) -> float:
    """
    Run a single evaluation
    
    Args:
        model_path: Path to model
        test_data_path: Path to test data
        dataset_root: Root directory for dataset
        output_path: Output path for results
        batch_size: Batch size
        attack_type: Attack type
        epsilon: Epsilon for attacks
        pgd_alpha: Alpha for PGD
        pgd_iters: Iterations for PGD
        
    Returns:
        Accuracy
    """
    # Build command
    cmd = [
        sys.executable,
        "eval_qwen2_5vl_adversarial.py",
        "--model_path", model_path,
        "--test_data", test_data_path,
        "--dataset_root", dataset_root,
        "--output_path", output_path,
        "--batch_size", str(batch_size),
        "--attack_type", attack_type,
        "--epsilon", str(epsilon),
        "--pgd_alpha", str(pgd_alpha),
        "--pgd_iters", str(pgd_iters)
    ]
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Parse accuracy from output
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                results = json.load(f)
                return results['accuracy']
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(e.stderr)
        return 0.0
    
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation across modalities and attack types"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Qwen 2.5 VL model checkpoint"
    )
    
    # Data arguments
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="Splits",
        help="Path to Splits directory containing test data"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=".",
        help="Root directory where OmniMedVQA dataset is located"
    )
    parser.add_argument(
        "--evaluation_type",
        type=str,
        choices=['modality', 'question_type', 'both'],
        default='modality',
        help="Type of evaluation to run"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/adversarial_eval",
        help="Directory to save evaluation results"
    )
    
    # Attack arguments
    parser.add_argument(
        "--attacks",
        type=str,
        nargs='+',
        default=['clean', 'fgsm', 'pgd'],
        choices=['clean', 'fgsm', 'pgd'],
        help="Types of attacks to evaluate"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Perturbation magnitude for adversarial attacks"
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=0.01,
        help="Step size for PGD attack"
    )
    parser.add_argument(
        "--pgd_iters",
        type=int,
        default=10,
        help="Number of iterations for PGD attack"
    )
    
    # Filtering arguments
    parser.add_argument(
        "--modalities",
        type=str,
        nargs='+',
        default=None,
        help="Specific modalities to evaluate (if not specified, evaluate all)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get test files
    test_files = []
    if args.evaluation_type in ['modality', 'both']:
        test_files.extend(get_test_files(args.splits_dir, 'modality'))
    if args.evaluation_type in ['question_type', 'both']:
        test_files.extend(get_test_files(args.splits_dir, 'question_type'))
    
    # Filter by modalities if specified
    if args.modalities:
        test_files = [tf for tf in test_files if tf['name'] in args.modalities]
    
    print(f"Found {len(test_files)} test files to evaluate")
    
    # Run evaluations
    all_results = []
    
    for test_file in test_files:
        print(f"\n{'='*80}")
        print(f"Evaluating: {test_file['name']} ({test_file['type']})")
        print(f"{'='*80}\n")
        
        test_results = {
            'name': test_file['name'],
            'type': test_file['type'],
            'path': test_file['path'],
            'attacks': {}
        }
        
        for attack_type in args.attacks:
            print(f"\nRunning {attack_type.upper()} evaluation...")
            
            # Construct output path
            output_filename = f"{test_file['name']}_{attack_type}"
            if attack_type != 'clean':
                output_filename += f"_eps{args.epsilon}"
            output_path = os.path.join(args.output_dir, f"{output_filename}.json")
            
            # Run evaluation
            accuracy = run_single_evaluation(
                model_path=args.model_path,
                test_data_path=test_file['path'],
                dataset_root=args.dataset_root,
                output_path=output_path,
                batch_size=args.batch_size,
                attack_type=attack_type,
                epsilon=args.epsilon,
                pgd_alpha=args.pgd_alpha,
                pgd_iters=args.pgd_iters
            )
            
            test_results['attacks'][attack_type] = {
                'accuracy': accuracy,
                'output_path': output_path
            }
            
            print(f"{attack_type.upper()} Accuracy: {accuracy:.2f}%")
        
        all_results.append(test_results)
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'model_path': args.model_path,
            'evaluation_type': args.evaluation_type,
            'attacks': args.attacks,
            'epsilon': args.epsilon,
            'results': all_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}\n")
    
    # Print summary table
    print(f"{'Test':<40} {'Clean':<10} {'FGSM':<10} {'PGD':<10}")
    print("-" * 70)
    
    for result in all_results:
        name = result['name'][:38]
        clean_acc = result['attacks'].get('clean', {}).get('accuracy', 0)
        fgsm_acc = result['attacks'].get('fgsm', {}).get('accuracy', 0)
        pgd_acc = result['attacks'].get('pgd', {}).get('accuracy', 0)
        
        print(f"{name:<40} {clean_acc:>6.2f}%   {fgsm_acc:>6.2f}%   {pgd_acc:>6.2f}%")
    
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
