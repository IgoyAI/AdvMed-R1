# Adversarial Evaluation for Qwen 2.5 VL 3B

This directory contains scripts for zero-shot evaluation of Qwen 2.5 VL 3B on the OmniMedVQA dataset with support for adversarial attacks (FGSM and PGD).

## Overview

The evaluation framework supports:
- **Clean evaluation**: Standard zero-shot evaluation without adversarial perturbations
- **FGSM attack**: Fast Gradient Sign Method - single-step adversarial attack
- **PGD attack**: Projected Gradient Descent - iterative adversarial attack

## Prerequisites

Ensure you have the following:
1. Qwen 2.5 VL 3B model checkpoint
2. OmniMedVQA dataset (should be placed in the repository root)
3. Required Python packages (see main requirements.txt)

## Dataset Structure

The OmniMedVQA dataset should be organized as follows:
```
/path/to/repo/
├── OmniMedVQA/
│   └── Images/
│       ├── Chest CT Scan/
│       ├── X-Ray/
│       ├── MRI/
│       └── ...
├── Splits/
│   ├── modality/
│   │   └── test/
│   │       ├── CT(Computed Tomography)_test.json
│   │       ├── X-Ray_test.json
│   │       └── ...
│   └── question_type/
│       └── test/
│           └── ...
```

## Usage

### Single Evaluation

Run evaluation on a single test file:

```bash
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --test_data Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root . \
    --output_path results/ct_clean.json \
    --batch_size 8 \
    --attack_type clean
```

### FGSM Attack Evaluation

```bash
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --test_data Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root . \
    --output_path results/ct_fgsm.json \
    --batch_size 4 \
    --attack_type fgsm \
    --epsilon 0.03
```

### PGD Attack Evaluation

```bash
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --test_data Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root . \
    --output_path results/ct_pgd.json \
    --batch_size 4 \
    --attack_type pgd \
    --epsilon 0.03 \
    --pgd_alpha 0.01 \
    --pgd_iters 10
```

### Batch Evaluation (All Modalities)

Run evaluation across all modalities and attack types:

```bash
cd src/eval_vqa

python batch_eval_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --splits_dir ../../Splits \
    --dataset_root ../.. \
    --output_dir ../../results/adversarial_eval \
    --batch_size 8 \
    --attacks clean fgsm pgd \
    --epsilon 0.03 \
    --pgd_alpha 0.01 \
    --pgd_iters 10 \
    --evaluation_type modality
```

### Evaluate Specific Modalities

```bash
python batch_eval_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --splits_dir ../../Splits \
    --dataset_root ../.. \
    --output_dir ../../results/adversarial_eval \
    --modalities "CT(Computed Tomography)" "X-Ray" "MR (Mag-netic Resonance Imaging)" \
    --attacks clean fgsm pgd
```

## Command Line Arguments

### eval_qwen2_5vl_adversarial.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | **required** | Path to Qwen 2.5 VL model checkpoint |
| `--test_data` | str | **required** | Path to test data JSON file |
| `--dataset_root` | str | `.` | Root directory where OmniMedVQA is located |
| `--batch_size` | int | `8` | Batch size for inference |
| `--output_path` | str | **required** | Path to save results |
| `--attack_type` | str | `clean` | Attack type: `clean`, `fgsm`, or `pgd` |
| `--epsilon` | float | `0.03` | Perturbation magnitude (L∞ norm bound) |
| `--pgd_alpha` | float | `0.01` | Step size for PGD attack |
| `--pgd_iters` | int | `10` | Number of iterations for PGD |

### batch_eval_adversarial.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | **required** | Path to Qwen 2.5 VL model checkpoint |
| `--splits_dir` | str | `Splits` | Path to Splits directory |
| `--dataset_root` | str | `.` | Root directory where OmniMedVQA is located |
| `--evaluation_type` | str | `modality` | Type: `modality`, `question_type`, or `both` |
| `--batch_size` | int | `8` | Batch size for inference |
| `--output_dir` | str | `results/adversarial_eval` | Directory to save results |
| `--attacks` | list | `[clean, fgsm, pgd]` | Types of attacks to evaluate |
| `--epsilon` | float | `0.03` | Perturbation magnitude |
| `--pgd_alpha` | float | `0.01` | Step size for PGD |
| `--pgd_iters` | int | `10` | Iterations for PGD |
| `--modalities` | list | `None` | Specific modalities to evaluate |

## Output Format

Results are saved in JSON format with the following structure:

```json
{
  "test_data_path": "path/to/test.json",
  "attack_type": "fgsm",
  "epsilon": 0.03,
  "accuracy": 85.5,
  "num_samples": 100,
  "num_correct": 85,
  "detailed_results": [
    {
      "model_output": "<think>...</think><answer>A</answer>",
      "model_answer": "A",
      "ground_truth": "<answer> A </answer>",
      "gt_answer": "A",
      "correct": true
    },
    ...
  ]
}
```

## Adversarial Attack Details

### FGSM (Fast Gradient Sign Method)

FGSM is a single-step attack that adds perturbations in the direction of the gradient:

```
x_adv = x + ε * sign(∇_x L(x, y))
```

- Fast and efficient
- Good for testing basic robustness
- Typical ε values: 0.01 - 0.05

### PGD (Projected Gradient Descent)

PGD is an iterative attack that refines perturbations over multiple steps:

```
x_adv^(t+1) = Π_ε(x_adv^(t) + α * sign(∇_x L(x_adv^(t), y)))
```

- More powerful than FGSM
- Better approximates the optimal adversarial perturbation
- Typical settings: ε=0.03, α=0.01, iterations=10

## Performance Considerations

1. **Batch Size**: Adversarial attacks require gradient computation, which uses more memory. Reduce batch size if you encounter OOM errors.

2. **Attack Speed**: 
   - Clean evaluation: Fast
   - FGSM: ~2x slower than clean
   - PGD: ~10-20x slower than clean (depending on iterations)

3. **GPU Memory**: 
   - Recommended: At least 24GB VRAM for batch_size=8
   - For smaller GPUs, reduce batch_size to 2-4

## Example Workflow

Complete workflow for evaluating on CT modality:

```bash
# 1. Navigate to eval_vqa directory
cd src/eval_vqa

# 2. Run clean evaluation
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root ../.. \
    --output_path ../../results/ct_clean.json \
    --batch_size 8 \
    --attack_type clean

# 3. Run FGSM evaluation
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root ../.. \
    --output_path ../../results/ct_fgsm.json \
    --batch_size 4 \
    --attack_type fgsm \
    --epsilon 0.03

# 4. Run PGD evaluation
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root ../.. \
    --output_path ../../results/ct_pgd.json \
    --batch_size 4 \
    --attack_type pgd \
    --epsilon 0.03 \
    --pgd_alpha 0.01 \
    --pgd_iters 10
```

## Batch Evaluation for All Modalities

```bash
cd src/eval_vqa

python batch_eval_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --splits_dir ../../Splits \
    --dataset_root ../.. \
    --output_dir ../../results/adversarial_eval \
    --batch_size 8 \
    --evaluation_type modality
```

This will evaluate all modalities (CT, MRI, X-Ray, etc.) with all three attack types and produce a comprehensive summary.

## Troubleshooting

### OOM (Out of Memory) Errors

Reduce batch size:
```bash
--batch_size 2  # or even 1 for very limited memory
```

### Slow Evaluation

For faster results on PGD:
```bash
--pgd_iters 5  # reduce iterations
```

### Model Not Found

Ensure the model path is correct and the model is downloaded:
```bash
# Example: Download from Hugging Face
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /path/to/model
```

### Dataset Path Issues

Make sure:
1. OmniMedVQA folder is in the repository root
2. Image paths in JSON files match the actual file structure
3. `--dataset_root` points to the directory containing the OmniMedVQA folder

## Citation

If you use this evaluation framework, please cite:

```bibtex
@article{lai2025med,
  title={Med-R1: Reinforcement Learning for Generalizable Medical Reasoning in Vision-Language Models},
  author={Lai, Yuxiang and Zhong, Jike and Li, Ming and Zhao, Shitian and Yang, Xiaofeng},
  journal={arXiv preprint arXiv:2503.13939},
  year={2025}
}
```
