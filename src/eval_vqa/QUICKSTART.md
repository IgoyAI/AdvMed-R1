# Quick Start Guide: Zero-Shot Adversarial Evaluation

This guide will help you quickly set up and run zero-shot evaluation with adversarial attacks on the OmniMedVQA dataset.

## Prerequisites

1. **Model**: Download Qwen 2.5 VL 3B Instruct
2. **Dataset**: OmniMedVQA dataset (place in repository root)
3. **Hardware**: NVIDIA GPU with at least 24GB VRAM (recommended)

## Setup

### 1. Download the Model

#### Option A: Automatic Download (Recommended)

Use our utility script to automatically download the model:

```bash
# Download Qwen2.5-VL-3B-Instruct
python src/utils/download_model.py --model qwen2.5-vl-3b --output ./models/Qwen2.5-VL-3B-Instruct

# Or download Med-R1 checkpoint
python src/utils/download_model.py --model med-r1 --output ./checkpoints/Med-R1
```

#### Option B: Manual Download

Using Hugging Face CLI:

```bash
# Using Hugging Face CLI
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/Qwen2.5-VL-3B-Instruct
```

Or specify a different model path if you already have it downloaded.

### 2. Place the Dataset

Place the OmniMedVQA dataset folder in the repository root:

```
AdvMed-R1/
├── OmniMedVQA/           # ← Your dataset folder here
│   └── Images/
│       ├── Chest CT Scan/
│       ├── X-Ray/
│       └── ...
├── Splits/               # ← Already in the repo
│   └── modality/test/
└── src/
```

## Quick Evaluation

### Option 1: Using Shell Script (Easiest)

```bash
cd src/eval_vqa

# Set your model path
export MODEL_PATH="/path/to/Qwen2.5-VL-3B-Instruct"

# Run batch evaluation on all modalities
./run_adversarial_eval.sh batch

# Or evaluate specific modality
./run_adversarial_eval.sh ct      # CT only
./run_adversarial_eval.sh xray    # X-Ray only
./run_adversarial_eval.sh mri     # MRI only
```

### Option 1.5: Using SLURM (For Cluster Environments)

If you're running on a cluster with SLURM:

```bash
cd src/eval_vqa

# Configure environment variables (optional)
export MODEL_PATH="/path/to/model"
export SPLITS_DIR="/path/to/Splits"
export OUTPUT_DIR="/path/to/output"

# Submit batch jobs for all modalities
./submit_adversarial_eval.sh batch

# Or submit for specific modality
./submit_adversarial_eval.sh ct      # CT only
./submit_adversarial_eval.sh xray    # X-Ray only
./submit_adversarial_eval.sh mri     # MRI only
```

This submits separate SLURM jobs for each attack type (clean, fgsm, pgd), allowing parallel execution across the cluster.

### Option 1.6: Using SLURM for Concurrent Clean Zero-Shot Evaluation (Recommended for Efficiency)

For efficient clean zero-shot evaluation of all modalities on a single GPU with concurrent execution:

```bash
cd src/eval_vqa

# Configure environment variables (optional)
export MODEL_PATH="/path/to/model"
export SPLITS_DIR="/path/to/Splits"
export OUTPUT_DIR="/path/to/output"
export CONCURRENT_JOBS=4  # Number of modalities to evaluate concurrently

# Submit single job that evaluates all modalities concurrently
./submit_clean_zeroshot.sh
```

This submits a single SLURM job that evaluates all 8 modalities concurrently on 1 GPU, making efficient use of GPU resources. See `CLEAN_ZEROSHOT_README.md` for detailed documentation.

**Benefits:**
- Uses only 1 GPU instead of multiple jobs
- Evaluates multiple modalities in parallel
- Faster overall completion time
- More efficient resource utilization

### Option 2: Using Python Scripts Directly

#### Clean Zero-Shot Evaluation (No Attack)

```bash
cd src/eval_vqa

python eval_qwen2_5vl_zeroshot.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root ../.. \
    --output_path ../../results/ct_clean.json \
    --batch_size 8
```

#### FGSM Attack Evaluation

```bash
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root ../.. \
    --output_path ../../results/ct_fgsm.json \
    --batch_size 4 \
    --attack_type fgsm \
    --epsilon 0.03
```

#### PGD Attack Evaluation

```bash
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

#### Batch Evaluation (All Modalities)

```bash
python batch_eval_adversarial.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --splits_dir ../../Splits \
    --dataset_root ../.. \
    --output_dir ../../results/adversarial_eval \
    --batch_size 8 \
    --evaluation_type modality
```

## Understanding Results

Results are saved as JSON files with:

```json
{
  "accuracy": 85.5,
  "num_samples": 100,
  "num_correct": 85,
  "attack_type": "fgsm",
  "epsilon": 0.03,
  "detailed_results": [...]
}
```

### Summary Report

The batch evaluation creates a summary file:

```bash
cat results/adversarial_eval/evaluation_summary.json
```

This shows accuracy for all modalities and attack types in one place.

## Common Commands

### Evaluate All Modalities (Clean)

```bash
cd src/eval_vqa

python batch_eval_adversarial.py \
    --model_path /path/to/model \
    --splits_dir ../../Splits \
    --dataset_root ../.. \
    --output_dir ../../results/clean \
    --attacks clean \
    --batch_size 8
```

### Evaluate with Different Attack Strengths

```bash
# Weak attack (epsilon=0.01)
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/model \
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root ../.. \
    --output_path ../../results/ct_fgsm_weak.json \
    --attack_type fgsm \
    --epsilon 0.01

# Medium attack (epsilon=0.03)
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/model \
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root ../.. \
    --output_path ../../results/ct_fgsm_medium.json \
    --attack_type fgsm \
    --epsilon 0.03

# Strong attack (epsilon=0.05)
python eval_qwen2_5vl_adversarial.py \
    --model_path /path/to/model \
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root ../.. \
    --output_path ../../results/ct_fgsm_strong.json \
    --attack_type fgsm \
    --epsilon 0.05
```

### Evaluate Specific Modalities Only

```bash
python batch_eval_adversarial.py \
    --model_path /path/to/model \
    --splits_dir ../../Splits \
    --dataset_root ../.. \
    --output_dir ../../results/selected \
    --modalities "CT(Computed Tomography)" "X-Ray" \
    --attacks clean fgsm pgd
```

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
--batch_size 2  # or even 1
```

### Slow Evaluation

- For FGSM: Already optimized (single step)
- For PGD: Reduce iterations
  ```bash
  --pgd_iters 5  # instead of 10
  ```

### Model Not Found

Check the model path:
```bash
ls /path/to/Qwen2.5-VL-3B-Instruct
# Should show: config.json, model files, etc.
```

### Dataset Not Found

Verify the dataset structure:
```bash
ls OmniMedVQA/Images/
# Should show: Chest CT Scan/, X-Ray/, etc.
```

Check image paths in test JSONs:
```bash
cat Splits/modality/test/CT\(Computed\ Tomography\)_test.json | head -20
```

## Performance Tips

1. **Use larger batch sizes** for clean evaluation (8-16)
2. **Use smaller batch sizes** for adversarial attacks (2-4)
3. **Enable mixed precision** (already enabled with bfloat16)
4. **Use flash attention** (already enabled)

## Expected Runtime

On a single A100 GPU (40GB):

| Evaluation Type | Samples | Batch Size | Approx. Time |
|----------------|---------|------------|--------------|
| Clean          | 100     | 8          | 2-3 minutes  |
| FGSM           | 100     | 4          | 5-8 minutes  |
| PGD (10 iter)  | 100     | 4          | 20-30 minutes|

## Next Steps

1. **View Examples**: Run `python example_usage.py` to see more examples
2. **Read Documentation**: See `README_ADVERSARIAL.md` for detailed documentation
3. **Analyze Results**: Use the JSON outputs to compare model robustness

## Support

For issues or questions:
1. Check `README_ADVERSARIAL.md` for detailed documentation
2. Review example usage: `python example_usage.py`
3. Open an issue on GitHub
