# Evaluation Scripts Comparison

This document provides a comparison of different evaluation approaches available in the repository.

## Quick Comparison Table

| Feature | submit_clean_zeroshot.sh | submit_adversarial_eval.sh | batch_eval_adversarial.py | Python Scripts Directly |
|---------|--------------------------|----------------------------|---------------------------|-------------------------|
| **Execution** | SLURM | SLURM | Local Python | Local Python |
| **GPUs Used** | 1 (shared) | Multiple (1 per job) | 1 | 1 |
| **Concurrent** | ✓ Yes | ✗ No | ✗ No | ✗ No |
| **Attack Types** | Clean only | Clean, FGSM, PGD | Clean, FGSM, PGD | Configurable |
| **Modalities** | All 8 (concurrent) | Configurable | All or specific | Single |
| **Resource Efficiency** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Setup Complexity** | Easy | Easy | Medium | Complex |
| **Best For** | Clean evaluation of all modalities | Full adversarial evaluation | Batch processing locally | Single tests |

## Detailed Comparison

### 1. submit_clean_zeroshot.sh (NEW - Concurrent Evaluation)

**Purpose:** Efficient clean zero-shot evaluation of all modalities on a single GPU

**Characteristics:**
- ✅ Uses 1 GPU for all evaluations
- ✅ Evaluates multiple modalities concurrently (default: 4 at a time)
- ✅ Clean zero-shot only (no adversarial attacks)
- ✅ Fast overall completion time
- ✅ Efficient GPU utilization

**When to Use:**
- You need clean zero-shot evaluation for all modalities
- You want to maximize GPU efficiency
- You have limited GPU resources
- You want faster overall completion time

**Example:**
```bash
cd src/eval_vqa
export CONCURRENT_JOBS=4
./submit_clean_zeroshot.sh
```

**Completion Time:** ~4-6 hours (depending on model and concurrent jobs)

---

### 2. submit_adversarial_eval.sh (Original SLURM Script)

**Purpose:** Comprehensive adversarial evaluation with multiple attack types

**Characteristics:**
- ⚠️ Submits separate SLURM jobs for each modality and attack type
- ✅ Supports clean, FGSM, and PGD attacks
- ✅ Can evaluate specific modalities (ct, xray, mri) or all
- ⚠️ Uses multiple GPUs (1 per job)
- ✅ Jobs run independently

**When to Use:**
- You need adversarial attack evaluation (FGSM, PGD)
- You have access to multiple GPUs
- You want jobs to run independently
- You need flexibility in attack parameters

**Example:**
```bash
cd src/eval_vqa
export MODEL_PATH="/path/to/model"
./submit_adversarial_eval.sh batch  # All modalities
./submit_adversarial_eval.sh ct     # CT only
```

**Completion Time:** Varies (multiple jobs run in parallel if GPUs available)

---

### 3. batch_eval_adversarial.py (Python Batch Script)

**Purpose:** Local batch evaluation with Python control

**Characteristics:**
- ✅ Runs locally without SLURM
- ✅ Supports all attack types
- ✅ Can filter specific modalities
- ✅ Sequential execution on 1 GPU
- ✅ Easy to debug and monitor

**When to Use:**
- You don't have SLURM access
- You want to run evaluations locally
- You need fine-grained control over parameters
- You want to debug evaluation issues

**Example:**
```bash
cd src/eval_vqa
python batch_eval_adversarial.py \
    --model_path /path/to/model \
    --splits_dir ../../Splits \
    --dataset_root ../.. \
    --output_dir ../../results \
    --attacks clean fgsm pgd
```

**Completion Time:** ~8-12 hours (sequential execution)

---

### 4. Python Scripts Directly

**Purpose:** Manual evaluation for single modalities/tests

**Characteristics:**
- ⚠️ Most manual approach
- ✅ Maximum control over parameters
- ✅ Best for testing and debugging
- ⚠️ Requires specifying all parameters
- ⚠️ Most time-consuming for batch evaluation

**When to Use:**
- You're testing a new configuration
- You need to evaluate a single modality
- You're debugging issues
- You want maximum control

**Example:**
```bash
cd src/eval_vqa
python eval_qwen2_5vl_zeroshot.py \
    --model_path /path/to/model \
    --test_data ../../Splits/modality/test/CT\(Computed\ Tomography\)_test.json \
    --dataset_root ../.. \
    --output_path ../../results/ct_clean.json
```

**Completion Time:** ~1-2 hours per modality

---

## Resource Usage Comparison

### GPU Memory

| Method | GPU Memory per Job | Total GPU-Hours |
|--------|-------------------|-----------------|
| submit_clean_zeroshot.sh | ~60-70GB (shared) | ~1 GPU × 4-6h = 4-6 GPU-hours |
| submit_adversarial_eval.sh | ~60-70GB per job | ~24 jobs × 1-2h = 24-48 GPU-hours |
| batch_eval_adversarial.py | ~60-70GB | ~1 GPU × 8-12h = 8-12 GPU-hours |
| Python Direct | ~60-70GB | ~1 GPU × 1-2h per modality = 8-16 GPU-hours |

### CPU Cores

| Method | CPU Cores | Recommended |
|--------|-----------|-------------|
| submit_clean_zeroshot.sh | 8 | For concurrent processing |
| submit_adversarial_eval.sh | 1 per job | Standard |
| batch_eval_adversarial.py | 4-8 | For data loading |
| Python Direct | 2-4 | Minimal |

### Storage

All methods generate similar output (~100-500MB per modality depending on save_sample_images setting)

---

## Decision Guide

### Choose `submit_clean_zeroshot.sh` if:
- ✅ You only need clean zero-shot evaluation
- ✅ You want the fastest overall completion time
- ✅ You have 1 GPU available
- ✅ You want efficient resource usage

### Choose `submit_adversarial_eval.sh` if:
- ✅ You need adversarial attack evaluation
- ✅ You have multiple GPUs available
- ✅ You want independent job execution
- ✅ You need all attack types (clean, FGSM, PGD)

### Choose `batch_eval_adversarial.py` if:
- ✅ You don't have SLURM access
- ✅ You prefer Python-based execution
- ✅ You want local control
- ✅ You need custom attack parameters

### Choose Python scripts directly if:
- ✅ You're testing configurations
- ✅ You need a single modality
- ✅ You're debugging
- ✅ You want maximum control

---

## Performance Tips

### For Concurrent Evaluation (submit_clean_zeroshot.sh)

1. **Adjust Concurrent Jobs:**
   ```bash
   export CONCURRENT_JOBS=4  # Default: good for 80GB GPU
   export CONCURRENT_JOBS=6  # For high-memory GPUs (>80GB)
   export CONCURRENT_JOBS=2  # For lower-memory GPUs (<40GB)
   ```

2. **Optimize Batch Size:**
   ```bash
   export BATCH_SIZE=8   # Default: balanced
   export BATCH_SIZE=4   # Lower memory usage
   export BATCH_SIZE=16  # Higher throughput (if memory allows)
   ```

3. **Monitor GPU Usage:**
   ```bash
   watch -n 1 nvidia-smi
   ```

### For Sequential Evaluation

1. **Use Higher Batch Size:**
   - Since only one evaluation runs at a time, you can use larger batch sizes
   - Try `BATCH_SIZE=16` or `BATCH_SIZE=32`

2. **Save Resources:**
   - Reduce `save_sample_images` to 0 unless you need visual outputs
   - Use `--attn_implementation sdpa` if flash attention causes issues

---

## Migration Guide

### From submit_adversarial_eval.sh to submit_clean_zeroshot.sh

If you were using:
```bash
./submit_adversarial_eval.sh batch
```

And only need clean evaluation, switch to:
```bash
./submit_clean_zeroshot.sh
```

**Benefits:**
- 4-6x faster completion time
- Uses 1 GPU instead of 24+ jobs
- Same output format

**Trade-off:**
- No FGSM/PGD attacks
- All modalities run on same GPU

---

## Output Format

All methods produce the same JSON output format:

```json
{
  "accuracy": 85.5,
  "total_questions": 1000,
  "correct_answers": 855,
  "results": [
    {
      "image": "path/to/image.png",
      "question": "What imaging technique...",
      "ground_truth": "CT",
      "prediction": "CT",
      "correct": true
    }
  ],
  "metadata": {
    "model_path": "...",
    "attack_type": "clean",
    ...
  }
}
```

This ensures compatibility across all evaluation methods.
