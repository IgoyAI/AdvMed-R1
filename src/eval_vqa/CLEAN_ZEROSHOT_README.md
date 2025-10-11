# Clean Zero-Shot Concurrent Evaluation

This directory contains scripts for efficient concurrent evaluation of multiple modalities on a single GPU using SLURM.

## Overview

The concurrent evaluation scripts allow you to evaluate all 8 modalities simultaneously on a single GPU by running multiple evaluation processes in parallel. This is more efficient than submitting separate jobs for each modality.

## Key Features

- **Single GPU Usage**: All evaluations run on 1 GPU
- **Concurrent Execution**: Multiple modalities are evaluated in parallel (default: 4 concurrent jobs)
- **Clean Zero-Shot Only**: Focuses on clean evaluation without adversarial attacks
- **Batch Processing**: Modalities are processed in batches to avoid overwhelming the GPU
- **Automatic Summary**: Generates a summary report with accuracy for all modalities

## Scripts

### `submit_clean_zeroshot.sh`

The main submission script that launches the SLURM job.

**Usage:**

```bash
cd src/eval_vqa

# Using default configuration
./submit_clean_zeroshot.sh

# Or with custom environment variables
export MODEL_PATH="/path/to/your/model"
export SPLITS_DIR="/path/to/Splits"
export OUTPUT_DIR="/path/to/output"
export BATCH_SIZE=8
export CONCURRENT_JOBS=4
./submit_clean_zeroshot.sh
```

**Environment Variables:**

- `MODEL_PATH`: Path to the model checkpoint (default: `/local/scratch/ylai76/Code/R1-V/output/modality_2_5_3B_think/VQA_CT`)
- `SPLITS_DIR`: Path to Splits directory (default: `/local/scratch/ylai76/Code/R1-V/Splits`)
- `OUTPUT_DIR`: Output directory for results (default: `/local/scratch/ylai76/Code/R1-V/src/eval_vqa/logs/clean_zeroshot`)
- `BATCH_SIZE`: Batch size for evaluation (default: 8)
- `DATASET_ROOT`: Root directory for dataset (default: `../..`)
- `CONCURRENT_JOBS`: Number of modalities to evaluate concurrently (default: 4)

### `run_clean_zeroshot_concurrent.sh`

The SLURM job script that performs the actual evaluation. This is called by `submit_clean_zeroshot.sh`.

**SLURM Configuration:**

- Nodes: 1
- GPUs: 1
- CPUs: 8
- Memory: 80GB
- Time limit: 80 hours

## Evaluated Modalities

The script evaluates all 8 modalities:

1. **CT** (Computed Tomography)
2. **X-Ray**
3. **MRI** (Magnetic Resonance Imaging)
4. **US** (Ultrasound)
5. **Der** (Dermoscopy)
6. **FP** (Fundus Photography)
7. **Micro** (Microscopy Images)
8. **OCT** (Optical Coherence Tomography)

## Output

Results are saved to individual JSON files in the output directory:

```
output_dir/
├── CT_clean.json
├── X-Ray_clean.json
├── MRI_clean.json
├── US_clean.json
├── Der_clean.json
├── FP_clean.json
├── Micro_clean.json
└── OCT_clean.json
```

Each JSON file contains:
- Accuracy score
- Detailed results for each test sample
- Evaluation metadata

## Monitoring

After submitting the job, you can monitor its progress:

```bash
# Check job status
squeue -u $USER

# View live output
tail -f /local/scratch/ylai76/Code/R1-V/slurm_out/clean_zeroshot_concurrent_<JOB_ID>.out

# View errors (if any)
tail -f /local/scratch/ylai76/Code/R1-V/slurm_out/clean_zeroshot_concurrent_<JOB_ID>.err
```

## Checking Results

After the job completes, you can view a summary of all results:

```bash
cd src/eval_vqa

# Check results in default directory
./check_clean_zeroshot_results.sh

# Or check results in custom directory
./check_clean_zeroshot_results.sh /path/to/output
```

This will display a formatted table showing:
- Accuracy for each modality
- Number of correct answers
- Total questions
- Overall statistics

## Performance Tuning

### Concurrent Jobs

The `CONCURRENT_JOBS` parameter controls how many modalities run in parallel:

- **Lower value (2-3)**: More memory available per evaluation, safer for large models
- **Higher value (4-6)**: Faster overall completion, but requires more GPU memory
- **Recommended**: Start with 4 and adjust based on GPU memory usage

### Batch Size

The `BATCH_SIZE` parameter affects memory usage and speed:

- **Smaller (4-8)**: Lower memory usage, good for concurrent execution
- **Larger (16-32)**: Faster per-modality evaluation, but uses more memory

## Example Workflows

### Basic Usage

```bash
cd src/eval_vqa
./submit_clean_zeroshot.sh
```

### Custom Model and Output Directory

```bash
export MODEL_PATH="/path/to/my/model"
export OUTPUT_DIR="/path/to/my/results"
./submit_clean_zeroshot.sh
```

### High Concurrency (if you have high-memory GPU)

```bash
export CONCURRENT_JOBS=6
export BATCH_SIZE=4
./submit_clean_zeroshot.sh
```

### Conservative Settings (for memory-constrained GPUs)

```bash
export CONCURRENT_JOBS=2
export BATCH_SIZE=4
./submit_clean_zeroshot.sh
```

## Comparison with Other Evaluation Scripts

| Script | GPUs | Modalities | Attacks | Concurrent |
|--------|------|-----------|---------|-----------|
| `submit_adversarial_eval.sh` | Multiple (1 per job) | 1 per job | Clean, FGSM, PGD | No |
| `submit_clean_zeroshot.sh` | 1 | All 8 | Clean only | Yes |

The concurrent script is specifically optimized for:
- Evaluating multiple modalities efficiently
- Clean zero-shot evaluation only
- Single GPU resource allocation
- Faster overall completion time

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. Reduce `CONCURRENT_JOBS`: `export CONCURRENT_JOBS=2`
2. Reduce `BATCH_SIZE`: `export BATCH_SIZE=4`
3. Check GPU memory usage: `nvidia-smi`

### Job Fails to Start

Check if:
- Model path exists and is accessible
- Splits directory exists
- You have permissions to write to output directory
- SLURM queue is not full: `squeue -u $USER`

### Incomplete Results

If some modalities are missing in results:
- Check the error log file for that specific modality
- Look for pattern: `[MODALITY_NAME]` in the output log
- Verify test JSON files exist in Splits directory

## Notes

- The script uses `eval_qwen2_5vl_zeroshot.py` for clean evaluation
- Each modality's output is prefixed with `[MODALITY_NAME]` for easy debugging
- A summary report is generated at the end showing all accuracies
- Failed evaluations are marked but don't stop other modalities from completing
