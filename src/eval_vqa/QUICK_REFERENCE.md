# Quick Reference Guide: Clean Zero-Shot Concurrent Evaluation

## TL;DR - Quick Start

```bash
cd src/eval_vqa

# Check what will be executed (recommended first step)
./submit_clean_zeroshot_dryrun.sh

# Submit the job
./submit_clean_zeroshot.sh

# Check results after completion
./check_clean_zeroshot_results.sh
```

## What This Does

Evaluates **all 8 medical imaging modalities** on **1 GPU** with **concurrent execution** for clean zero-shot evaluation.

### Modalities Evaluated
1. CT (Computed Tomography) - 3,241 samples
2. X-Ray - 1,615 samples
3. MRI (Magnetic Resonance Imaging) - 6,370 samples
4. Ultrasound - 2,074 samples
5. Dermoscopy - 1,306 samples
6. Fundus Photography - 1,098 samples
7. Microscopy - 1,110 samples
8. OCT (Optical Coherence Tomography) - 848 samples

**Total: 17,662 samples**

## Why Use This?

✅ **Efficient**: Uses 1 GPU instead of 24+ separate jobs
✅ **Fast**: Completes in 4-6 hours vs 8-12 hours sequential
✅ **Simple**: Single command submission
✅ **Clean**: Zero-shot evaluation only (no adversarial attacks)
✅ **Concurrent**: Evaluates 4 modalities in parallel by default

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/local/scratch/ylai76/Code/R1-V/output/modality_2_5_3B_think/VQA_CT` | Model checkpoint path |
| `SPLITS_DIR` | `/local/scratch/ylai76/Code/R1-V/Splits` | Test splits directory |
| `OUTPUT_DIR` | `/local/scratch/ylai76/Code/R1-V/src/eval_vqa/logs/clean_zeroshot` | Results output directory |
| `BATCH_SIZE` | `8` | Evaluation batch size |
| `CONCURRENT_JOBS` | `4` | Modalities to run concurrently |

## Customization Examples

### Use Different Model
```bash
export MODEL_PATH="/path/to/your/model"
./submit_clean_zeroshot.sh
```

### Higher Concurrency (for high-memory GPUs)
```bash
export CONCURRENT_JOBS=6
./submit_clean_zeroshot.sh
```

### Lower Memory Usage
```bash
export CONCURRENT_JOBS=2
export BATCH_SIZE=4
./submit_clean_zeroshot.sh
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f /local/scratch/ylai76/Code/R1-V/slurm_out/clean_zeroshot_concurrent_*.out

# Check specific modality progress (search for [MODALITY] tags)
grep "\[CT\]" /local/scratch/ylai76/Code/R1-V/slurm_out/clean_zeroshot_concurrent_*.out
```

## Output Format

Results saved as JSON files:
```
output_dir/
├── CT_clean.json          # CT results
├── X-Ray_clean.json       # X-Ray results
├── MRI_clean.json         # MRI results
├── US_clean.json          # Ultrasound results
├── Der_clean.json         # Dermoscopy results
├── FP_clean.json          # Fundus Photography results
├── Micro_clean.json       # Microscopy results
└── OCT_clean.json         # OCT results
```

Each file contains:
```json
{
  "accuracy": 85.5,
  "total_questions": 1000,
  "correct_answers": 855,
  "results": [...],
  "metadata": {...}
}
```

## Resource Requirements

- **GPU**: 1 x 80GB
- **CPU**: 8 cores
- **Memory**: 80GB RAM
- **Time**: ~4-6 hours
- **Disk**: ~1GB for results

## Troubleshooting

### Out of Memory
```bash
export CONCURRENT_JOBS=2  # Reduce concurrency
export BATCH_SIZE=4       # Reduce batch size
```

### Check Specific Modality
```bash
# Check if CT completed successfully
ls -lh output_dir/CT_clean.json
cat output_dir/CT_clean.json | grep accuracy
```

### Job Not Starting
```bash
# Check SLURM queue
squeue -u $USER

# Check error logs
ls -lt /local/scratch/ylai76/Code/R1-V/slurm_out/*.err | head -1
```

## Files in This Package

| File | Purpose |
|------|---------|
| `submit_clean_zeroshot.sh` | Main submission script |
| `run_clean_zeroshot_concurrent.sh` | SLURM job script (called by submit) |
| `check_clean_zeroshot_results.sh` | Results summary viewer |
| `submit_clean_zeroshot_dryrun.sh` | Preview execution plan |
| `CLEAN_ZEROSHOT_README.md` | Detailed documentation |
| `EVALUATION_COMPARISON.md` | Comparison with other methods |
| `QUICKSTART.md` | Quick start guide (updated) |

## See Also

- **Detailed Documentation**: `CLEAN_ZEROSHOT_README.md`
- **Method Comparison**: `EVALUATION_COMPARISON.md`
- **General Quickstart**: `QUICKSTART.md`
- **Adversarial Evaluation**: `submit_adversarial_eval.sh`

## Support

For issues or questions:
1. Check the dry-run output: `./submit_clean_zeroshot_dryrun.sh`
2. Review logs: `/local/scratch/ylai76/Code/R1-V/slurm_out/`
3. Check SLURM job status: `squeue -u $USER`
4. Review documentation: `CLEAN_ZEROSHOT_README.md`
