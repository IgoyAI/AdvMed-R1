#!/bin/bash

# Directory configuration
MODEL_PATH="${MODEL_PATH:-/local/scratch/ylai76/Code/R1-V/output/modality_2_5_3B_think/VQA_CT}"
SPLITS_DIR="${SPLITS_DIR:-/local/scratch/ylai76/Code/R1-V/Splits}"
OUTPUT_DIR="${OUTPUT_DIR:-/local/scratch/ylai76/Code/R1-V/src/eval_vqa/logs/clean_zeroshot}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DATASET_ROOT="${DATASET_ROOT:-../..}"
CONCURRENT_JOBS="${CONCURRENT_JOBS:-4}"

echo "========================================"
echo "Clean Zero-Shot Evaluation Submission"
echo "========================================"
echo "MODEL_PATH: $MODEL_PATH"
echo "SPLITS_DIR: $SPLITS_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "DATASET_ROOT: $DATASET_ROOT"
echo "CONCURRENT_JOBS: $CONCURRENT_JOBS"
echo "========================================"
echo ""

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable or update this script"
    exit 1
fi

# Check if splits directory exists
if [ ! -d "$SPLITS_DIR" ]; then
    echo "Error: Splits directory does not exist: $SPLITS_DIR"
    echo "Please set SPLITS_DIR environment variable or update this script"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Submitting concurrent clean zero-shot evaluation job..."
echo ""

# Submit the job
JOB_ID=$(sbatch --parsable \
    run_clean_zeroshot_concurrent.sh \
    "$MODEL_PATH" \
    "$SPLITS_DIR" \
    "$OUTPUT_DIR" \
    "$BATCH_SIZE" \
    "$DATASET_ROOT" \
    "$CONCURRENT_JOBS")

if [ $? -eq 0 ]; then
    echo "✓ Job submitted successfully!"
    echo "  Job ID: $JOB_ID"
    echo "  Job Name: clean_zeroshot_concurrent"
    echo ""
    echo "Monitor job status with:"
    echo "  squeue -j $JOB_ID"
    echo ""
    echo "View job output:"
    echo "  tail -f /local/scratch/ylai76/Code/R1-V/slurm_out/clean_zeroshot_concurrent_${JOB_ID}.out"
    echo ""
    echo "Results will be saved to:"
    echo "  $OUTPUT_DIR"
else
    echo "✗ Job submission failed!"
    exit 1
fi

echo "========================================"
echo ""
echo "Configuration Summary:"
echo "  - Single GPU will be allocated"
echo "  - $CONCURRENT_JOBS modalities will run concurrently"
echo "  - All 8 modalities will be evaluated in batches"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Attack type: clean (no adversarial attacks)"
echo "========================================"
