#!/bin/bash

# Dry-run script to show what would be executed
# Usage: ./submit_clean_zeroshot_dryrun.sh

# Directory configuration
MODEL_PATH="${MODEL_PATH:-/local/scratch/ylai76/Code/R1-V/output/modality_2_5_3B_think/VQA_CT}"
SPLITS_DIR="${SPLITS_DIR:-/local/scratch/ylai76/Code/R1-V/Splits}"
OUTPUT_DIR="${OUTPUT_DIR:-/local/scratch/ylai76/Code/R1-V/src/eval_vqa/logs/clean_zeroshot}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DATASET_ROOT="${DATASET_ROOT:-../..}"
CONCURRENT_JOBS="${CONCURRENT_JOBS:-4}"

echo "========================================"
echo "DRY RUN - Clean Zero-Shot Evaluation"
echo "========================================"
echo "This is a dry-run. No actual jobs will be submitted."
echo ""
echo "Configuration:"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  SPLITS_DIR: $SPLITS_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  BATCH_SIZE: $BATCH_SIZE"
echo "  DATASET_ROOT: $DATASET_ROOT"
echo "  CONCURRENT_JOBS: $CONCURRENT_JOBS"
echo "========================================"
echo ""

# Check if model path exists
echo "Checking prerequisites..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "  ✗ Model path does not exist: $MODEL_PATH"
    MODEL_EXISTS=false
else
    echo "  ✓ Model path exists"
    MODEL_EXISTS=true
fi

# Check if splits directory exists
if [ ! -d "$SPLITS_DIR" ]; then
    echo "  ✗ Splits directory does not exist: $SPLITS_DIR"
    SPLITS_EXISTS=false
else
    echo "  ✓ Splits directory exists"
    SPLITS_EXISTS=true
fi

echo ""

# Define all modality test files
declare -a MODALITY_FILES=(
    "CT(Computed Tomography)_test.json"
    "X-Ray_test.json"
    "MR (Mag-netic Resonance Imaging)_test.json"
    "ultrasound_test.json"
    "Dermoscopy_test.json"
    "Fundus Photography_test.json"
    "Microscopy Images_test.json"
    "OCT (Optical Coherence Tomography_test.json"
)

declare -a MODALITY_NAMES=(
    "CT"
    "X-Ray"
    "MRI"
    "US"
    "Der"
    "FP"
    "Micro"
    "OCT"
)

# Check test files
echo "Checking test files..."
missing_count=0
for ((i=0; i<${#MODALITY_FILES[@]}; i++)); do
    test_file="$SPLITS_DIR/modality/test/${MODALITY_FILES[$i]}"
    modality_name="${MODALITY_NAMES[$i]}"
    
    if [ -f "$test_file" ]; then
        num_samples=$(grep -o '"image"' "$test_file" | wc -l)
        echo "  ✓ $modality_name: $num_samples samples"
    else
        echo "  ✗ $modality_name: FILE NOT FOUND"
        missing_count=$((missing_count + 1))
    fi
done

echo ""

if [ $missing_count -gt 0 ]; then
    echo "⚠️  Warning: $missing_count test file(s) missing"
    echo ""
fi

# Show execution plan
echo "Execution Plan:"
echo "==============="
echo ""
echo "SLURM Job Configuration:"
echo "  Job Name: clean_zeroshot_concurrent"
echo "  Nodes: 1"
echo "  GPUs: 1"
echo "  CPUs: 8"
echo "  Memory: 80GB"
echo "  Time Limit: 80 hours"
echo ""

echo "Evaluation Strategy:"
echo "  Total Modalities: ${#MODALITY_FILES[@]}"
echo "  Concurrent Jobs: $CONCURRENT_JOBS"
echo "  Number of Batches: $(( (${#MODALITY_FILES[@]} + CONCURRENT_JOBS - 1) / CONCURRENT_JOBS ))"
echo ""

# Show batches
echo "Batch Execution Order:"
total_modalities=${#MODALITY_FILES[@]}
batch_num=1
for ((i=0; i<$total_modalities; i+=$CONCURRENT_JOBS)); do
    echo "  Batch $batch_num:"
    for ((j=0; j<$CONCURRENT_JOBS && i+j<$total_modalities; j++)); do
        idx=$((i+j))
        modality_name="${MODALITY_NAMES[$idx]}"
        echo "    - $modality_name"
    done
    batch_num=$((batch_num + 1))
done

echo ""
echo "Output Files:"
echo "  Directory: $OUTPUT_DIR"
echo "  Files:"
for modality_name in "${MODALITY_NAMES[@]}"; do
    echo "    - ${modality_name}_clean.json"
done

echo ""
echo "========================================"
echo "Estimated Resource Usage:"
echo "  GPU Memory: ~60-70GB (shared across concurrent jobs)"
echo "  Total GPU-Hours: ~4-6 hours (depending on model)"
echo "  Disk Space: ~500MB-1GB for results"
echo "========================================"

echo ""
if [ "$MODEL_EXISTS" = true ] && [ "$SPLITS_EXISTS" = true ] && [ $missing_count -eq 0 ]; then
    echo "✓ All prerequisites met!"
    echo ""
    echo "To submit the job, run:"
    echo "  ./submit_clean_zeroshot.sh"
    echo ""
    echo "Or with custom settings:"
    echo "  export MODEL_PATH=\"/path/to/model\""
    echo "  export CONCURRENT_JOBS=6"
    echo "  ./submit_clean_zeroshot.sh"
else
    echo "✗ Prerequisites not met. Please fix the issues above before submitting."
    exit 1
fi
