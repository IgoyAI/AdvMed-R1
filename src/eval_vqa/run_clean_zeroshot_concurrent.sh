#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/local/scratch/ylai76/Code/R1-V/slurm_out/%x_%j.out
#SBATCH --error=/local/scratch/ylai76/Code/R1-V/slurm_out/%x_%j.err
#SBATCH --job-name=clean_zeroshot_concurrent

# Load environment
source ~/.bashrc
conda activate r1-v

export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Check GPU
echo "Checking GPU..."
nvidia-smi
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Cache directories
export TRANSFORMERS_CACHE=/local/scratch/ylai76/cache
export TRITON_CACHE=/local/scratch/ylai76/triton_cache
export HF_HOME=/local/scratch/ylai76/huggingface_cache

echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "TRITON_CACHE: $TRITON_CACHE"
echo "HF_HOME: $HF_HOME"

# Navigate to code directory
cd /local/scratch/ylai76/Code/R1-V/src/eval_vqa

# Get parameters passed from sbatch
MODEL_PATH="$1"
SPLITS_DIR="$2"
OUTPUT_DIR="$3"
BATCH_SIZE="${4:-8}"
DATASET_ROOT="${5:-../..}"
CONCURRENT_JOBS="${6:-4}"  # Number of concurrent evaluations to run

# Ensure required parameters are not empty
if [ -z "$MODEL_PATH" ] || [ -z "$SPLITS_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required parameters!"
    echo "Usage: sbatch run_clean_zeroshot_concurrent.sh MODEL_PATH SPLITS_DIR OUTPUT_DIR [BATCH_SIZE] [DATASET_ROOT] [CONCURRENT_JOBS]"
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Clean Zero-Shot Concurrent Evaluation"
echo "========================================"
echo "MODEL_PATH: $MODEL_PATH"
echo "SPLITS_DIR: $SPLITS_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "DATASET_ROOT: $DATASET_ROOT"
echo "CONCURRENT_JOBS: $CONCURRENT_JOBS"
echo "========================================"

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

# Define corresponding short names for output files
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

# Function to run evaluation for a single modality
run_evaluation() {
    local test_file=$1
    local modality_name=$2
    local full_path="$SPLITS_DIR/modality/test/$test_file"
    local output_path="$OUTPUT_DIR/${modality_name}_clean.json"
    
    if [ ! -f "$full_path" ]; then
        echo "Warning: Test file not found: $full_path"
        return 1
    fi
    
    echo "Starting evaluation for $modality_name..."
    
    python eval_qwen2_5vl_zeroshot.py \
        --model_path "$MODEL_PATH" \
        --test_data "$full_path" \
        --dataset_root "$DATASET_ROOT" \
        --output_path "$output_path" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | sed "s/^/[$modality_name] /"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ $modality_name evaluation completed successfully"
    else
        echo "✗ $modality_name evaluation failed with exit code $exit_code"
    fi
    
    return $exit_code
}

# Export function and variables for parallel execution
export -f run_evaluation
export MODEL_PATH SPLITS_DIR OUTPUT_DIR BATCH_SIZE DATASET_ROOT

echo ""
echo "Starting concurrent evaluations..."
echo "Running $CONCURRENT_JOBS evaluations in parallel"
echo "========================================"
echo ""

# Track PIDs for monitoring
declare -a PIDS=()
declare -a RUNNING_MODALITIES=()

# Run evaluations in batches of CONCURRENT_JOBS
total_modalities=${#MODALITY_FILES[@]}
for ((i=0; i<$total_modalities; i+=$CONCURRENT_JOBS)); do
    # Start a batch of concurrent jobs
    for ((j=0; j<$CONCURRENT_JOBS && i+j<$total_modalities; j++)); do
        idx=$((i+j))
        modality_file="${MODALITY_FILES[$idx]}"
        modality_name="${MODALITY_NAMES[$idx]}"
        
        # Run evaluation in background
        run_evaluation "$modality_file" "$modality_name" &
        pid=$!
        PIDS+=($pid)
        RUNNING_MODALITIES+=("$modality_name")
        
        echo "Started $modality_name (PID: $pid)"
    done
    
    # Wait for this batch to complete before starting next batch
    echo ""
    echo "Waiting for batch to complete..."
    for ((j=0; j<${#PIDS[@]}; j++)); do
        pid=${PIDS[$j]}
        modality=${RUNNING_MODALITIES[$j]}
        if wait $pid; then
            echo "✓ $modality completed successfully"
        else
            echo "✗ $modality failed"
        fi
    done
    
    # Clear arrays for next batch
    PIDS=()
    RUNNING_MODALITIES=()
    
    echo ""
    echo "Batch $((i/$CONCURRENT_JOBS + 1)) completed"
    echo "========================================"
    echo ""
done

echo ""
echo "========================================"
echo "All evaluations completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# Generate summary report
echo ""
echo "Summary of Results:"
echo "-------------------"
for ((i=0; i<${#MODALITY_NAMES[@]}; i++)); do
    modality_name="${MODALITY_NAMES[$i]}"
    output_path="$OUTPUT_DIR/${modality_name}_clean.json"
    
    if [ -f "$output_path" ]; then
        accuracy=$(python -c "import json; f=open('$output_path'); d=json.load(f); print(f\"{d['accuracy']:.2f}\")" 2>/dev/null || echo "N/A")
        echo "  $modality_name: $accuracy%"
    else
        echo "  $modality_name: MISSING"
    fi
done
echo "========================================"
