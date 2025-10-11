#!/bin/bash

# Directory configuration
MODEL_PATH="${MODEL_PATH:-/local/scratch/ylai76/Code/R1-V/output/modality_2_5_3B_think/VQA_CT}"
SPLITS_DIR="${SPLITS_DIR:-/local/scratch/ylai76/Code/R1-V/Splits}"
OUTPUT_DIR="${OUTPUT_DIR:-/local/scratch/ylai76/Code/R1-V/src/eval_vqa/logs/adversarial_eval}"
DATASET_ROOT="${DATASET_ROOT:-../..}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPSILON="${EPSILON:-0.03}"
PGD_ALPHA="${PGD_ALPHA:-0.01}"
PGD_ITERS="${PGD_ITERS:-10}"
SAVE_SAMPLES="${SAVE_SAMPLES:-3}"

# Attack types to run
ATTACKS=("clean" "fgsm" "pgd")

# Parse command line arguments
EVAL_TYPE="${1:-batch}"  # Options: batch, ct, xray, mri

echo "========================================"
echo "Batch Adversarial Evaluation Submission"
echo "========================================"
echo "Evaluation Type: $EVAL_TYPE"
echo "Model Path: $MODEL_PATH"
echo "Splits Dir: $SPLITS_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epsilon: $EPSILON"
echo "PGD Alpha: $PGD_ALPHA"
echo "PGD Iterations: $PGD_ITERS"
echo "Save Sample Images: $SAVE_SAMPLES"
echo "========================================"
echo ""

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable or update this script"
    exit 1
fi

# Function to submit jobs for a test file
submit_jobs_for_test() {
    local test_file=$1
    local modality_name=$2
    
    if [ ! -f "$test_file" ]; then
        echo "Warning: Test file not found: $test_file"
        return
    fi
    
    echo "Submitting jobs for $modality_name..."
    
    for attack in "${ATTACKS[@]}"; do
        JOB_NAME="adv_eval_${modality_name}_${attack}"
        
        echo "  Submitting $attack attack..."
        sbatch --job-name="$JOB_NAME" \
            run_adversarial_eval_slurm.sh \
            "$MODEL_PATH" \
            "$test_file" \
            "$attack" \
            "$OUTPUT_DIR" \
            "$BATCH_SIZE" \
            "$EPSILON" \
            "$PGD_ALPHA" \
            "$PGD_ITERS" \
            "$DATASET_ROOT" \
            "$SAVE_SAMPLES"
        
        echo "  Job $JOB_NAME submitted!"
    done
    echo ""
}

case $EVAL_TYPE in
    batch)
        echo "Submitting batch evaluation for all modalities..."
        echo ""
        
        # CT modality
        submit_jobs_for_test \
            "$SPLITS_DIR/modality/test/CT(Computed Tomography)_test.json" \
            "CT"
        
        # X-Ray modality
        submit_jobs_for_test \
            "$SPLITS_DIR/modality/test/X-Ray_test.json" \
            "X-Ray"
        
        # MRI modality
        submit_jobs_for_test \
            "$SPLITS_DIR/modality/test/MR (Mag-netic Resonance Imaging)_test.json" \
            "MRI"
        ;;
    
    ct)
        echo "Submitting evaluation for CT modality only..."
        submit_jobs_for_test \
            "$SPLITS_DIR/modality/test/CT(Computed Tomography)_test.json" \
            "CT"
        ;;
    
    xray)
        echo "Submitting evaluation for X-Ray modality only..."
        submit_jobs_for_test \
            "$SPLITS_DIR/modality/test/X-Ray_test.json" \
            "X-Ray"
        ;;
    
    mri)
        echo "Submitting evaluation for MRI modality only..."
        submit_jobs_for_test \
            "$SPLITS_DIR/modality/test/MR (Mag-netic Resonance Imaging)_test.json" \
            "MRI"
        ;;
    
    *)
        echo "Usage: $0 [batch|ct|xray|mri]"
        echo ""
        echo "Options:"
        echo "  batch  - Submit jobs for all modalities (default)"
        echo "  ct     - Submit jobs for CT modality only"
        echo "  xray   - Submit jobs for X-Ray modality only"
        echo "  mri    - Submit jobs for MRI modality only"
        echo ""
        echo "Environment variables:"
        echo "  MODEL_PATH    - Path to model checkpoint"
        echo "  SPLITS_DIR    - Path to Splits directory"
        echo "  OUTPUT_DIR    - Directory for output results"
        echo "  DATASET_ROOT  - Root directory for dataset"
        echo "  BATCH_SIZE    - Batch size (default: 4)"
        echo "  EPSILON       - Perturbation magnitude (default: 0.03)"
        echo "  PGD_ALPHA     - PGD step size (default: 0.01)"
        echo "  PGD_ITERS     - PGD iterations (default: 10)"
        echo "  SAVE_SAMPLES  - Number of sample images to save (default: 3)"
        exit 1
        ;;
esac

echo "========================================"
echo "All jobs submitted!"
echo "Results will be saved to: $OUTPUT_DIR"
echo "========================================"
