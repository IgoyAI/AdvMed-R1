#!/bin/bash

# Adversarial Evaluation Script for Qwen 2.5 VL 3B
# This script runs zero-shot evaluation with FGSM and PGD attacks

# Check PyTorch version (requires 2.6.0+ for flash-attn 2.7.0+)
echo "Checking PyTorch version..."
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "Error: PyTorch is not installed"
    echo "Please run: bash setup.sh"
    exit 1
fi

TORCH_MAJOR=$(echo $TORCH_VERSION | cut -d. -f1)
TORCH_MINOR=$(echo $TORCH_VERSION | cut -d. -f2)

if [ "$TORCH_MAJOR" -lt 2 ] || ([ "$TORCH_MAJOR" -eq 2 ] && [ "$TORCH_MINOR" -lt 6 ]); then
    echo "Error: PyTorch $TORCH_VERSION is installed, but version 2.6.0 or higher is required"
    echo "Please upgrade PyTorch: pip install 'torch>=2.6.0' --upgrade"
    exit 1
fi
echo "✓ PyTorch $TORCH_VERSION detected"
echo ""

# Default configuration
MODEL_PATH=${MODEL_PATH:-"/path/to/Qwen2.5-VL-3B-Instruct"}
DATASET_ROOT=${DATASET_ROOT:-"../.."}
SPLITS_DIR=${SPLITS_DIR:-"../../Splits"}
OUTPUT_DIR=${OUTPUT_DIR:-"../../results/adversarial_eval"}
BATCH_SIZE=${BATCH_SIZE:-8}
EPSILON=${EPSILON:-0.03}
PGD_ALPHA=${PGD_ALPHA:-0.01}
PGD_ITERS=${PGD_ITERS:-10}

# Print configuration
echo "========================================"
echo "Adversarial Evaluation Configuration"
echo "========================================"
echo "Model Path: $MODEL_PATH"
echo "Dataset Root: $DATASET_ROOT"
echo "Splits Dir: $SPLITS_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epsilon: $EPSILON"
echo "PGD Alpha: $PGD_ALPHA"
echo "PGD Iterations: $PGD_ITERS"
echo "========================================"
echo ""

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    echo "Please set MODEL_PATH environment variable or update this script"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run single evaluation
run_evaluation() {
    local test_data=$1
    local attack_type=$2
    local output_name=$3
    
    echo "Running $attack_type evaluation on $output_name..."
    
    python eval_qwen2_5vl_adversarial.py \
        --model_path "$MODEL_PATH" \
        --test_data "$test_data" \
        --dataset_root "$DATASET_ROOT" \
        --output_path "$OUTPUT_DIR/${output_name}_${attack_type}.json" \
        --batch_size "$BATCH_SIZE" \
        --attack_type "$attack_type" \
        --epsilon "$EPSILON" \
        --pgd_alpha "$PGD_ALPHA" \
        --pgd_iters "$PGD_ITERS"
    
    if [ $? -eq 0 ]; then
        echo "✓ $attack_type evaluation completed successfully"
    else
        echo "✗ $attack_type evaluation failed"
    fi
    echo ""
}

# Parse command line arguments
EVAL_TYPE=${1:-"batch"}  # Options: batch, single, ct, xray, mri

case $EVAL_TYPE in
    batch)
        echo "Running batch evaluation on all modalities..."
        python batch_eval_adversarial.py \
            --model_path "$MODEL_PATH" \
            --splits_dir "$SPLITS_DIR" \
            --dataset_root "$DATASET_ROOT" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size "$BATCH_SIZE" \
            --attacks clean fgsm pgd \
            --epsilon "$EPSILON" \
            --pgd_alpha "$PGD_ALPHA" \
            --pgd_iters "$PGD_ITERS" \
            --evaluation_type modality
        ;;
    
    ct)
        echo "Running evaluation on CT modality..."
        TEST_DATA="$SPLITS_DIR/modality/test/CT(Computed Tomography)_test.json"
        run_evaluation "$TEST_DATA" clean ct
        run_evaluation "$TEST_DATA" fgsm ct
        run_evaluation "$TEST_DATA" pgd ct
        ;;
    
    xray)
        echo "Running evaluation on X-Ray modality..."
        TEST_DATA="$SPLITS_DIR/modality/test/X-Ray_test.json"
        run_evaluation "$TEST_DATA" clean xray
        run_evaluation "$TEST_DATA" fgsm xray
        run_evaluation "$TEST_DATA" pgd xray
        ;;
    
    mri)
        echo "Running evaluation on MRI modality..."
        TEST_DATA="$SPLITS_DIR/modality/test/MR (Mag-netic Resonance Imaging)_test.json"
        run_evaluation "$TEST_DATA" clean mri
        run_evaluation "$TEST_DATA" fgsm mri
        run_evaluation "$TEST_DATA" pgd mri
        ;;
    
    single)
        # For custom single evaluation
        if [ -z "$2" ]; then
            echo "Error: Please specify test data path"
            echo "Usage: $0 single <test_data_path> [output_name]"
            exit 1
        fi
        
        TEST_DATA=$2
        OUTPUT_NAME=${3:-"custom"}
        
        echo "Running evaluation on custom test data..."
        run_evaluation "$TEST_DATA" clean "$OUTPUT_NAME"
        run_evaluation "$TEST_DATA" fgsm "$OUTPUT_NAME"
        run_evaluation "$TEST_DATA" pgd "$OUTPUT_NAME"
        ;;
    
    *)
        echo "Usage: $0 [batch|single|ct|xray|mri]"
        echo ""
        echo "Options:"
        echo "  batch  - Run evaluation on all modalities (default)"
        echo "  ct     - Run evaluation on CT modality only"
        echo "  xray   - Run evaluation on X-Ray modality only"
        echo "  mri    - Run evaluation on MRI modality only"
        echo "  single <test_data> [name] - Run evaluation on custom test data"
        echo ""
        echo "Environment variables:"
        echo "  MODEL_PATH    - Path to model checkpoint"
        echo "  DATASET_ROOT  - Root directory for dataset"
        echo "  SPLITS_DIR    - Path to Splits directory"
        echo "  OUTPUT_DIR    - Directory for output results"
        echo "  BATCH_SIZE    - Batch size (default: 8)"
        echo "  EPSILON       - Perturbation magnitude (default: 0.03)"
        echo "  PGD_ALPHA     - PGD step size (default: 0.01)"
        echo "  PGD_ITERS     - PGD iterations (default: 10)"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
