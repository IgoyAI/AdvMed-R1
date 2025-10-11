#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/local/scratch/ylai76/Code/R1-V/slurm_out/%x_%j.out
#SBATCH --error=/local/scratch/ylai76/Code/R1-V/slurm_out/%x_%j.err

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
TEST_DATA="$2"
ATTACK_TYPE="$3"
OUTPUT_DIR="$4"
BATCH_SIZE="${5:-4}"
EPSILON="${6:-0.03}"
PGD_ALPHA="${7:-0.01}"
PGD_ITERS="${8:-10}"
DATASET_ROOT="${9:-../..}"
SAVE_SAMPLES="${10:-3}"

# Ensure required parameters are not empty
if [ -z "$MODEL_PATH" ] || [ -z "$TEST_DATA" ] || [ -z "$ATTACK_TYPE" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required parameters!"
    echo "Usage: sbatch run_adversarial_eval_slurm.sh MODEL_PATH TEST_DATA ATTACK_TYPE OUTPUT_DIR [BATCH_SIZE] [EPSILON] [PGD_ALPHA] [PGD_ITERS] [DATASET_ROOT] [SAVE_SAMPLES]"
    exit 1
fi

# Generate output path
FILE_NAME=$(basename -- "$TEST_DATA")
OUTPUT_NAME="${FILE_NAME%.json}"
OUTPUT_PATH="$OUTPUT_DIR/${OUTPUT_NAME}_${ATTACK_TYPE}.json"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Adversarial Evaluation Configuration"
echo "========================================"
echo "MODEL_PATH: $MODEL_PATH"
echo "TEST_DATA: $TEST_DATA"
echo "ATTACK_TYPE: $ATTACK_TYPE"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "EPSILON: $EPSILON"
echo "PGD_ALPHA: $PGD_ALPHA"
echo "PGD_ITERS: $PGD_ITERS"
echo "DATASET_ROOT: $DATASET_ROOT"
echo "SAVE_SAMPLES: $SAVE_SAMPLES"
echo "========================================"

# Run Python script
python eval_qwen2_5vl_adversarial.py \
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_DATA" \
    --dataset_root "$DATASET_ROOT" \
    --output_path "$OUTPUT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --attack_type "$ATTACK_TYPE" \
    --epsilon "$EPSILON" \
    --pgd_alpha "$PGD_ALPHA" \
    --pgd_iters "$PGD_ITERS" \
    --save_sample_images "$SAVE_SAMPLES"

if [ $? -eq 0 ]; then
    echo "✓ Adversarial evaluation completed successfully"
    echo "Results saved to: $OUTPUT_PATH"
else
    echo "✗ Adversarial evaluation failed"
    exit 1
fi

echo "Finished processing $TEST_DATA with $ATTACK_TYPE attack"
