#!/bin/bash

# Check results from clean zero-shot concurrent evaluation
# Usage: ./check_clean_zeroshot_results.sh [OUTPUT_DIR]

OUTPUT_DIR="${1:-/local/scratch/ylai76/Code/R1-V/src/eval_vqa/logs/clean_zeroshot}"

echo "========================================"
echo "Clean Zero-Shot Evaluation Results"
echo "========================================"
echo "Output Directory: $OUTPUT_DIR"
echo ""

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory does not exist: $OUTPUT_DIR"
    exit 1
fi

# Modality names
MODALITIES=("CT" "X-Ray" "MRI" "US" "Der" "FP" "Micro" "OCT")
MODALITY_FULL_NAMES=(
    "CT (Computed Tomography)"
    "X-Ray"
    "MRI (Magnetic Resonance Imaging)"
    "Ultrasound"
    "Dermoscopy"
    "Fundus Photography"
    "Microscopy"
    "OCT (Optical Coherence Tomography)"
)

# Print header
printf "%-45s %10s %10s %10s\n" "Modality" "Accuracy" "Correct" "Total"
printf "%-45s %10s %10s %10s\n" "--------" "--------" "-------" "-----"

total_correct=0
total_questions=0
completed=0

# Check each modality
for ((i=0; i<${#MODALITIES[@]}; i++)); do
    modality="${MODALITIES[$i]}"
    modality_full="${MODALITY_FULL_NAMES[$i]}"
    output_file="$OUTPUT_DIR/${modality}_clean.json"
    
    if [ -f "$output_file" ]; then
        # Extract metrics using Python
        result=$(python3 -c "
import json
import sys
try:
    with open('$output_file', 'r') as f:
        data = json.load(f)
        acc = data.get('accuracy', 0)
        correct = data.get('correct_answers', 0)
        total = data.get('total_questions', 0)
        print(f'{acc:.2f}|{correct}|{total}')
except Exception as e:
    print('ERROR|0|0')
    sys.exit(1)
" 2>/dev/null)
        
        if [ $? -eq 0 ] && [ "$result" != "ERROR|0|0" ]; then
            IFS='|' read -r accuracy correct total <<< "$result"
            printf "%-45s %9.2f%% %10d %10d\n" "$modality_full" "$accuracy" "$correct" "$total"
            total_correct=$((total_correct + correct))
            total_questions=$((total_questions + total))
            completed=$((completed + 1))
        else
            printf "%-45s %10s %10s %10s\n" "$modality_full" "ERROR" "-" "-"
        fi
    else
        printf "%-45s %10s %10s %10s\n" "$modality_full" "MISSING" "-" "-"
    fi
done

echo ""
echo "========================================"

# Calculate overall statistics
if [ $total_questions -gt 0 ]; then
    overall_accuracy=$(echo "scale=2; $total_correct * 100 / $total_questions" | bc)
    echo "Overall Statistics:"
    echo "  Completed Modalities: $completed / ${#MODALITIES[@]}"
    echo "  Total Questions: $total_questions"
    echo "  Total Correct: $total_correct"
    echo "  Overall Accuracy: ${overall_accuracy}%"
else
    echo "No results found or all evaluations failed."
fi

echo "========================================"

# Check for detailed error logs if some results are missing
missing=$((${#MODALITIES[@]} - completed))
if [ $missing -gt 0 ]; then
    echo ""
    echo "⚠️  $missing modality/modalities missing or failed"
    echo "Check SLURM output logs for details:"
    echo "  ls -lt /local/scratch/ylai76/Code/R1-V/slurm_out/clean_zeroshot_concurrent_*.{out,err}"
fi
