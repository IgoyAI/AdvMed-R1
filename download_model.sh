#!/bin/bash
# Convenience wrapper for downloading model checkpoints

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python download script
python "$SCRIPT_DIR/src/utils/download_model.py" "$@"
