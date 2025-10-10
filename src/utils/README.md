# Model Download Utility

Automatic model checkpoint downloader for AdvMed-R1 project.

## Features

- 🚀 **Easy to use**: Simple command-line interface
- 📦 **Pre-configured models**: Quick shortcuts for common models
- 🔄 **Resume support**: Automatically resume interrupted downloads
- 🔐 **Private model support**: Authentication with Hugging Face tokens
- ⚡ **Fast downloads**: Multi-threaded downloading

## Installation

The download utility requires `huggingface-hub`:

```bash
pip install huggingface-hub
```

This is already included in the project's `requirements.txt`.

## Quick Start

### List Available Models

```bash
python src/utils/download_model.py --list
```

### Download Pre-configured Models

```bash
# Download Qwen2.5-VL-3B-Instruct
python src/utils/download_model.py --model qwen2.5-vl-3b --output ./models/Qwen2.5-VL-3B-Instruct

# Download Med-R1 checkpoint
python src/utils/download_model.py --model med-r1 --output ./checkpoints/Med-R1

# Download Qwen2-VL-7B
python src/utils/download_model.py --model qwen2-vl-7b --output ./models/Qwen2-VL-7B
```

### Download Any Model from Hugging Face

```bash
# Using full repository ID
python src/utils/download_model.py --model Qwen/Qwen2.5-VL-3B-Instruct --output ./models

# Download other models
python src/utils/download_model.py --model facebook/opt-125m --output ./models/opt-125m
```

## Usage

### Basic Usage

```bash
python src/utils/download_model.py --model MODEL_NAME --output OUTPUT_DIR
```

### With Authentication Token

For private models or to increase rate limits:

```bash
python src/utils/download_model.py \
    --model MODEL_NAME \
    --output OUTPUT_DIR \
    --token YOUR_HF_TOKEN
```

Get your token from: https://huggingface.co/settings/tokens

### Advanced Options

```bash
python src/utils/download_model.py \
    --model qwen2.5-vl-3b \
    --output ./models/Qwen2.5-VL-3B-Instruct \
    --max-workers 8 \              # Use 8 concurrent download threads
    --no-resume                     # Don't resume interrupted downloads
```

## Pre-configured Models

The following models have shortcuts for convenience:

| Shortcut | Repository ID | Description |
|----------|---------------|-------------|
| `qwen2.5-vl-3b` | `Qwen/Qwen2.5-VL-3B-Instruct` | Qwen2.5-VL-3B-Instruct base model |
| `qwen2-vl-7b` | `Qwen/Qwen2-VL-7B-Instruct` | Qwen2-VL-7B-Instruct base model |
| `med-r1` | `yuxianglai117/Med-R1` | Med-R1 trained checkpoint |

## Command-line Options

```
--model, -m          Model name/ID or shortcut (required)
--output, -o         Output directory to save the model (required)
--token, -t          Hugging Face API token (optional)
--no-resume          Don't resume interrupted downloads
--max-workers        Number of concurrent download threads (default: 4)
--list, -l           List available pre-configured models
--help, -h           Show help message
```

## Examples

### Example 1: Download Base Model

```bash
python src/utils/download_model.py \
    --model qwen2.5-vl-3b \
    --output ./models/Qwen2.5-VL-3B-Instruct
```

### Example 2: Download with Authentication

```bash
export HF_TOKEN="your_token_here"
python src/utils/download_model.py \
    --model med-r1 \
    --output ./checkpoints/Med-R1 \
    --token $HF_TOKEN
```

### Example 3: Fast Download

```bash
python src/utils/download_model.py \
    --model qwen2.5-vl-3b \
    --output ./models \
    --max-workers 16
```

## Troubleshooting

### Authentication Errors

If you get a 401 or 403 error:
- The model may be private or gated
- Provide a valid Hugging Face token using `--token`
- Make sure your token has read access
- Accept the model's license agreement on Hugging Face

### Model Not Found (404)

- Check the model name/ID is correct
- Verify the model exists on Hugging Face Hub
- Some models may have been renamed or removed

### Download Interrupted

The script automatically resumes interrupted downloads by default. If you want to start fresh:

```bash
python src/utils/download_model.py --model MODEL --output DIR --no-resume
```

### Slow Downloads

- Increase the number of workers: `--max-workers 8`
- Check your internet connection
- Some models are very large (3B model ≈ 6GB)

## Integration with Training Scripts

Once downloaded, use the model path in training scripts:

```bash
# Training example
torchrun --nproc_per_node=2 \
    src/open_r1/grpo_vqa_nothink.py \
    --model_name_or_path ./models/Qwen2.5-VL-3B-Instruct \
    --output_dir ./output/Modality_CT \
    ...
```

## Programmatic Usage

You can also use the download function in your Python code:

```python
from src.utils.download_model import download_model

# Download a model
model_path = download_model(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    output_dir="./models/Qwen2.5-VL-3B-Instruct",
    token=None,  # Optional
    resume=True,
    max_workers=4
)

print(f"Model downloaded to: {model_path}")
```

## Support

For issues or questions:
1. Check the [main README](../../README.md)
2. Review the [QUICKSTART guide](../eval_vqa/QUICKSTART.md)
3. Open an issue on GitHub
