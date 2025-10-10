"""
Simple zero-shot evaluation script for Qwen 2.5 VL 3B without adversarial attacks
Optimized for clean evaluation on OmniMedVQA test splits
"""

import argparse
import json
import os
import re
import warnings
from typing import List, Dict, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Handle flash attention import errors at module import time
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except (ImportError, RuntimeError) as e:
    # If flash attention fails at import time, set environment variable and retry
    error_msg = str(e)
    if 'flash_attn' in error_msg.lower() or 'undefined symbol' in error_msg.lower():
        warnings.warn(
            f"Failed to import transformers due to flash attention error: {error_msg}\n"
            "Setting TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa to bypass flash attention at import time."
        )
        os.environ['TRANSFORMERS_ATTN_IMPLEMENTATION'] = 'sdpa'
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    else:
        raise

from qwen_vl_utils import process_vision_info


class Qwen25VLZeroShotEvaluator:
    """Simple evaluator for Qwen 2.5 VL model - clean zero-shot evaluation"""
    
    def __init__(self, model_path: str, device: str = 'cuda', attn_implementation: str = 'auto'):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the Qwen 2.5 VL model
            device: Device to run the model on
            attn_implementation: Attention implementation to use ('auto', 'flash_attention_2', 'sdpa', 'eager')
        """
        self.device = device
        
        # Determine attention implementation
        if attn_implementation == 'auto':
            # Try flash_attention_2 first, fall back to sdpa/eager if it fails
            try:
                print(f"Loading model from {model_path} with flash_attention_2...")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
                print("✓ Successfully loaded model with flash_attention_2")
            except (ImportError, RuntimeError) as e:
                error_msg = str(e)
                if 'flash_attn' in error_msg.lower() or 'undefined symbol' in error_msg.lower():
                    warnings.warn(
                        f"Failed to load model with flash_attention_2 due to: {error_msg}\n"
                        "Falling back to sdpa attention implementation. "
                        "To fix this, try: pip uninstall flash-attn && pip install flash-attn --no-build-isolation"
                    )
                    print("Loading model with sdpa attention implementation...")
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="sdpa",
                        device_map="auto",
                    )
                    print("✓ Successfully loaded model with sdpa attention")
                else:
                    raise
        else:
            # Use specified attention implementation
            print(f"Loading model from {model_path} with {attn_implementation}...")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                device_map="auto",
            )
            print(f"✓ Successfully loaded model with {attn_implementation}")
        
        self.model.eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"
        
        self.question_template = "{Question} Think through the question step by step in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags."
    
    def load_test_data(self, json_path: str, dataset_root: str = ".") -> List[Dict]:
        """Load test data from JSON file"""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Update image paths to be absolute
        for item in data:
            if not os.path.isabs(item['image']):
                item['image'] = os.path.join(dataset_root, item['image'])
        
        return data
    
    def prepare_messages(self, data: List[Dict]) -> List[Dict]:
        """Prepare messages in the format expected by the model"""
        messages = []
        for item in data:
            message = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{item['image']}"
                    },
                    {
                        "type": "text",
                        "text": self.question_template.format(Question=item['problem'])
                    }
                ]
            }]
            messages.append(message)
        return messages
    
    def evaluate_batch(self, messages: List[Dict], batch_size: int = 8) -> List[str]:
        """Evaluate a batch of samples"""
        all_outputs = []
        
        for i in tqdm(range(0, len(messages), batch_size), desc="Evaluating"):
            batch_messages = messages[i:i + batch_size]
            
            # Prepare batch for inference
            text = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                   for msg in batch_messages]
            
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate outputs
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    use_cache=True, 
                    max_new_tokens=256, 
                    do_sample=False
                )
            
            # Decode outputs
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            batch_output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            all_outputs.extend(batch_output_text)
        
        return all_outputs
    
    @staticmethod
    def extract_answer(output_str: str) -> Optional[str]:
        """Extract answer from model output"""
        answer_pattern = r'<answer>\s*(\w+)\s*</answer>'
        match = re.search(answer_pattern, output_str, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        return None
    
    def compute_accuracy(self, outputs: List[str], ground_truths: List[str]) -> Tuple[float, List[Dict]]:
        """Compute accuracy and detailed results"""
        correct_count = 0
        detailed_results = []
        
        for output, ground_truth in zip(outputs, ground_truths):
            model_answer = self.extract_answer(output)
            gt_answer = self.extract_answer(ground_truth)
            
            is_correct = (model_answer is not None and 
                         gt_answer is not None and 
                         model_answer.upper() == gt_answer.upper())
            
            if is_correct:
                correct_count += 1
            
            detailed_results.append({
                'model_output': output,
                'model_answer': model_answer,
                'ground_truth': ground_truth,
                'gt_answer': gt_answer,
                'correct': is_correct
            })
        
        accuracy = (correct_count / len(outputs) * 100) if len(outputs) > 0 else 0.0
        return accuracy, detailed_results
    
    def run_evaluation(self, test_data_path: str, dataset_root: str, 
                      output_path: str, batch_size: int = 8):
        """Run complete evaluation pipeline"""
        # Load data
        print(f"Loading test data from {test_data_path}...")
        data = self.load_test_data(test_data_path, dataset_root)
        print(f"Loaded {len(data)} samples")
        
        # Prepare messages
        messages = self.prepare_messages(data)
        
        # Evaluate
        print(f"\nRunning zero-shot evaluation...")
        outputs = self.evaluate_batch(messages, batch_size)
        
        # Compute accuracy
        ground_truths = [item['solution'] for item in data]
        accuracy, detailed_results = self.compute_accuracy(outputs, ground_truths)
        
        print(f"\nAccuracy: {accuracy:.2f}%")
        
        # Save results
        results = {
            'test_data_path': test_data_path,
            'model_evaluation': 'zero_shot_clean',
            'accuracy': accuracy,
            'num_samples': len(data),
            'num_correct': sum(1 for r in detailed_results if r['correct']),
            'detailed_results': detailed_results
        }
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")
        
        return accuracy


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of Qwen 2.5 VL 3B (clean, no adversarial attacks)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Qwen 2.5 VL model checkpoint"
    )
    
    # Data arguments
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data JSON file (from Splits directory)"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=".",
        help="Root directory where OmniMedVQA dataset is located"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention implementation to use. 'auto' tries flash_attention_2 first, falls back to sdpa if unavailable"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Qwen25VLZeroShotEvaluator(args.model_path, attn_implementation=args.attn_implementation)
    
    # Run evaluation
    evaluator.run_evaluation(
        test_data_path=args.test_data,
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
