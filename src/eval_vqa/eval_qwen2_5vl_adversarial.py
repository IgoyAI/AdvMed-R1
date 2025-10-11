"""
Zero-shot evaluation script for Qwen 2.5 VL 3B with adversarial attacks (FGSM and PGD)
Supports evaluation on OmniMedVQA dataset using Splits directory structure
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
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
import numpy as np


class AdversarialAttacker:
    """Class for generating adversarial examples using FGSM and PGD attacks"""
    
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
    
    def fgsm_attack(self, image_tensor: torch.Tensor, epsilon: float, gradient: torch.Tensor) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack
        
        Args:
            image_tensor: Original image tensor
            epsilon: Perturbation magnitude
            gradient: Gradient of loss w.r.t. input
            
        Returns:
            Adversarial image tensor
        """
        # Compute perturbation
        sign_gradient = gradient.sign()
        perturbed_image = image_tensor + epsilon * sign_gradient
        
        # Clip to maintain valid pixel range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image
    
    def pgd_attack(self, image_tensor: torch.Tensor, epsilon: float, alpha: float, 
                   num_iter: int, gradient_fn) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack
        
        Args:
            image_tensor: Original image tensor
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            num_iter: Number of iterations
            gradient_fn: Function to compute gradient
            
        Returns:
            Adversarial image tensor
        """
        perturbed_image = image_tensor.clone().detach()
        original_image = image_tensor.clone().detach()
        
        for i in range(num_iter):
            perturbed_image.requires_grad = True
            gradient = gradient_fn(perturbed_image)
            
            # Update with gradient step
            perturbed_image = perturbed_image.detach() + alpha * gradient.sign()
            
            # Project back to epsilon ball
            perturbation = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
            perturbed_image = torch.clamp(original_image + perturbation, 0, 1)
        
        return perturbed_image


class Qwen25VLEvaluator:
    """Evaluator for Qwen 2.5 VL model with adversarial attack support"""
    
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
        
        # Initialize adversarial attacker
        self.attacker = AdversarialAttacker(self.model, self.processor, device)
        
        self.question_template = "{Question} Think through the question step by step in <think>...</think> tags. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags."
    
    def load_test_data(self, json_path: str, dataset_root: str = ".") -> List[Dict]:
        """
        Load test data from JSON file
        
        Args:
            json_path: Path to the JSON file containing test data
            dataset_root: Root directory for the dataset (where OmniMedVQA folder is)
            
        Returns:
            List of data samples
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Update image paths to be absolute
        for item in data:
            if not os.path.isabs(item['image']):
                item['image'] = os.path.join(dataset_root, item['image'])
        
        return data
    
    def prepare_messages(self, data: List[Dict]) -> List[Dict]:
        """
        Prepare messages in the format expected by the model
        
        Args:
            data: List of data samples
            
        Returns:
            List of formatted messages
        """
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
    
    def compute_loss_for_attack(self, inputs: Dict, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for adversarial attack
        
        Args:
            inputs: Model inputs
            target_ids: Target token IDs
            
        Returns:
            Loss tensor
        """
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        # Calculate loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        
        return loss
    
    def generate_adversarial_image(self, image_path: str, text_input: str, 
                                   attack_type: str, epsilon: float = 0.03,
                                   pgd_alpha: float = 0.01, pgd_iters: int = 10) -> Image.Image:
        """
        Generate adversarial example for a given image
        
        Args:
            image_path: Path to the original image
            text_input: Text prompt
            attack_type: Type of attack ('fgsm' or 'pgd')
            epsilon: Perturbation magnitude
            pgd_alpha: Step size for PGD
            pgd_iters: Number of iterations for PGD
            
        Returns:
            Adversarial image as PIL Image
        """
        # Load and prepare image
        original_image = Image.open(image_path).convert('RGB')
        
        # Prepare input
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": original_image},
                {"type": "text", "text": text_input}
            ]
        }]
        
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[original_image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get pixel values
        pixel_values = inputs['pixel_values'].float()
        pixel_values.requires_grad = True
        
        # Create a simple target (maximize loss)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        # Function to compute gradient
        def compute_gradient(perturbed_pixels):
            inputs_copy = inputs.copy()
            inputs_copy['pixel_values'] = perturbed_pixels.to(self.model.dtype)
            
            # Forward pass
            outputs = self.model(**inputs_copy)
            logits = outputs.logits
            
            # Use the model's own prediction as target and maximize loss
            target = torch.argmax(logits[:, -1, :], dim=-1)
            loss = F.cross_entropy(logits[:, -1, :], target)
            
            # Compute gradient
            if perturbed_pixels.grad is not None:
                perturbed_pixels.grad.zero_()
            loss.backward()
            
            return perturbed_pixels.grad.data
        
        # Generate adversarial example
        if attack_type == 'fgsm':
            gradient = compute_gradient(pixel_values)
            perturbed_pixels = self.attacker.fgsm_attack(pixel_values, epsilon, gradient)
        elif attack_type == 'pgd':
            perturbed_pixels = self.attacker.pgd_attack(
                pixel_values, epsilon, pgd_alpha, pgd_iters, compute_gradient
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Convert back to PIL Image
        perturbed_pixels = perturbed_pixels.detach().cpu()
        
        # Denormalize and convert to image
        # Note: The exact denormalization depends on the processor's normalization
        # For now, we'll use a simplified approach
        perturbed_array = perturbed_pixels[0].permute(1, 2, 0).numpy()
        perturbed_array = np.clip(perturbed_array * 255, 0, 255).astype(np.uint8)
        adversarial_image = Image.fromarray(perturbed_array)
        
        return adversarial_image
    
    def evaluate_batch(self, messages: List[Dict], batch_size: int = 8, 
                      attack_type: Optional[str] = None, epsilon: float = 0.03,
                      pgd_alpha: float = 0.01, pgd_iters: int = 10) -> List[str]:
        """
        Evaluate a batch of samples
        
        Args:
            messages: List of formatted messages
            batch_size: Batch size for inference
            attack_type: Type of attack ('fgsm', 'pgd', or None for clean)
            epsilon: Perturbation magnitude for attacks
            pgd_alpha: Step size for PGD
            pgd_iters: Number of iterations for PGD
            
        Returns:
            List of model outputs
        """
        all_outputs = []
        
        for i in tqdm(range(0, len(messages), batch_size), desc="Evaluating"):
            batch_messages = messages[i:i + batch_size]
            
            # If attack is specified, generate adversarial examples
            if attack_type is not None:
                # For adversarial attacks, process one by one (more complex)
                # This is a simplified version - in practice, you'd need more sophisticated handling
                batch_messages_adv = []
                for msg in batch_messages:
                    # Extract image path from message
                    image_path = msg[0]['content'][0]['image'].replace('file://', '')
                    text_input = msg[0]['content'][1]['text']
                    
                    # Generate adversarial image
                    adv_image = self.generate_adversarial_image(
                        image_path, text_input, attack_type, epsilon, pgd_alpha, pgd_iters
                    )
                    
                    # Update message with adversarial image
                    msg_adv = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": adv_image},
                            {"type": "text", "text": text_input}
                        ]
                    }]
                    batch_messages_adv.append(msg_adv)
                
                batch_messages = batch_messages_adv
            
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
        """
        Extract answer from model output
        
        Args:
            output_str: Model output string
            
        Returns:
            Extracted answer or None
        """
        answer_pattern = r'<answer>\s*(\w+)\s*</answer>'
        match = re.search(answer_pattern, output_str, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        return None
    
    def compute_accuracy(self, outputs: List[str], ground_truths: List[str]) -> Tuple[float, List[Dict]]:
        """
        Compute accuracy and detailed results
        
        Args:
            outputs: List of model outputs
            ground_truths: List of ground truth solutions
            
        Returns:
            Tuple of (accuracy, detailed_results)
        """
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
    
    def save_sample_perturbed_images(self, data: List[Dict], attack_type: str, 
                                     epsilon: float, pgd_alpha: float, pgd_iters: int,
                                     num_samples: int, output_path: str):
        """
        Save sample perturbed images alongside originals for visualization
        
        Args:
            data: List of data samples
            attack_type: Type of attack ('fgsm' or 'pgd')
            epsilon: Perturbation magnitude
            pgd_alpha: Step size for PGD
            pgd_iters: Number of iterations for PGD
            num_samples: Number of samples to save
            output_path: Path where results are saved (used to determine image output directory)
        """
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Create output directory for images
        output_dir = os.path.dirname(output_path)
        images_dir = os.path.join(output_dir, "sample_images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Select random samples to visualize
        num_samples = min(num_samples, len(data))
        sample_indices = np.random.choice(len(data), num_samples, replace=False)
        
        for idx in sample_indices:
            sample = data[idx]
            image_path = sample['image']
            text_input = self.question_template.format(Question=sample['problem'])
            
            try:
                # Load original image
                original_image = Image.open(image_path).convert('RGB')
                
                # Generate adversarial image
                adversarial_image = self.generate_adversarial_image(
                    image_path, text_input, attack_type, epsilon, pgd_alpha, pgd_iters
                )
                
                # Create comparison figure
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original image
                axes[0].imshow(original_image)
                axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
                axes[0].axis('off')
                
                # Adversarial image
                axes[1].imshow(adversarial_image)
                attack_title = f'{attack_type.upper()} Attack (ε={epsilon}'
                if attack_type == 'pgd':
                    attack_title += f', α={pgd_alpha}, iters={pgd_iters}'
                attack_title += ')'
                axes[1].set_title(attack_title, fontsize=14, fontweight='bold')
                axes[1].axis('off')
                
                plt.tight_layout()
                
                # Save figure
                output_filename = f"sample_{idx}_{attack_type}_eps{epsilon}.png"
                output_image_path = os.path.join(images_dir, output_filename)
                plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"  Saved sample {idx + 1}/{num_samples}: {output_filename}")
                
            except Exception as e:
                print(f"  Warning: Failed to save sample {idx}: {e}")
                continue
        
        print(f"Sample images saved to: {images_dir}")
    
    def run_evaluation(self, test_data_path: str, dataset_root: str, 
                      output_path: str, batch_size: int = 8,
                      attack_type: Optional[str] = None, epsilon: float = 0.03,
                      pgd_alpha: float = 0.01, pgd_iters: int = 10,
                      save_sample_images: int = 0):
        """
        Run complete evaluation pipeline
        
        Args:
            test_data_path: Path to test data JSON file
            dataset_root: Root directory for dataset
            output_path: Path to save results
            batch_size: Batch size for inference
            attack_type: Type of attack (None, 'fgsm', 'pgd')
            epsilon: Perturbation magnitude
            pgd_alpha: Step size for PGD
            pgd_iters: Number of iterations for PGD
            save_sample_images: Number of sample perturbed images to save (0 = none)
        """
        # Load data
        print(f"Loading test data from {test_data_path}...")
        data = self.load_test_data(test_data_path, dataset_root)
        print(f"Loaded {len(data)} samples")
        
        # Prepare messages
        messages = self.prepare_messages(data)
        
        # Evaluate
        attack_str = f"{attack_type.upper()} (eps={epsilon})" if attack_type else "Clean"
        print(f"\nRunning evaluation with {attack_str}...")
        outputs = self.evaluate_batch(
            messages, batch_size, attack_type, epsilon, pgd_alpha, pgd_iters
        )
        
        # Save sample perturbed images if requested
        if save_sample_images > 0 and attack_type is not None:
            print(f"\nSaving {save_sample_images} sample perturbed images...")
            self.save_sample_perturbed_images(
                data, attack_type, epsilon, pgd_alpha, pgd_iters, 
                save_sample_images, output_path
            )
        
        # Compute accuracy
        ground_truths = [item['solution'] for item in data]
        accuracy, detailed_results = self.compute_accuracy(outputs, ground_truths)
        
        print(f"\nAccuracy ({attack_str}): {accuracy:.2f}%")
        
        # Save results
        results = {
            'test_data_path': test_data_path,
            'attack_type': attack_type if attack_type else 'clean',
            'epsilon': epsilon if attack_type else None,
            'pgd_alpha': pgd_alpha if attack_type == 'pgd' else None,
            'pgd_iters': pgd_iters if attack_type == 'pgd' else None,
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
        description="Zero-shot evaluation of Qwen 2.5 VL 3B with adversarial attacks"
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
    
    # Attack arguments
    parser.add_argument(
        "--attack_type",
        type=str,
        choices=['clean', 'fgsm', 'pgd'],
        default='clean',
        help="Type of evaluation: clean (no attack), fgsm, or pgd"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.03,
        help="Perturbation magnitude for adversarial attacks"
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=0.01,
        help="Step size for PGD attack"
    )
    parser.add_argument(
        "--pgd_iters",
        type=int,
        default=10,
        help="Number of iterations for PGD attack"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
        help="Attention implementation to use. 'auto' tries flash_attention_2 first, falls back to sdpa if unavailable"
    )
    parser.add_argument(
        "--save_sample_images",
        type=int,
        default=0,
        help="Number of sample perturbed images to save for visualization (0 = none, default: 0)"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Qwen25VLEvaluator(args.model_path, attn_implementation=args.attn_implementation)
    
    # Run evaluation
    attack_type = None if args.attack_type == 'clean' else args.attack_type
    
    evaluator.run_evaluation(
        test_data_path=args.test_data,
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        batch_size=args.batch_size,
        attack_type=attack_type,
        epsilon=args.epsilon,
        pgd_alpha=args.pgd_alpha,
        pgd_iters=args.pgd_iters,
        save_sample_images=args.save_sample_images
    )


if __name__ == "__main__":
    main()
