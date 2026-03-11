"""
Evaluation utilities for fine-tuning experiments.
Adapted for case where test prompts have no ground truth images.
"""

import torch
import json
from pathlib import Path
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

def evaluate_model(pipe, tracker, dataloader, avg_loss, experiment_name):
    """
    Evaluate fine-tuned model.
    
    Strategy:
    1. VISUAL TEST: Generate images from test prompts (qualitative check)
    2. FID: Compare generated vs real training images (quantitative)
    3. CLIP Score: Text-image alignment on generated images
    """
    
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Load test prompts
    with open("data/test_prompts.json") as f:
        test_prompts = json.load(f)
    
    print("\nEvaluation strategy:")
    print("  1. Visual test: Generate from test prompts (for visual inspection)")
    print("  2. FID: Generated training prompts vs real training images")
    print("  3. CLIP Score: Text-image alignment on generated images")
    print()
    
    # ========================================================================
    # 1. VISUAL TEST: Generate images from test prompts
    # ========================================================================
    
    print("[1/4] Visual Test - Generating images from test prompts...")
    output_dir = Path("results/generated_images") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_test_images = []
    for i, prompt in enumerate(test_prompts[:5]):
        print(f"  Generating: '{prompt}'")
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
            
            # Save for visual inspection
            safe_name = prompt[:40].replace(' ', '_').replace('/', '_')
            image.save(output_dir / f"test_{i:02d}_{safe_name}.png")
            generated_test_images.append(image)
    
    print(f"\n✓ Test images saved to: {output_dir}/")
    print("  Review these images to check if the model learned the concepts")
    
    # ========================================================================
    # 2. FID: Use training images as ground truth
    # ========================================================================
    
    print("\n[2/4] Computing FID (training images as reference)...")
    
    # Get all training images
    all_train_images = []
    all_train_prompts = []
    for batch in dataloader:
        all_train_images.extend(batch['image'])
        all_train_prompts.extend(batch['prompt'])
    
    # Sample images for FID
    import random
    random.seed(42)
    num_samples = min(10, len(all_train_images))
    indices = random.sample(range(len(all_train_images)), num_samples)
    
    reference_images = [all_train_images[i] for i in indices]
    reference_prompts = [all_train_prompts[i] for i in indices]
    
    # Generate images from same prompts
    print(f"  Generating {num_samples} images from training prompts...")
    generated_train_images = []
    for prompt in reference_prompts:
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
            generated_train_images.append(image)
    
    # Prepare for FID computation
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Convert to tensors
    real_tensors = torch.stack([
        transform(transforms.ToPILImage()(img))
        for img in reference_images
    ]).to("cuda")
    
    gen_tensors = torch.stack([
        transform(img)
        for img in generated_train_images
    ]).to("cuda")
    
    # Compute FID
    fid_score = tracker.compute_fid(real_tensors, gen_tensors)
    print(f"  FID: {fid_score:.2f}")
    print("  (Lower is better. Compares generated vs real training images)")
    
    # ========================================================================
    # 3. CLIP SCORE: Text-image alignment
    # ========================================================================
    
    print("\n[3/4] Computing CLIP Score (text-image alignment)...")
    
    # Load CLIP
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Compute on test images (do they match their prompts?)
    clip_score_test = tracker.compute_clip_score(
        generated_test_images,
        test_prompts[:5],
        clip_model,
        clip_processor
    )
    
    # Also compute on training images (consistency check)
    clip_score_train = tracker.compute_clip_score(
        generated_train_images,
        reference_prompts,
        clip_model,
        clip_processor
    )
    
    # Use average
    clip_score = (clip_score_test + clip_score_train) / 2
    
    print(f"  CLIP Score (test prompts): {clip_score_test:.4f}")
    print(f"  CLIP Score (train prompts): {clip_score_train:.4f}")
    print(f"  CLIP Score (average): {clip_score:.4f}")
    print("  (Higher is better. Measures text-image alignment)")
    
    # ========================================================================
    # 4. INFERENCE METRICS
    # ========================================================================
    
    print("\n[4/4] Measuring inference performance...")
    tracker.measure_inference(pipe, test_prompts)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    tracker.record_quality_metrics(
        fid_score=fid_score,
        clip_score=clip_score,
        val_loss=avg_loss
    )
    
    tracker.save()
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nVisual results: {output_dir}/")
    print(f"Metrics: results/metrics/{experiment_name}.json")
    
    return {
        'fid': fid_score,
        'clip_score': clip_score,
        'clip_score_test': clip_score_test,
        'clip_score_train': clip_score_train,
        'visual_output_dir': str(output_dir)
    }


def quick_visual_eval(pipe, test_prompts, output_dir, experiment_name):
    """
    Quick visual evaluation - just generate images.
    Use this for rapid iteration without full metrics.
    """
    
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nQuick visual eval - generating {len(test_prompts)} images...")
    
    for i, prompt in enumerate(test_prompts):
        print(f"  [{i+1}/{len(test_prompts)}] {prompt}")
        
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
        
        safe_name = prompt[:40].replace(' ', '_').replace('/', '_')
        image.save(output_path / f"{i:02d}_{safe_name}.png")
    
    print(f"\n✓ Images saved to: {output_path}/")
    print("  Open the images to visually check quality!")
    
    return str(output_path)