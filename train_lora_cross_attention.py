"""
LoRA on cross-attention layers only for FLUX.2-Klein.
Uses rank 16 from previous sweep.
"""

import torch
import random
import numpy as np
from diffusers import Flux2KleinPipeline
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup
from dataset_loader import get_dataloader
from metrics_utils import MetricsTracker
import json
from pathlib import Path
import time

def set_seed(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_lora_cross_attention(
    model_path="models/flux2-klein-base-4b",
    output_dir="models/lora_cross_attention_10epochs",
    rank=16,  # ← Votre meilleur rank
    epochs=15,
    seed=42
):
    """
    LoRA fine-tuning on cross-attention layers only.
    Cross-attention = interaction between image and text
    Optimized for 30 images.
    """
    
    set_seed(seed)
    
    experiment_name = f"lora_cross_attention_rank{rank}_10epochs"
    print("="*60)
    print(f"LORA CROSS-ATTENTION ONLY - RANK {rank}")
    print("="*60)
    
    # Configuration
    config = {
        "experiment": experiment_name,
        "model_path": model_path,
        "output_dir": output_dir,
        "rank": rank,
        "lora_alpha": rank * 2,
        "epochs": epochs,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "weight_decay": 0.01,
        "lora_dropout": 0.1,
        # Cross-attention only: K and V projections
        "target_modules": ["to_k", "to_v"],
        "seed": seed,
        "trainable": "lora_cross_attention_only",
        "frozen": "base_model+self_attention+text_encoder+vae"
    }
    
    tracker = MetricsTracker(experiment_name)
    
    # Load model
    print("\n[1/5] Loading model...")
    dtype = torch.bfloat16
    
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True
    )
    pipe.to("cuda")
    
    model = pipe.transformer
    total_params = sum(p.numel() for p in model.parameters())
    
    # Freeze text encoder and VAE
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False
    
    # Configure LoRA on cross-attention only
    print(f"\n[2/5] Configuring LoRA on cross-attention (rank={rank})...")
    print("Target: K and V projections (text-image interaction)")
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["to_k", "to_v"],  # Cross-attention only
        lora_dropout=0.1,
        bias="none"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable (LoRA cross-attn): {trainable_params:,} ({trainable_params/total_params*100:.3f}%)")
    
    # Load dataset
    print("\n[3/5] Loading dataset...")
    dataloader = get_dataloader(batch_size=1) # Load dataset with default path: simple prompts
    print(f"Dataset size: {len(dataloader.dataset)} images, simple prompts")
    
    # If you want dataset with detailed prompts
    #dataloader = get_dataloader("data/dataset_detailed.json",batch_size=1) # Load dataset with detailed prompts
    #print(f"Dataset size: {len(dataloader.dataset)} images, detailed prompts")
    
    # Optimizer
    print("\n[4/5] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    total_steps = epochs * len(dataloader) // 4
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print(f"Learning rate: 1e-4")
    print(f"Total epochs: {epochs}")
    
    # Training
    print("\n[5/5] Training...")
    tracker.start_training()
    
    model.train()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    optimizer.zero_grad()
    
    gradient_accumulation_steps = 4
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to("cuda", dtype=dtype)
            prompts = batch['prompt']
            
            with torch.amp.autocast('cuda', dtype=dtype):
                # Encode images
                with torch.no_grad():
                    image_latents = pipe.vae.encode(images).latent_dist.sample()
                    image_latents = pipe._patchify_latents(image_latents)
                    latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
                    latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps)
                    latents = (image_latents - latents_bn_mean) / latents_bn_std
                    del image_latents
                
                # Flow matching
                noise = torch.randn_like(latents)
                timesteps = torch.rand(latents.shape[0], device="cuda")
                timesteps_expanded = timesteps.view(-1, 1, 1, 1)
                noisy_latents = (1 - timesteps_expanded) * latents + timesteps_expanded * noise
                target = noise - latents
                
                del noise, timesteps_expanded
                
                # Encode prompts
                with torch.no_grad():
                    prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=prompts,
                        device="cuda",
                        num_images_per_prompt=1,
                        max_sequence_length=512,
                        text_encoder_out_layers=(9, 18, 27)
                    )
                
                # Pack latents
                noisy_latents_packed = pipe._pack_latents(noisy_latents)
                latent_ids = pipe._prepare_latent_ids(noisy_latents).to("cuda")
                del noisy_latents
                
                # Predict velocity
                velocity_pred = model(
                    hidden_states=noisy_latents_packed,
                    timestep=timesteps,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False
                )[0]
                
                # Compute loss
                velocity_pred_unpacked = pipe._unpack_latents_with_ids(velocity_pred, latent_ids)
                target_packed = pipe._pack_latents(target)
                target_unpacked = pipe._unpack_latents_with_ids(target_packed, latent_ids)
                
                del velocity_pred, target_packed, noisy_latents_packed
                
                loss = torch.nn.functional.mse_loss(velocity_pred_unpacked, target_unpacked)
                loss = loss / gradient_accumulation_steps
                
                del velocity_pred_unpacked, target_unpacked
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            del loss, images, latents, target
        
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    tracker.end_training(model, total_params)
    
    # Save LoRA adapters
    print(f"\nSaving LoRA adapters to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    
    with open(Path(output_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Generate test images
    print("\nGenerating test images...")
    model.eval()
    
    with open("data/test_prompts.json") as f:
        test_prompts = json.load(f)
    
    image_output_dir = Path(f"results/generated_images/{experiment_name}")
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    inference_times = []
    generated_files = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"  [{i+1}/{len(test_prompts)}] {prompt}")
        
        torch.cuda.empty_cache()
        
        start_time = time.time()
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=4.0,
                height=512,
                width=512,
                max_sequence_length=512,
                text_encoder_out_layers=(9, 18, 27)
            ).images[0]
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        safe_name = prompt[:50].replace(' ', '_').replace('/', '_')
        filename = f"{i:02d}_{safe_name}.png"
        filepath = image_output_dir / filename
        image.save(filepath)
        
        generated_files.append({
            "prompt": prompt,
            "filename": filename,
            "filepath": str(filepath),
            "inference_time": inference_time
        })
    
    print(f"\n✓ Images saved to: {image_output_dir}/")
    
    # Save metadata
    generation_metadata = {
        "experiment": experiment_name,
        "test_prompts": test_prompts,
        "generated_files": generated_files,
        "inference_stats": {
            "avg_latency_s": float(np.mean(inference_times)),
            "std_latency_s": float(np.std(inference_times)),
            "min_latency_s": float(np.min(inference_times)),
            "max_latency_s": float(np.max(inference_times)),
            "throughput_img_per_s": 1.0 / np.mean(inference_times)
        },
        "output_directory": str(image_output_dir)
    }
    
    (image_output_dir / "generation_metadata.json").write_text(json.dumps(generation_metadata, indent=2))
    (image_output_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    # Save metrics
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    tracker.record_quality_metrics(fid_score=None, clip_score=None, val_loss=avg_loss)
    tracker.metrics["inference"] = generation_metadata["inference_stats"]
    tracker.metrics["config"] = config
    tracker.save()
    tracker.print_summary()
    
    print("\n✓ LoRA cross-attention training complete")
    print(f"✓ Adapters: {output_dir}/")
    print(f"✓ Images: {image_output_dir}/")
    print(f"\nNext: Run 'python scripts/evaluate_all.py' after all experiments")
    
    return pipe

if __name__ == "__main__":
    train_lora_cross_attention(rank=16, epochs=10)