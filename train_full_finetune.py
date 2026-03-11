"""
Full fine-tuning of FLUX.2-Klein-4B model.
"""

import torch
import random
import numpy as np
from diffusers import Flux2KleinPipeline
from transformers import get_cosine_schedule_with_warmup
from dataset_loader import get_dataloader
from metrics_utils import MetricsTracker
import json
from pathlib import Path
import time
import gc

def set_seed(seed=42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def full_finetune(
    model_path="models/flux2-klein-base-4b",
    output_dir="models/full_finetune_detailed",
    epochs=3,
    learning_rate=2e-6,
    batch_size=1,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    use_bf16=True,
    seed=42,
    gradient_checkpointing=True,  # ← NOUVEAU
):
    """
    Full fine-tuning: all transformer parameters trainable.
    Text encoder frozen.
    Optimized for 30 images dataset.
    """
    
    set_seed(seed)
    
    print("="*60)
    print("FULL FINE-TUNING - FLUX.2-Klein-4B")
    print("="*60)
    
    config = {
        "experiment": "full_finetune_detailed",
        "model_path": model_path,
        "output_dir": output_dir,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "weight_decay": weight_decay,
        "use_bf16": use_bf16,
        "seed": seed,
        "gradient_checkpointing": gradient_checkpointing,
        "trainable": "all_transformer",
        "frozen": "text_encoder+vae"
    }
    
    #tracker = MetricsTracker("full_finetune")
    tracker = MetricsTracker("full_finetune_detailed")
    
    # Load model
    print("\n[1/5] Loading model...")
    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")
    
    # Charger avec Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        local_files_only=True
    )
    pipe.to("cuda")
    
    model = pipe.transformer
    total_params = sum(p.numel() for p in model.parameters())
    
    # Enable gradient checkpointing to save memory
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing: ENABLED ✓")
    
    # Freeze text encoder and VAE
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False
    
    # All transformer parameters trainable
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable: {trainable:,} ({trainable/total_params*100:.2f}%)")
    print("Text encoder (Qwen3): FROZEN ✓")
    print("VAE: FROZEN ✓")
    
    # Load dataset
    print("\n[3/5] Loading dataset...")
    #dataloader = get_dataloader(batch_size=1) # Load dataset with default path: simple prompts
    #print(f"Dataset size: {len(dataloader.dataset)} images, simple prompts")
    
    # If you want dataset with detailed prompts
    dataloader = get_dataloader("data/dataset_detailed.json",batch_size=1) # Load dataset with detailed prompts
    print(f"Dataset size: {len(dataloader.dataset)} images, detailed prompts")
    
    # Optimizer & scheduler - use 8bit optimizer to save memory
    print("\n[3/5] Setting up optimizer...")
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
        print("Using 8-bit AdamW optimizer ✓")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
        print("Using standard AdamW optimizer")
    
    total_steps = epochs * len(dataloader) // gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print(f"Learning rate: {learning_rate}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    # Training
    print("\n[4/5] Training...")
    tracker.start_training()
    
    model.train()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    optimizer.zero_grad()
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Clear cache before each batch
            torch.cuda.empty_cache()
            
            images = batch['image'].to("cuda", dtype=dtype)
            prompts = batch['prompt']
            
            with torch.amp.autocast('cuda', dtype=dtype):
                # Encode images to latents
                with torch.no_grad():
                    image_latents = pipe.vae.encode(images).latent_dist.sample()
                    # Apply patchify and normalization
                    image_latents = pipe._patchify_latents(image_latents)
                    latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
                    latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps)
                    latents = (image_latents - latents_bn_mean) / latents_bn_std
                    
                    # Free VAE outputs
                    del image_latents
                
                # Flow matching: add noise using interpolation
                noise = torch.randn_like(latents)
                timesteps = torch.rand(latents.shape[0], device="cuda")
                timesteps_expanded = timesteps.view(-1, 1, 1, 1)
                
                # Flow matching interpolation
                noisy_latents = (1 - timesteps_expanded) * latents + timesteps_expanded * noise
                target = noise - latents
                
                # Free intermediate tensors
                del noise, timesteps_expanded
                
                # Encode prompts using pipeline method
                max_length = 512
                with torch.no_grad():
                    prompt_embeds, text_ids = pipe.encode_prompt(
                        prompt=prompts,
                        device="cuda",
                        num_images_per_prompt=1,
                        max_sequence_length=max_length,
                        text_encoder_out_layers=(9, 18, 27)
                    )
                
                # Pack latents
                noisy_latents_packed = pipe._pack_latents(noisy_latents)
                latent_ids = pipe._prepare_latent_ids(noisy_latents).to("cuda")
                
                # Free unpacked latents
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
                
                # Unpack for loss computation
                velocity_pred_unpacked = pipe._unpack_latents_with_ids(velocity_pred, latent_ids)
                target_packed = pipe._pack_latents(target)
                target_unpacked = pipe._unpack_latents_with_ids(target_packed, latent_ids)
                
                # Free packed tensors
                del velocity_pred, target_packed, noisy_latents_packed
                
                loss = torch.nn.functional.mse_loss(velocity_pred_unpacked, target_unpacked)
                loss = loss / gradient_accumulation_steps
                
                # Free prediction tensors
                del velocity_pred_unpacked, target_unpacked
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Clear cache after optimizer step
                torch.cuda.empty_cache()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            if (batch_idx + 1) % 5 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"  Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item() * gradient_accumulation_steps:.4f} - GPU: {allocated:.2f}GB / {reserved:.2f}GB")
            
            # Free loss
            del loss, images, latents, target
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
    tracker.end_training(model, total_params)
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(output_dir)
    
    # Generate test images
    print("\n[5/5] Generating test images...")
    model.eval()
    
    #with open("data/test_prompts.json") as f:
    #    test_prompts = json.load(f)
        
    with open("data/test_prompts_detailed.json") as f:
        test_prompts = json.load(f)
    
    #image_output_dir = Path("results/generated_images/full_finetune")
    image_output_dir = Path("results/generated_images/full_finetune_detailed")
    image_output_dir.mkdir(parents=True, exist_ok=True)
    
    inference_times = []
    generated_files = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"  [{i+1}/{len(test_prompts)}] {prompt}")
        
        # Clear cache before inference
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
        
        safe_name = prompt[:20].replace(' ', '_').replace('/', '_')
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
        "experiment": "full_finetune_detailed ",
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
     
    #(image_output_dir / "generation_metadata.json").write_text(json.dumps(generation_metadata, indent=2))
    #(image_output_dir / "config.json").write_text(json.dumps(config, indent=2))
    
    (image_output_dir / "generation_metadata_detailed.json").write_text(json.dumps(generation_metadata, indent=2))
    (image_output_dir / "config_detailed.json").write_text(json.dumps(config, indent=2))
    
    tracker.record_quality_metrics(fid_score=None, clip_score=None, val_loss=avg_loss)
    tracker.metrics["inference"] = generation_metadata["inference_stats"]
    tracker.metrics["config"] = config
    tracker.save()
    tracker.print_summary()
    
    print("\n✓ Full fine-tuning complete")
    print(f"✓ Model: {output_dir}/")
    print(f"✓ Images: {image_output_dir}/")
    #print(f"✓ Metrics: results/metrics/full_finetune.json")
    print(f"✓ Metrics: results/metrics/full_finetune_detailed.json")
    
    return pipe

if __name__ == "__main__":
    full_finetune()