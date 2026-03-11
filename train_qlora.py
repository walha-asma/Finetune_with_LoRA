"""
QLoRA fine-tuning with 4-bit quantization on cross-attention only.
Optimized for FLUX.2-Klein with 30 images.
"""

import torch
import random
import numpy as np
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
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

def train_qlora(
    model_path="models/flux2-klein-base-4b",
    output_dir="models/qlora_cross_attention",
    rank=16,
    epochs=15,
    seed=42
):
    """
    QLoRA fine-tuning with 4-bit quantization on cross-attention only.
    Uses PipelineQuantizationConfig for proper quantization.
    """
    
    set_seed(seed)
    
    experiment_name = f"qlora_cross_attention_rank{rank}"
    print("="*60)
    print(f"QLORA CROSS-ATTENTION ONLY - RANK {rank}")
    print("="*60)
    
    # Cross-attention only
    target_modules = ["to_k", "to_v"]
    
    config = {
        "experiment": experiment_name,
        "model_path": model_path,
        "output_dir": output_dir,
        "rank": rank,
        "lora_alpha": rank * 2,
        "target_modules": target_modules,
        "epochs": epochs,
        "learning_rate": 2e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "weight_decay": 0.01,
        "lora_dropout": 0.1,
        "quantization": "4bit_nf4",
        "seed": seed,
        "trainable": "qlora_cross_attention_only",
        "frozen": "quantized_base+text_encoder+vae"
    }
    
    tracker = MetricsTracker(experiment_name)
    
    # Use PipelineQuantizationConfig (proper way for diffusers)
    print("\n[1/5] Configuring 4-bit quantization...")
    from diffusers import BitsAndBytesConfig
    
    quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True
        },
        components_to_quantize=["transformer"]  # Only quantize transformer
    )
    
    # Load pipeline with quantization
    print("\n[2/5] Loading quantized FLUX.2-Klein pipeline...")
    dtype = torch.bfloat16
    
    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=dtype,
        local_files_only=True
    )
    pipe.to("cuda")
    
    model = pipe.transformer
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ Transformer loaded with 4-bit quantization")
    print(f"✓ Text encoder and VAE loaded in {dtype}")
    
    # Freeze text encoder and VAE
    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False
    
    # Enable gradient checkpointing for quantized model
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Configure QLoRA on cross-attention
    print(f"\n[3/5] Configuring QLoRA on cross-attention (rank={rank})...")
    print("Target: K and V projections (text-image interaction)")
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable (QLoRA): {trainable_params:,} ({trainable_params/total_params*100:.3f}%)")
    
    # Load dataset
    print("\n[3/5] Loading dataset...")
    dataloader = get_dataloader(batch_size=1) # Load dataset with default path: simple prompts
    print(f"Dataset size: {len(dataloader.dataset)} images, simple prompts")
    
    # If you want dataset with detailed prompts
    #dataloader = get_dataloader("data/dataset_detailed.json",batch_size=1) # Load dataset with detailed prompts
    #print(f"Dataset size: {len(dataloader.dataset)} images, detailed prompts")
    
    # 8-bit optimizer
    print("\n[5/5] Setting up 8-bit optimizer...")
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=2e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    total_steps = epochs * len(dataloader) // 4
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print(f"Learning rate: 2e-4")
    print(f"Total epochs: {epochs}")
    
    # Training
    print("\nTraining...")
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
            
            # Monitor VRAM
            if (batch_idx + 1) % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"  Epoch {epoch+1}/{epochs} - Batch {batch_idx+1} - GPU: {allocated:.2f}GB")
            
            del loss, images, latents, target
        
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    tracker.end_training(model, total_params)
    
    # Save adapters
    print(f"\nSaving QLoRA adapters to {output_dir}...")
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
            # Encode prompt
            prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=[prompt],
                device="cuda",
                num_images_per_prompt=1,
                max_sequence_length=512,
                text_encoder_out_layers=(9, 18, 27)
            )
            
            # Generate latents
            latents = torch.randn(
                (1, 64, 64, 64),
                device="cuda",
                dtype=torch.bfloat16  # ← Important: utiliser bfloat16
            )
            
            # Pack for transformer
            latents_packed = pipe._pack_latents(latents)
            latent_ids = pipe._prepare_latent_ids(latents).to("cuda")
            
            # Denoising loop
            for t in pipe.scheduler.timesteps:
                # Predict
                velocity_pred = model(
                    hidden_states=latents_packed,
                    timestep=torch.tensor([t / 1000], device="cuda"),
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False
                )[0]
                
                # Unpack
                velocity_pred_unpacked = pipe._unpack_latents_with_ids(velocity_pred, latent_ids)
                
                # Update (simple Euler step for flow matching)
                dt = 1.0 / len(pipe.scheduler.timesteps)
                latents = latents - dt * velocity_pred_unpacked
                
                # Repack
                latents_packed = pipe._pack_latents(latents)
            
            # Unpatchify
            latents_unpatchified = pipe._unpatchify_latents(latents)
            
            # Denormalize
            latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents_unpatchified.device, latents_unpatchified.dtype)
            latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps)
            latents_denorm = latents_unpatchified * latents_bn_std + latents_bn_mean
            
            # ← IMPORTANT: Convertir en bfloat16 avant décodage VAE
            latents_denorm = latents_denorm.to(dtype=torch.bfloat16)
            
            # Decode
            image = pipe.vae.decode(latents_denorm, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        
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
    
    print("\n✓ QLoRA cross-attention training complete")
    print(f"✓ Adapters: {output_dir}/")
    print(f"✓ Images: {image_output_dir}/")
    print(f"\nExpected VRAM: ~12-15 GB (vs ~19 GB for standard LoRA)")
    
    return pipe

if __name__ == "__main__":
    train_qlora(rank=16, epochs=15)