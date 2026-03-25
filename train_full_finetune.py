import torch
import random
import numpy as np
from diffusers import Flux2KleinPipeline
from transformers import get_cosine_schedule_with_warmup
from dataset_loader import get_train_dataloader, get_val_dataloader
from metrics_utils import MetricsTracker
from evaluation import compute_val_loss
from src.monitoring import ResourceMonitor
import json
from pathlib import Path
import gc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def full_finetune(
    model_path="models/flux2-klein-base-4b",
    output_dir="models/full_finetune",
    epochs=15,
    learning_rate=2e-6,
    batch_size=1,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    use_bf16=True,
    seed=42,
    gradient_checkpointing=True,
    early_stopping_patience=4,
):
    set_seed(seed)

    experiment_name = "full_finetune"
    print("="*60)
    print("FULL FINE-TUNING - FLUX.2-Klein-4B")
    print("="*60)

    config = {
        "experiment": experiment_name,
        "model_path": model_path,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "weight_decay": weight_decay,
        "use_bf16": use_bf16,
        "seed": seed,
        "gradient_checkpointing": gradient_checkpointing,
        "early_stopping_patience": early_stopping_patience,
        "trainable": "all_transformer",
        "frozen": "text_encoder+vae"
    }

    tracker = MetricsTracker(experiment_name)

    print("\n[1/5] Loading model...")
    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {dtype}")

    pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
    pipe.to("cuda")

    model = pipe.transformer
    total_params = sum(p.numel() for p in model.parameters())

    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing: ENABLED")

    for param in pipe.text_encoder.parameters():
        param.requires_grad = False
    for param in pipe.vae.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable:,} ({trainable/total_params*100:.2f}%)")

    print("\n[2/5] Loading dataset...")
    train_dataloader = get_train_dataloader(batch_size=batch_size)
    val_dataloader = get_val_dataloader(batch_size=batch_size)
    print(f"  Train: {len(train_dataloader.dataset)} | Val: {len(val_dataloader.dataset)}")

    print("\n[3/5] Setting up optimizer...")
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay
        )
        print("Using 8-bit AdamW")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay
        )
        print("Using standard AdamW")

    total_steps = epochs * len(train_dataloader) // gradient_accumulation_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    print("\n  Pre-computing text embeddings for all training samples...")
    pipe.text_encoder.eval()
    all_prompt_embeds = {}
    with torch.no_grad():
        for batch in train_dataloader:
            for prompt in batch["prompt"]:
                if prompt not in all_prompt_embeds:
                    pe, ti = pipe.encode_prompt(
                        prompt=prompt, device="cuda", num_images_per_prompt=1,
                        max_sequence_length=512, text_encoder_out_layers=(9, 18, 27)
                    )
                    all_prompt_embeds[prompt] = (pe.cpu(), ti.cpu())
    print(f"  Cached {len(all_prompt_embeds)} unique prompt embeddings.")

    print("\n[4/5] Training...")
    model.train()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    optimizer.zero_grad()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    tracker.start_training()

    with ResourceMonitor(sample_rate_hz=10.0) as monitor:
        for epoch in range(epochs):
            epoch_loss = 0

            for batch_idx, batch in enumerate(train_dataloader):
                torch.cuda.empty_cache()
                images = batch["image"].to("cuda", dtype=dtype)
                prompts = batch["prompt"]

                with torch.amp.autocast("cuda", dtype=dtype):
                    with torch.no_grad():
                        image_latents = pipe.vae.encode(images).latent_dist.sample()
                        image_latents = pipe._patchify_latents(image_latents)
                        latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
                        latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps)
                        latents = (image_latents - latents_bn_mean) / latents_bn_std
                        del image_latents

                    noise = torch.randn_like(latents)
                    timesteps = torch.rand(latents.shape[0], device="cuda")
                    timesteps_expanded = timesteps.view(-1, 1, 1, 1)
                    noisy_latents = (1 - timesteps_expanded) * latents + timesteps_expanded * noise
                    target = noise - latents
                    del noise, timesteps_expanded

                    prompt_embeds = all_prompt_embeds[prompts[0]][0].to("cuda")
                    text_ids      = all_prompt_embeds[prompts[0]][1].to("cuda")

                    noisy_latents_packed = pipe._pack_latents(noisy_latents)
                    latent_ids = pipe._prepare_latent_ids(noisy_latents).to("cuda")
                    del noisy_latents

                    velocity_pred = model(
                        hidden_states=noisy_latents_packed,
                        timestep=timesteps,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        return_dict=False
                    )[0]

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

            train_loss = epoch_loss / len(train_dataloader)
            val_loss = compute_val_loss(pipe, model, val_dataloader, dtype)
            tracker.record_epoch_losses(epoch + 1, train_loss, val_loss)
            if (epoch + 1) % 3 == 0:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                pipe.save_pretrained(output_dir)
                print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve}/{early_stopping_patience} epochs")
                if epochs_no_improve >= early_stopping_patience:
                    print(f"  Early stopping triggered at epoch {epoch+1}")
                    break

    resource_metrics = monitor.get_metrics()
    resource_metrics.save_csv(f"results/metrics/{experiment_name}_resources.csv")

    tracker.end_training(model, total_params)
    tracker.record_validation_metrics(best_val_loss)

    print(f"\n[5/5] Done. Best model saved to {output_dir}/")

    tracker.metrics["config"] = config
    tracker.save()
    tracker.print_summary()

    print(f"\n Full fine-tuning complete")
    return pipe


if __name__ == "__main__":
    full_finetune()