import torch
import time
import numpy as np
from pathlib import Path
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel


def evaluate_on_test_set(pipe, tracker, test_dataloader, experiment_name):
    """
    Run after training. Generates images from test set prompts and computes:
    - FID (generated vs real test images)
    - CLIP score
    - OCR accuracy
    - Inference latency
    """

    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)

    output_dir = Path("results/generated_images") / experiment_name / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect test samples
    all_real_images = []
    all_prompts = []
    all_texts = []

    for batch in test_dataloader:
        all_prompts.extend(batch["prompt"])
        all_texts.extend(batch["text"])
        for img_tensor in batch["image"]:
            pil = transforms.ToPILImage()(img_tensor * 0.5 + 0.5)
            all_real_images.append(pil)

    print(f"Test set: {len(all_prompts)} samples")

    # Generate images
    print("\n[1/4] Generating images from test prompts...")
    generated_images = []
    inference_times = []

    for i, prompt in enumerate(all_prompts):
        torch.cuda.empty_cache()
        start = time.time()
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
        inference_times.append(time.time() - start)

        safe_name = prompt[:40].replace(" ", "_").replace("/", "_")
        image.save(output_dir / f"{i:02d}_{safe_name}.png")
        generated_images.append(image)
        print(f"  [{i+1}/{len(all_prompts)}] done ({inference_times[-1]:.2f}s)")

    # FID
    print("\n[2/4] Computing FID...")
    transform_fid = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    real_tensors = torch.stack([transform_fid(img) for img in all_real_images]).to("cuda")
    gen_tensors = torch.stack([transform_fid(img) for img in generated_images]).to("cuda")
    fid_score = tracker.compute_fid(real_tensors, gen_tensors)
    print(f"  FID: {fid_score:.2f}  (lower is better)")

    # CLIP
    print("\n[3/4] Computing CLIP score...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_score = tracker.compute_clip_score(generated_images, all_prompts, clip_model, clip_processor)
    print(f"  CLIP score: {clip_score:.4f}  (higher is better)")
    del clip_model, clip_processor
    torch.cuda.empty_cache()

    # OCR
    print("\n[4/4] Computing OCR accuracy...")
    ocr_accuracy = tracker.compute_ocr_accuracy(generated_images, all_texts)
    if ocr_accuracy is not None:
        print(f"  OCR accuracy: {ocr_accuracy:.4f}  (fraction of images where target text is found)")
    else:
        print("  OCR accuracy: skipped")

    # Inference stats
    inference_stats = {
        "avg_latency_s": round(float(np.mean(inference_times)), 3),
        "std_latency_s": round(float(np.std(inference_times)), 3),
        "min_latency_s": round(float(np.min(inference_times)), 3),
        "max_latency_s": round(float(np.max(inference_times)), 3),
        "throughput_img_per_s": round(1.0 / np.mean(inference_times), 3)
    }

    tracker.record_test_metrics(
        fid=fid_score,
        clip_score=clip_score,
        ocr_accuracy=ocr_accuracy,
        inference_stats=inference_stats
    )

    print("\n" + "="*70)
    print("TEST EVALUATION COMPLETE")
    print("="*70)
    print(f"  FID:          {fid_score:.2f}")
    print(f"  CLIP score:   {clip_score:.4f}")
    print(f"  OCR accuracy: {ocr_accuracy}")
    print(f"  Avg latency:  {inference_stats['avg_latency_s']}s")
    print(f"  Images:       {output_dir}/")

    return {
        "fid": fid_score,
        "clip_score": clip_score,
        "ocr_accuracy": ocr_accuracy,
        "inference_stats": inference_stats
    }


def compute_val_loss(pipe, model, val_dataloader, dtype):
    """Run a no-grad forward pass over val set and return average loss."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            images = batch["image"].to("cuda", dtype=dtype)
            prompts = batch["prompt"]

            with torch.amp.autocast("cuda", dtype=dtype):
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

                prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=prompts, device="cuda", num_images_per_prompt=1,
                    max_sequence_length=512, text_encoder_out_layers=(9, 18, 27)
                )

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
                total_loss += loss.item()
                del loss, images, latents, target, velocity_pred_unpacked, target_unpacked

    model.train()
    return total_loss / len(val_dataloader)
