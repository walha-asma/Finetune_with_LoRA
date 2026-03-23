import torch
import json
from pathlib import Path
from diffusers import Flux2KleinPipeline
from peft import PeftModel
from dataset_loader import get_test_dataloader
from evaluation import evaluate_on_test_set
from metrics_utils import MetricsTracker

MODEL_BASE = "models/flux2-klein-base-4b"
dtype = torch.bfloat16

FID_REFERENCE_DIR = Path("results/fid_reference/full_finetune")

EXPERIMENTS = [
    {
        "name": "original_baseline",
        "type": "full",
        "model_path": MODEL_BASE,       # No fine-tuning — starting point
    },
    {
        "name": "full_finetune",
        "type": "full",
        "model_path": "models/full_finetune",
    },
    {
        "name": "lora_flux2klein_rank8",
        "type": "lora",
        "model_path": "models/lora_flux2klein/rank_8",
    },
    {
        "name": "lora_flux2klein_rank16",
        "type": "lora",
        "model_path": "models/lora_flux2klein/rank_16",
    },
    {
        "name": "lora_flux2klein_rank32",
        "type": "lora",
        "model_path": "models/lora_flux2klein/rank_32",
    },
    {
        "name": "lora_flux2klein_rank64",
        "type": "lora",
        "model_path": "models/lora_flux2klein/rank_64",
    },
    {
        "name": "lora_cross_attention_rank16",
        "type": "lora",
        "model_path": "models/lora_cross_attention/rank_16",
    },
    {
        "name": "lora_cross_attention_rank32",
        "type": "lora",
        "model_path": "models/lora_cross_attention/rank_32",
    },
    {
        "name": "qlora_cross_attention_rank16",
        "type": "qlora",
        "model_path": "models/qlora_cross_attention",
    },
]


def load_pipeline(experiment):
    exp_type = experiment["type"]
    path = experiment["model_path"]

    if exp_type == "full":
        pipe = Flux2KleinPipeline.from_pretrained(
            path, torch_dtype=dtype, local_files_only=True
        )

    elif exp_type == "lora":
        pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_BASE, torch_dtype=dtype, local_files_only=True
        )
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, path)

    elif exp_type == "qlora":
        from diffusers.quantizers import PipelineQuantizationConfig
        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
            },
            components_to_quantize=["transformer"]
        )
        pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_BASE,
            quantization_config=quant_config,
            torch_dtype=dtype,
            local_files_only=True
        )
        pipe.vae = pipe.vae.to(dtype=torch.bfloat16)
        pipe.text_encoder = pipe.text_encoder.to(dtype=torch.bfloat16)
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, path)

    pipe.to("cuda")
    return pipe


def get_adapter_size_mb(experiment):
    """Return adapter .safetensors size in MB, or total model size for full models."""
    path = Path(experiment["model_path"])
    safetensors_files = list(path.rglob("*.safetensors"))
    if not safetensors_files:
        return None
    total_bytes = sum(f.stat().st_size for f in safetensors_files)
    return round(total_bytes / (1024 ** 2), 2)


# 100 prompts × 3 = 300 reference images — reliable covariance estimate.
# Seeds: (2024 + k*1000) + i, identical to evaluate_on_test_set.
FID_IMAGES_PER_PROMPT = 3


def build_fid_reference(pipe, test_dataloader, reference_dir,
                        images_per_prompt=FID_IMAGES_PER_PROMPT):
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)

    all_prompts = []
    for batch in test_dataloader:
        all_prompts.extend(batch["prompt"])

    expected_count = len(all_prompts) * images_per_prompt
    existing = list(reference_dir.glob("*.png"))
    if len(existing) >= expected_count:
        print(f"  FID reference already exists ({len(existing)} images), skipping generation.")
        from PIL import Image
        return [Image.open(p).convert("RGB") for p in sorted(existing)]

    for f in existing:
        f.unlink()

    print(f"  Generating FID reference ({images_per_prompt} images/prompt x {len(all_prompts)} = {expected_count} total)...")
    reference_images = []
    pipe.transformer.eval()
    idx = 0

    for i, prompt in enumerate(all_prompts):
        for k in range(images_per_prompt):
            seed = (2024 + k * 1000) + i
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
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
            image.save(reference_dir / f"{idx:04d}.png")
            reference_images.append(image)
            idx += 1
        print(f"    [{i+1}/{len(all_prompts)}] {images_per_prompt} reference images generated")

    print(f"  FID reference saved to {reference_dir} ({len(reference_images)} images)")
    return reference_images


def evaluate_all():
    test_dataloader = get_test_dataloader(batch_size=1)
    results_summary = {}

    # === Step 1: build FID reference from full_finetune FIRST ===
    # All experiments including original_baseline use this same reference
    # so every FID score is on the same scale and directly comparable.
    print("\n" + "="*60)
    print("BUILDING FID REFERENCE (full_finetune)")
    print("="*60)
    full_ft_exp = next(e for e in EXPERIMENTS if e["name"] == "full_finetune")
    pipe_ft = load_pipeline(full_ft_exp)
    pipe_ft.transformer.eval()
    fid_reference_images = build_fid_reference(pipe_ft, test_dataloader, FID_REFERENCE_DIR)
    del pipe_ft
    torch.cuda.empty_cache()

    # === Step 2: evaluate every experiment with the same FID reference ===
    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        metrics_path = Path(f"results/metrics/{name}.json")
        tracker = MetricsTracker(name)
        if metrics_path.exists():
            with open(metrics_path) as f:
                tracker.metrics = json.load(f)

        # Adapter / model storage size
        adapter_size = get_adapter_size_mb(exp)
        if adapter_size is not None:
            tracker.metrics.setdefault("training", {})
            tracker.metrics["training"]["adapter_size_mb"] = adapter_size
            print(f"  Adapter size: {adapter_size} MB")

        pipe = load_pipeline(exp)
        pipe.transformer.eval()

        use_autocast = exp["type"] == "qlora"
        evaluate_on_test_set(
            pipe, tracker, test_dataloader, name,
            use_autocast=use_autocast,
            fid_reference_images=fid_reference_images  # same reference for ALL experiments
        )

        tracker.save()
        results_summary[name] = tracker.metrics.get("test", {})

        del pipe
        torch.cuda.empty_cache()

    # Final comparison table
    print("\n" + "="*110)
    print("FINAL COMPARISON")
    print("="*110)
    print(f"{'Experiment':<38} {'FID':>8} {'CLIP':>8} {'OCR-Exact':>10} {'OCR-Word':>10} {'OCR-CER':>9} {'Size(MB)':>10}")
    print("-"*110)
    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        test = results_summary.get(exp_name, {})
        metrics_path = Path(f"results/metrics/{exp_name}.json")
        size = "N/A"
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            size = m.get("training", {}).get("adapter_size_mb", "N/A")
        fid  = test.get("fid", "N/A")
        clip = test.get("clip_score", "N/A")
        em   = test.get("ocr_exact_match", "N/A")
        wa   = test.get("ocr_word_accuracy", "N/A")
        cer  = test.get("ocr_cer", "N/A")
        print(f"{exp_name:<38} {str(fid):>8} {str(clip):>8} {str(em):>10} {str(wa):>10} {str(cer):>9} {str(size):>10}")

    with open("results/metrics/final_comparison.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\n✓ Saved to results/metrics/final_comparison.json")


if __name__ == "__main__":
    evaluate_all()