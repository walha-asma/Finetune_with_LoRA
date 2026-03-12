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

EXPERIMENTS = [
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
        "name": "lora_cross_attention_rank32",
        "type": "lora",
        "model_path": "models/lora_cross_attention",
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
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

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


def evaluate_all():
    test_dataloader = get_test_dataloader(batch_size=1)

    results_summary = {}

    for exp in EXPERIMENTS:
        name = exp["name"]
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        # Load existing metrics JSON so training stats are preserved
        metrics_path = Path(f"results/metrics/{name}.json")
        tracker = MetricsTracker(name)
        if metrics_path.exists():
            with open(metrics_path) as f:
                tracker.metrics = json.load(f)

        # Load model
        pipe = load_pipeline(exp)
        pipe.transformer.eval()

        # use_autocast=True only for QLoRA — fixes bfloat16/float32 VAE mismatch.
        # DO NOT wrap non-QLoRA models in autocast — it causes FID to fail.
        use_autocast = exp["type"] == "qlora"
        evaluate_on_test_set(pipe, tracker, test_dataloader, name, use_autocast=use_autocast)

        # Save merged metrics (training stats + test metrics)
        tracker.save()

        results_summary[name] = tracker.metrics.get("test", {})

        # Free VRAM before loading next model
        del pipe
        torch.cuda.empty_cache()

    # Print final comparison table
    print("\n" + "="*90)
    print("FINAL COMPARISON")
    print("="*90)
    print(f"{'Experiment':<35} {'FID':>8} {'CLIP':>8} {'OCR-Exact':>10} {'OCR-Word':>10}")
    print("-"*90)
    for exp_name, test in results_summary.items():
        fid  = test.get("fid", "N/A")
        clip = test.get("clip_score", "N/A")
        em   = test.get("ocr_exact_match", "N/A")
        wa   = test.get("ocr_word_accuracy", "N/A")
        print(f"{exp_name:<35} {str(fid):>8} {str(clip):>8} {str(em):>10} {str(wa):>10}")

    # Save summary JSON
    with open("results/metrics/final_comparison.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    print("\n✓ Saved to results/metrics/final_comparison.json")


if __name__ == "__main__":
    evaluate_all()