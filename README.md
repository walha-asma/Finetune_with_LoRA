# Efficient Text Rendering via LoRA Fine-Tuning of Flux.2 Klein 4B

Fine-tuning strategies for text-in-image generation on the Flux.2 Klein 4B model.
Covers full fine-tuning, LoRA rank sweep, cross-attention LoRA, and QLoRA —
with a custom 500-sample dataset built from AnyWord-3M.

Part of the **Efficient Image Generation with Text Rendering** project (ENSTA Paris, 2026).

---

## Project structure

```
Finetune_with_LoRA/
├── train_full_finetune.py        # Full fine-tuning (all transformer params)
├── train_lora_rank_sweep.py      # LoRA rank sweep: r ∈ {8, 16, 32, 64}
├── train_lora_cross_attention.py # CA-LoRA: to_k + to_v only, r ∈ {16, 32}
├── train_qlora.py                # QLoRA: 4-bit NF4 backbone + LoRA adapters
├── evaluate_all.py               # Full evaluation pipeline (FID, CLIP, OCR)
├── analyze_ocr_by_length.py      # OCR breakdown by target text length
├── generate_plots.py             # All result plots
├── dataset_loader.py             # Dataset loading + preprocessing
├── download_model.py             # Download Flux.2 Klein 4B from HuggingFace
├── metrics_utils.py              # MetricsTracker: FID, CLIP, OCR, energy
├── evaluation.py                 # evaluate_on_test_set + compute_val_loss
├── requirements.txt
├── src/
│   ├── evaluation/
│   │   └── fid.py                # InceptionV3-based FID implementation
│   └── monitoring/
│       ├── resource_monitor.py   # Background thread resource monitor
│       └── metrics.py            # ResourceMetrics + CSV export
├── jobs/
│   ├── sh_files/                 # SLURM job scripts
│   └── logs/                     # Job stdout/stderr logs
data/
├── train.json                    # 400 training samples
├── val.json                      # 50 validation samples
├── test.json                     # 50 test samples
└── test/images/                  # Test images
models/
├── flux2-klein-base-4b/          # Base model (downloaded)
├── full_finetune/                # Full fine-tune checkpoint
├── lora_flux2klein/
│   ├── rank_8/                   # LoRA r=8 adapter
│   ├── rank_16/
│   ├── rank_32/
│   └── rank_64/
├── lora_cross_attention_rank16/  # CA-LoRA adapters
├── lora_cross_attention_rank32/
└── qlora_cross_attention/        # QLoRA adapter
results/
├── metrics/                      # Per-experiment JSON files + final_comparison.json
├── plots/                        # All generated figures
├── generated_images/             # Generated test images per experiment
└── fid_reference/                # FID reference images from full fine-tune
```

---

## Installation

> **Important:** Flux.2 Klein requires the development version of diffusers.
> The released PyPI version does not include `Flux2KleinPipeline`.

```bash
# 1. Clone and enter the project
git clone <repo_url>
cd text-in-image-generation/Finetune_with_LoRA

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install diffusers from source (required for Flux.2 Klein support)
pip install -U git+https://github.com/huggingface/diffusers.git

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Quickstart

### 1. Download the base model

```bash
export HF_TOKEN="your_huggingface_token"
python download_model.py
```

The model requires accepting the license on [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) before downloading.
The model is saved to `models/flux2-klein-base-4b/` (~8 GB).

### 2. Prepare the dataset

Place your dataset under `data/` with the following structure:

```
data/
├── train.json
├── val.json
├── test.json
└── test/images/   (and train/, val/ for training)
```

Each JSON file contains records with fields: `filepath`, `prompt`, `text`.

### 3. Train

```bash
# Full fine-tuning (5 epochs, ~30 min on L40S)
python train_full_finetune.py

# LoRA rank sweep (r=8,16,32,64 — 10 epochs each)
python train_lora_rank_sweep.py

# Cross-attention LoRA (r=16 and r=32 — 15 epochs each)
python train_lora_cross_attention.py

# QLoRA (4-bit NF4 + LoRA, 15 epochs)
python train_qlora.py
```

All scripts save the **best checkpoint** (lowest validation loss) automatically.
Metrics, loss curves, and resource usage are saved to `results/metrics/`.

### 4. Evaluate

```bash
# Run full evaluation pipeline (FID, CLIP, OCR) for all experiments
python evaluate_all.py
```

This builds the FID reference from the full fine-tune model first, then evaluates
all experiments against the same reference distribution for a fair comparison.
Generated images are saved to `results/generated_images/`.

### 5. Analyze OCR by text length

```bash
# Breakdown of OCR scores by word count (1-word / 2-word / 3+-word targets)
# Runs on already-generated images — no GPU required
python analyze_ocr_by_length.py
```

### 6. Generate plots

```bash
python generate_plots.py
```

Produces: `tradeoff_scatter.png`, `radar_chart.png`, `adapter_sizes.png`,
`loss_curves.png`, `ocr_comparison.png`, `ocr_by_length_heatmap.png`.

---

## Running on a SLURM cluster

SLURM job scripts are in `jobs/sh_files/`. Example submission:

```bash
sbatch jobs/sh_files/train_lora_sweep.sh
```

Logs are written to `jobs/logs/`.

---

## Key results

| Experiment | FID ↓ | OCR-Exact ↑ | OCR-Word ↑ | VRAM (GB) | Size (MB) |
|---|---|---|---|---|---|
| Original baseline | 228.91 | 0.16 | 0.413 | — | 22617 |
| Full fine-tune | 172.79 | 0.16 | 0.398 | 30.15 | 30449 |
| LoRA r=8 | 161.61 | **0.24** | 0.403 | 19.11 | 3.76 |
| LoRA r=16 | 167.03 | 0.22 | 0.435 | 19.13 | 7.51 |
| LoRA r=32 | 159.67 | 0.20 | 0.411 | 19.16 | 15.01 |
| LoRA r=64 | 159.17 | 0.18 | 0.380 | 19.23 | 30.01 |
| CA-LoRA r=16 | 161.51 | 0.20 | 0.412 | 19.02 | 3.75 |
| CA-LoRA r=32 | **152.48** | 0.20 | **0.453** | 19.04 | 7.50 |
| QLoRA r=16 | 164.22 | 0.18 | 0.350 | **16.56** | 3.75 |

FID is computed against 200 full fine-tune reference images.
OCR metrics use normalized string comparison (punctuation stripped, lowercased).

**Key findings:**
- CA-LoRA r=32 is the Pareto optimum: best FID and word accuracy with only 7.5 MB
- LoRA r=8 achieves the best exact match — higher ranks overfit to scene texture
- QLoRA requires the least VRAM (16.56 GB) with competitive quality
- OCR exact match collapses near zero for 3+ word targets across all models

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| diffusers (git) | latest | Flux.2 Klein pipeline |
| torch | ≥ 2.0 | Training |
| peft | ≥ 0.8 | LoRA / QLoRA |
| bitsandbytes | ≥ 0.41 | 4-bit quantization + 8-bit AdamW |
| transformers | ≥ 4.38 | Text encoder, scheduler |
| easyocr | latest | OCR evaluation |
| codecarbon | ≥ 2.3 | Energy tracking |
| scipy | latest | FID computation |

---

## Notes

- All training uses `bfloat16` mixed precision with gradient accumulation of 4 steps
- Text encoder and VAE are frozen throughout — only the transformer backbone is adapted
- Best model is saved based on validation loss (flow-matching MSE in latent space)
- Inference uses a fixed seed per sample (`seed = 2024 + i`) across all experiments
  for a fair comparison
