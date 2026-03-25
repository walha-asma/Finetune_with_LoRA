# Efficient Text Rendering via LoRA Fine-Tuning of Flux.2 Klein 4B

Fine-tuning strategies for text-in-image generation on the Flux.2 Klein 4B model.
Covers full fine-tuning, LoRA rank sweep, cross-attention LoRA, and QLoRA —
with a custom dataset built from AnyWord-3M.

Part of the **Efficient Image Generation with Text Rendering** project (ENSTA Paris, 2026).

> Full experimental results and analysis are available in the project report.

---

## Project structure

```
text-in-image-generation/
└── Finetune_with_LoRA/
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
├── train.json                        # Training samples
├── val.json                          # Validation samples
├── test.json                         # Test samples
└── test/images/                      # Test images
models/
├── flux2-klein-base-4b/              # Base model (downloaded)
├── full_finetune/                    # Full fine-tune checkpoint
├── lora_flux2klein/
│   ├── rank_8/
│   ├── rank_16/
│   ├── rank_32/
│   └── rank_64/
├── lora_cross_attention/
│   ├── rank_16/
│   └── rank_32/
└── qlora_cross_attention/
results/
├── metrics/                          # Per-experiment JSON files
├── plots/                            # Generated figures
├── generated_images/                 # Generated test images per experiment
└── fid_reference/                    # FID reference images from full fine-tune
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
├── train/images/
├── val/images/
└── test/images/
```

Each JSON file contains records with fields: `filepath`, `prompt`, `text`.

### 3. Train

```bash
# Full fine-tuning (15 epochs, early stopping patience=4)
python train_full_finetune.py

# LoRA rank sweep (r=8, 16, 32, 64 — 15 epochs each, early stopping patience=4)
python train_lora_rank_sweep.py

# Cross-attention LoRA (r=16 and r=32 — 20 epochs each, early stopping patience=5)
python train_lora_cross_attention.py

# QLoRA (4-bit NF4 + LoRA — 20 epochs, early stopping patience=5)
python train_qlora.py
```

All scripts save the **best checkpoint** (lowest validation loss) automatically and
stop early if validation loss does not improve for the configured patience.
Metrics, loss curves, and resource usage are saved to `results/metrics/`.

### 4. Evaluate

```bash
python evaluate_all.py
```

Builds the FID reference from the full fine-tune model first, then evaluates all
experiments against the same reference distribution for a fair comparison.
Generated images are saved to `results/generated_images/`.

### 5. Analyze OCR by text length

```bash
python analyze_ocr_by_length.py
```

Breakdown of OCR scores by word count (1-word / 2-word / 3+-word targets).
Runs on already-generated images — no GPU required.

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

## Training configuration

All experiments share the following setup:

- Precision: `bfloat16` mixed precision
- Optimizer: AdamW (8-bit for full fine-tuning and QLoRA, standard for LoRA)
- Gradient accumulation: 4 steps (effective batch size of 4)
- LR schedule: cosine with 10% warmup
- Frozen components: text encoder and VAE — only the transformer is adapted
- Text embeddings are pre-computed once before training to avoid redundant T5 forward passes
- Best checkpoint is selected by validation loss (flow-matching MSE in latent space)

| Script | Target modules | Epochs | Patience |
|---|---|---|---|
| `train_full_finetune.py` | All transformer params | 15 | 4 |
| `train_lora_rank_sweep.py` | to_q, to_k, to_v, to_out.0, ff.net.0.proj, ff.net.2 | 15 | 4 |
| `train_lora_cross_attention.py` | to_k, to_v | 20 | 5 |
| `train_qlora.py` | to_k, to_v (4-bit NF4 backbone) | 20 | 5 |

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