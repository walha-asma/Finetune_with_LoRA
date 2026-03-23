import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

METRICS_DIR = Path("results/metrics")
PLOTS_DIR   = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Color per experiment group
COLORS = {
    "original_baseline":           "#888888",
    "full_finetune":               "#E05C2A",
    "lora_flux2klein_rank8":       "#3B8BD4",
    "lora_flux2klein_rank16":      "#2A6FBB",
    "lora_flux2klein_rank32":      "#1A56A0",
    "lora_flux2klein_rank64":      "#0F3D80",
    "lora_cross_attention_rank16": "#1D9E75",
    "lora_cross_attention_rank32": "#0F6E56",
    "qlora_cross_attention_rank16":"#BA7517",
}

LABELS = {
    "original_baseline":           "Original",
    "full_finetune":               "Full FT",
    "lora_flux2klein_rank8":       "LoRA r8",
    "lora_flux2klein_rank16":      "LoRA r16",
    "lora_flux2klein_rank32":      "LoRA r32",
    "lora_flux2klein_rank64":      "LoRA r64",
    "lora_cross_attention_rank16": "CA-LoRA r16",
    "lora_cross_attention_rank32": "CA-LoRA r32",
    "qlora_cross_attention_rank16":"QLoRA r16",
}

def load_all_metrics():
    """Load individual experiment JSON files, return dict keyed by exp name."""
    data = {}
    for path in METRICS_DIR.glob("*.json"):
        if path.name == "final_comparison.json":
            continue
        name = path.stem
        with open(path) as f:
            data[name] = json.load(f)
    return data


def safe(d, *keys, default=None):
    """Safe nested dict access."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d

def _pareto_front(xs, ys, lower_x_better=True, lower_y_better=True):
    """Return indices of Pareto-optimal points (both axes: lower = better by default)."""
    points = list(zip(xs, ys, range(len(xs))))
    pareto = []
    for x, y, i in points:
        dominated = False
        for x2, y2, j in points:
            if i == j:
                continue
            x_better = (x2 <= x) if lower_x_better else (x2 >= x)
            y_better = (y2 <= y) if lower_y_better else (y2 >= y)
            x_strict = (x2 < x)  if lower_x_better else (x2 > x)
            y_strict = (y2 < y)  if lower_y_better else (y2 > y)
            if (x_better and y_better) and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto


def plot_tradeoff_scatter(all_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    vrams, fids, ocr_ws, pcts = [], [], [], []
    names_p1, names_p2 = [], []
    colors_p1, colors_p2 = [], []

    for name, m in all_metrics.items():
        fid   = safe(m, "test", "fid")
        ocr_w = safe(m, "test", "ocr_word_accuracy")
        vram  = safe(m, "training", "peak_vram_gb")
        pct   = safe(m, "training", "trainable_percentage")
        color = COLORS.get(name, "#999999")
        label = LABELS.get(name, name)

        # Plot 1: VRAM vs FID (both lower = better)
        if fid is not None and vram is not None and fid > 0:
            ax1.scatter(vram, fid, color=color, s=120, zorder=5,
                        edgecolors="white", linewidths=0.8)
            ax1.annotate(label, (vram, fid),
                         textcoords="offset points", xytext=(6, 4),
                         fontsize=8, color=color)
            vrams.append(vram); fids.append(fid)
            names_p1.append(label); colors_p1.append(color)

        # Plot 2: trainable% vs OCR word accuracy (lower% better, higher OCR better)
        if ocr_w is not None and isinstance(ocr_w, float) and pct is not None:
            ax2.scatter(pct, ocr_w, color=color, s=120, zorder=5,
                        edgecolors="white", linewidths=0.8)
            ax2.annotate(label, (pct, ocr_w),
                         textcoords="offset points", xytext=(6, 4),
                         fontsize=8, color=color)
            pcts.append(pct); ocr_ws.append(ocr_w)
            names_p2.append(label); colors_p2.append(color)

    # --- Pareto front plot 1: lower VRAM + lower FID = better ---
    if len(vrams) > 1:
        pidx = _pareto_front(vrams, fids, lower_x_better=True, lower_y_better=True)
        px = sorted([(vrams[i], fids[i]) for i in pidx], key=lambda t: t[0])
        ax1.plot([p[0] for p in px], [p[1] for p in px],
                 color="black", linewidth=1.2, linestyle="--",
                 zorder=4, label="Pareto front")
        ax1.legend(fontsize=8)

    # --- Pareto front plot 2: lower trainable% + higher OCR = better ---
    if len(pcts) > 1:
        pidx = _pareto_front(pcts, [-v for v in ocr_ws],
                             lower_x_better=True, lower_y_better=True)
        px = sorted([(pcts[i], ocr_ws[i]) for i in pidx], key=lambda t: t[0])
        ax2.plot([p[0] for p in px], [p[1] for p in px],
                 color="black", linewidth=1.2, linestyle="--",
                 zorder=4, label="Pareto front")
        ax2.legend(fontsize=8)

    ax1.set_xscale("log")
    ax1.set_xlabel("Peak VRAM (GB, log scale)", fontsize=11)
    ax1.set_ylabel("FID ↓ (lower is better)", fontsize=11)
    ax1.set_title("Quality vs Hardware Cost", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.set_xscale("log")
    ax2.set_xlabel("Trainable parameters (%, log scale)", fontsize=11)
    ax2.set_ylabel("OCR word accuracy ↑", fontsize=11)
    ax2.set_title("Text Rendering vs Parameter Efficiency", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = PLOTS_DIR / "tradeoff_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_radar(all_metrics):
    axes_labels = ["FID\n(inv)", "CLIP\nscore", "OCR\nword acc", "CER\n(inv)", "VRAM\n(inv)"]
    N = len(axes_labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Collect raw values to normalise
    raw = {name: {} for name in all_metrics}
    for name, m in all_metrics.items():
        raw[name]["fid"]  = safe(m, "test", "fid")
        raw[name]["clip"] = safe(m, "test", "clip_score")
        raw[name]["ocr"]  = safe(m, "test", "ocr_word_accuracy")
        raw[name]["cer"]  = safe(m, "test", "ocr_cer")
        raw[name]["vram"] = safe(m, "training", "peak_vram_gb")

    # Min/max per axis for normalisation
    def minmax(key):
        vals = [v[key] for v in raw.values()
                if v[key] is not None and isinstance(v[key], (int, float))]
        return (min(vals), max(vals)) if vals else (0, 1)

    fid_min,  fid_max  = minmax("fid")
    clip_min, clip_max = minmax("clip")
    ocr_min,  ocr_max  = minmax("ocr")
    cer_min,  cer_max  = minmax("cer")
    vram_min, vram_max = minmax("vram")

    def norm(val, lo, hi, invert=False):
        if val is None or not isinstance(val, (int, float)):
            return 0.0
        if hi == lo:
            return 0.5
        n = (val - lo) / (hi - lo)
        return 1.0 - n if invert else n

    for name, m in all_metrics.items():
        r = raw[name]
        # All axes: higher = better on radar
        values = [
            norm(r["fid"],  fid_min,  fid_max,  invert=True),
            norm(r["clip"], clip_min, clip_max,  invert=False),
            norm(r["ocr"],  ocr_min,  ocr_max,   invert=False),
            norm(r["cer"],  cer_min,  cer_max,   invert=True),   # lower CER = better
            norm(r["vram"], vram_min, vram_max,  invert=True),
        ]
        # Skip if all zeros (missing data)
        if all(v == 0.0 for v in values):
            continue

        values += values[:1]
        color = COLORS.get(name, "#999999")
        label = LABELS.get(name, name)

        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=11)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="gray")
    ax.set_ylim(0, 1)
    ax.set_title("Overall Experiment Comparison\n(larger polygon = better)",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)

    out = PLOTS_DIR / "radar_chart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")

def plot_adapter_sizes(all_metrics):
    names, sizes, colors = [], [], []
    for name, m in all_metrics.items():
        size = safe(m, "training", "adapter_size_mb")
        if size is not None:
            names.append(LABELS.get(name, name))
            sizes.append(size)
            colors.append(COLORS.get(name, "#999999"))

    if not names:
        print("  [SKIP] No adapter_size_mb data found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, sizes, color=colors, edgecolor="white", height=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Storage size (MB, log scale)", fontsize=11)
    ax.set_title("Adapter / Checkpoint Size on Disk", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_xlim(1, max(sizes) * 3)
    for bar, size in zip(bars, sizes):
        label = f"{size:.0f} MB" if size < 1000 else f"{size/1024:.1f} GB"
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / "adapter_sizes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_loss_curves(all_metrics):
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False

    for name, m in all_metrics.items():
        curve = safe(m, "training", "loss_curve")
        if not curve:
            continue
        epochs     = [e["epoch"]      for e in curve]
        train_loss = [e["train_loss"] for e in curve]
        val_loss   = [e.get("val_loss") for e in curve]
        color = COLORS.get(name, "#999999")
        label = LABELS.get(name, name)

        ax.plot(epochs, train_loss, color=color, linewidth=1.8,
                label=f"{label} train")
        if any(v is not None for v in val_loss):
            ax.plot(epochs, val_loss, color=color, linewidth=1.2,
                    linestyle="--", alpha=0.7, label=f"{label} val")
        plotted = True

    if not plotted:
        print("  [SKIP] No loss_curve data found.")
        plt.close()
        return

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("Training and Validation Loss Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = PLOTS_DIR / "loss_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_ocr_comparison(all_metrics):
    """
    Grouped horizontal bar chart showing all three OCR metrics
    (exact match, word accuracy, CER) side by side per experiment.
    Makes it easy to see which models actually render text correctly
    after the normalization fixes.
    """
    # Preserve experiment order
    order = [
        "original_baseline", "full_finetune",
        "lora_flux2klein_rank8", "lora_flux2klein_rank16",
        "lora_flux2klein_rank32", "lora_flux2klein_rank64",
        "lora_cross_attention_rank16", "lora_cross_attention_rank32",
        "qlora_cross_attention_rank16",
    ]

    names, exact, word_acc, cer = [], [], [], []
    for name in order:
        m = all_metrics.get(name)
        if m is None:
            continue
        em  = safe(m, "test", "ocr_exact_match")
        wa  = safe(m, "test", "ocr_word_accuracy")
        c   = safe(m, "test", "ocr_cer")
        if not isinstance(em, float): em = 0.0
        if not isinstance(wa, float): wa = 0.0
        if not isinstance(c,  float): c  = 1.0
        names.append(LABELS.get(name, name))
        exact.append(em)
        word_acc.append(wa)
        cer.append(c)

    if not names:
        print("  [SKIP] No OCR data found.")
        return

    y = np.arange(len(names))
    h = 0.25

    fig, ax = plt.subplots(figsize=(11, max(5, len(names) * 0.7)))
    ax.barh(y + h,   exact,    height=h, label="Exact match ↑",   color="#3B8BD4", edgecolor="white")
    ax.barh(y,       word_acc, height=h, label="Word accuracy ↑", color="#1D9E75", edgecolor="white")
    ax.barh(y - h,   cer,      height=h, label="CER ↓",           color="#E05C2A", edgecolor="white", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_title("OCR metrics per experiment (normalized comparison)",
                 fontsize=13, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = PLOTS_DIR / "ocr_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def main():
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    all_metrics = load_all_metrics()
    if not all_metrics:
        print("  [ERROR] No metrics JSON files found in results/metrics/")
        print("  Run evaluate_all.py first.")
        return

    print(f"  Loaded metrics for {len(all_metrics)} experiments:")
    for name in all_metrics:
        print(f"    - {name}")

    print("\n  Generating plots...")
    plot_tradeoff_scatter(all_metrics)
    plot_radar(all_metrics)
    plot_adapter_sizes(all_metrics)
    plot_loss_curves(all_metrics)
    plot_ocr_comparison(all_metrics)

    print(f"\n All plots saved to {PLOTS_DIR.absolute()}/")
    print("=" * 60)


if __name__ == "__main__":
    main()