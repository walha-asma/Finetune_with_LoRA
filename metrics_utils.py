import json
import time
import torch
import numpy as np
from pathlib import Path
from codecarbon import EmissionsTracker


class MetricsTracker:
    def __init__(self, experiment_name, output_dir="results/metrics"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            "experiment": experiment_name,
            "training": {},
            "validation": {},
            "test": {},
            "inference": {}
        }

        self.emissions_tracker = EmissionsTracker(
            project_name=experiment_name,
            output_dir=str(self.output_dir),
            log_level="error"
        )

    # === Training ===

    def start_training(self):
        self.emissions_tracker.start()
        self.training_start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

    def end_training(self, model, total_params, output_dir=None):
        emissions = self.emissions_tracker.stop()
        training_time = time.time() - self.training_start_time

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # Adapter / checkpoint size on disk
        adapter_size_mb = None
        if output_dir is not None:
            import os
            safetensors = list(Path(output_dir).rglob("*.safetensors"))
            if safetensors:
                total_bytes = sum(f.stat().st_size for f in safetensors)
                adapter_size_mb = round(total_bytes / (1024 ** 2), 2)

        self.metrics["training"] = {
            "trainable_params": trainable_params,
            "trainable_percentage": round(trainable_percentage, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "energy_kwh": round(emissions, 6) if emissions else 0,
            "carbon_gco2eq": round(emissions * 1000, 2) if emissions else 0,
            "training_time_hours": round(training_time / 3600, 2),
            "adapter_size_mb": adapter_size_mb,
        }

    def record_epoch_losses(self, epoch, train_loss, val_loss=None):
        if "loss_curve" not in self.metrics["training"]:
            self.metrics["training"]["loss_curve"] = []
        entry = {"epoch": epoch, "train_loss": round(train_loss, 4)}
        if val_loss is not None:
            entry["val_loss"] = round(val_loss, 4)
        self.metrics["training"]["loss_curve"].append(entry)

    # === FID ===

    def compute_fid(self, real_images_pil, generated_images_pil):
        """
        Uses src/evaluation/fid.py (shared with quantization workstream).
        Accepts lists of PIL images.

        The bfloat16 autocast context left over from training causes InceptionV3
        to fail. We disable autocast explicitly before running FID, and also
        force the inception model to float32 as a second safety net.
        Both measures together guarantee this works for ALL experiment types.
        """
        from src.evaluation.fid import (
            InceptionFeatureExtractor,
            compute_statistics,
            calculate_frechet_distance
        )
        import tempfile

        extractor = InceptionFeatureExtractor(device="cuda")
        # Safety net 1: force inception weights to float32
        extractor.model = extractor.model.float()

        with tempfile.TemporaryDirectory() as tmpdir:
            real_paths, gen_paths = [], []
            for i, img in enumerate(real_images_pil):
                p = Path(tmpdir) / f"real_{i:04d}.png"
                img.save(p)
                real_paths.append(p)
            for i, img in enumerate(generated_images_pil):
                p = Path(tmpdir) / f"gen_{i:04d}.png"
                img.save(p)
                gen_paths.append(p)

            # Safety net 2: disable autocast so bfloat16 context from training
            # does not bleed into InceptionV3 forward pass
            with torch.amp.autocast("cuda", enabled=False):
                real_features = extractor.extract_features(real_paths, batch_size=16)
                gen_features = extractor.extract_features(gen_paths, batch_size=16)

        mu1, sigma1 = compute_statistics(real_features)
        mu2, sigma2 = compute_statistics(gen_features)
        return float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))

    # === CLIP ===

    def compute_clip_score(self, images, prompts, clip_model, clip_processor):
        with torch.no_grad():
            inputs = clip_processor(
                text=prompts, images=images,
                return_tensors="pt", padding=True
            )
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
            outputs = clip_model(**inputs)
            similarity = outputs.logits_per_image.mean().item()
        return round(similarity, 4)

    # === OCR Accuracy ===

    def compute_ocr_accuracy(self, images, expected_texts):
        """
        Computes two OCR metrics using EasyOCR:

        1. ocr_exact_match  (strict)  : fraction of images where the FULL
           expected text appears as a substring in the OCR output.

        2. ocr_word_accuracy (lenient): average fraction of expected words
           found in the OCR output per image. Gives partial credit and is
           more informative on small datasets.

        Returns a dict with both metrics, or -1.0 if EasyOCR unavailable.
        """
        empty = {"ocr_exact_match": -1.0, "ocr_word_accuracy": -1.0}

        try:
            import easyocr
        except ImportError:
            print("  [WARNING] easyocr not installed. Run: pip install easyocr --break-system-packages")
            return empty

        try:
            reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)

            exact_correct = 0
            word_scores = []
            total = 0

            for image, expected in zip(images, expected_texts):
                if not expected:
                    continue
                total += 1

                img_np = np.array(image)
                results = reader.readtext(img_np, detail=0)
                ocr_output = " ".join(results).lower()
                expected_lower = expected.lower()

                # Metric 1: full string substring match
                if expected_lower in ocr_output:
                    exact_correct += 1

                # Metric 2: word-level match
                words = expected_lower.split()
                if words:
                    matched = sum(1 for w in words if w in ocr_output)
                    word_scores.append(matched / len(words))

            exact_match = round(exact_correct / total, 4) if total > 0 else 0.0
            word_accuracy = round(float(np.mean(word_scores)), 4) if word_scores else 0.0

            print(f"    OCR exact match:   {exact_match:.4f}")
            print(f"    OCR word accuracy: {word_accuracy:.4f}")

            return {"ocr_exact_match": exact_match, "ocr_word_accuracy": word_accuracy}

        except Exception as e:
            print(f"  [WARNING] OCR failed: {e}")
            return empty

    # === Record metrics ===

    def record_validation_metrics(self, val_loss):
        self.metrics["validation"]["val_loss"] = round(val_loss, 4)

    def record_test_metrics(self, fid=None, clip_score=None, ocr_results=None, inference_stats=None):
        self.metrics["test"] = {
            "fid": round(fid, 2) if fid is not None else None,
            "clip_score": round(clip_score, 4) if clip_score is not None else None,
        }
        if isinstance(ocr_results, dict):
            em = ocr_results.get("ocr_exact_match", -1.0)
            wa = ocr_results.get("ocr_word_accuracy", -1.0)
            self.metrics["test"]["ocr_exact_match"] = em if em >= 0 else "skipped"
            self.metrics["test"]["ocr_word_accuracy"] = wa if wa >= 0 else "skipped"
        else:
            self.metrics["test"]["ocr_exact_match"] = "skipped"
            self.metrics["test"]["ocr_word_accuracy"] = "skipped"

        if inference_stats:
            self.metrics["inference"] = inference_stats

    # === Save / Print ===

    def save(self):
        output_file = self.output_dir / f"{self.experiment_name}.json"
        with open(output_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Metrics saved to {output_file}")
        return self.metrics

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.experiment_name}")
        print(f"{'='*60}")

        print("\n[TRAINING EFFICIENCY]")
        for k, v in self.metrics["training"].items():
            if k != "loss_curve":
                print(f"  {k}: {v}")

        print("\n[VALIDATION]")
        for k, v in self.metrics["validation"].items():
            print(f"  {k}: {v}")

        print("\n[TEST METRICS]")
        for k, v in self.metrics["test"].items():
            print(f"  {k}: {v}")

        print("\n[INFERENCE]")
        for k, v in self.metrics["inference"].items():
            print(f"  {k}: {v}")

        print(f"{'='*60}\n")