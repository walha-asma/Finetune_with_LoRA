import json
import time
import torch
import numpy as np
from pathlib import Path
from codecarbon import EmissionsTracker
from scipy.linalg import sqrtm
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3


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

    def end_training(self, model, total_params):
        emissions = self.emissions_tracker.stop()
        training_time = time.time() - self.training_start_time

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

        self.metrics["training"] = {
            "trainable_params": trainable_params,
            "trainable_percentage": round(trainable_percentage, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "energy_kwh": round(emissions, 6) if emissions else 0,
            "carbon_gco2eq": round(emissions * 1000, 2) if emissions else 0,
            "training_time_hours": round(training_time / 3600, 2)
        }

    def record_epoch_losses(self, epoch, train_loss, val_loss=None):
        if "loss_curve" not in self.metrics["training"]:
            self.metrics["training"]["loss_curve"] = []
        entry = {"epoch": epoch, "train_loss": round(train_loss, 4)}
        if val_loss is not None:
            entry["val_loss"] = round(val_loss, 4)
        self.metrics["training"]["loss_curve"].append(entry)

    # === FID ===

    def compute_fid(self, real_images, generated_images):
        inception = inception_v3(pretrained=True, transform_input=False).eval().to("cuda")
        inception.fc = torch.nn.Identity()

        def get_activations(images):
            with torch.no_grad():
                return inception(images).cpu().numpy()

        real_act = get_activations(real_images)
        gen_act = get_activations(generated_images)

        mu1, sigma1 = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
        mu2, sigma2 = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)

        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

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
        For each generated image, run Tesseract and check if the expected
        text appears in the OCR output (case-insensitive).
        Returns accuracy as a float between 0 and 1.
        """
        try:
            import pytesseract
        except ImportError:
            print("pytesseract not installed, skipping OCR accuracy")
            return None

        correct = 0
        for image, expected in zip(images, expected_texts):
            if not expected:
                continue
            ocr_output = pytesseract.image_to_string(image).lower()
            if expected.lower() in ocr_output:
                correct += 1

        accuracy = correct / len(expected_texts) if expected_texts else 0
        return round(accuracy, 4)

    # === Record metrics ===

    def record_validation_metrics(self, val_loss):
        self.metrics["validation"]["val_loss"] = round(val_loss, 4)

    def record_test_metrics(self, fid=None, clip_score=None, ocr_accuracy=None, inference_stats=None):
        self.metrics["test"] = {
            "fid": round(fid, 2) if fid is not None else None,
            "clip_score": round(clip_score, 4) if clip_score is not None else None,
            "ocr_accuracy": round(ocr_accuracy, 4) if ocr_accuracy is not None else None,
        }
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
