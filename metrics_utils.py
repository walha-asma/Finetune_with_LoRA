"""
Metrics tracking and evaluation utilities.
"""

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
            "quality": {},
            "inference": {}
        }
        
        # Setup emissions tracker
        self.emissions_tracker = EmissionsTracker(
            project_name=experiment_name,
            output_dir=str(self.output_dir),
            log_level="error"
        )
    
    # === Training Metrics ===
    
    def start_training(self):
        """Start tracking training metrics."""
        self.emissions_tracker.start()
        self.training_start_time = time.time()
        torch.cuda.reset_peak_memory_stats()
    
    def end_training(self, model, total_params):
        """End training and record metrics."""
        emissions = self.emissions_tracker.stop()
        training_time = time.time() - self.training_start_time
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_percentage = (trainable_params / total_params) * 100
        
        # Peak VRAM
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        self.metrics["training"] = {
            "trainable_params": trainable_params,
            "trainable_percentage": round(trainable_percentage, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "energy_kwh": round(emissions, 6) if emissions else 0,
            "carbon_gco2eq": round(emissions * 1000, 2) if emissions else 0,
            "training_time_hours": round(training_time / 3600, 2)
        }
    
    # === Quality Metrics ===
    
    def compute_fid(self, real_images, generated_images):
        """Compute Fréchet Inception Distance."""
        inception = inception_v3(pretrained=True, transform_input=False).eval()
        inception.fc = torch.nn.Identity()
        
        def get_activations(images):
            with torch.no_grad():
                activations = inception(images)
            return activations.cpu().numpy()
        
        real_act = get_activations(real_images)
        gen_act = get_activations(generated_images)
        
        mu1, sigma1 = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
        mu2, sigma2 = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2)**2)
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2*covmean)
        return float(fid)
    
    def compute_clip_score(self, images, prompts, clip_model, clip_processor):
        """Compute CLIP directional similarity."""
        from transformers import CLIPModel, CLIPProcessor
        
        with torch.no_grad():
            inputs = clip_processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True
            )
            
            outputs = clip_model(**inputs)
            similarity = outputs.logits_per_image.mean().item()
        
        return round(similarity, 4)
    
    def record_quality_metrics(self, fid_score=None, clip_score=None, val_loss=None):
        """Record quality metrics"""
        self.metrics["quality"] = {
            "fid": round(fid_score, 2) if fid_score is not None else None,  # ← AJOUT
            "clip_score": round(clip_score, 2) if clip_score is not None else None,  # ← AJOUT
            "validation_loss": round(val_loss, 4) if val_loss is not None else None  # ← AJOUT
        }
    
    # === Inference Metrics ===
    
    def measure_inference(self, model, test_prompts, num_runs=5):
        """Measure inference performance."""
        latencies = []
        
        torch.cuda.synchronize()
        for prompt in test_prompts[:num_runs]:
            start = time.time()
            
            with torch.no_grad():
                _ = model(prompt)
            
            torch.cuda.synchronize()
            latency = time.time() - start
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        throughput = 1.0 / avg_latency
        
        self.metrics["inference"] = {
            "avg_latency_s": round(avg_latency, 3),
            "throughput_img_per_s": round(throughput, 3)
        }
    
    # === Save Results ===
    
    def save(self):
        """Save all metrics to JSON."""
        output_file = self.output_dir / f"{self.experiment_name}.json"
        with open(output_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Metrics saved to {output_file}")
        return self.metrics
    
    def print_summary(self):
        """Print metrics summary."""
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.experiment_name}")
        print(f"{'='*60}")
        
        print("\n[TRAINING EFFICIENCY]")
        for k, v in self.metrics["training"].items():
            print(f"  {k}: {v}")
        
        print("\n[QUALITY & FIDELITY]")
        for k, v in self.metrics["quality"].items():
            print(f"  {k}: {v}")
        
        print("\n[INFERENCE PERFORMANCE]")
        for k, v in self.metrics["inference"].items():
            print(f"  {k}: {v}")
        
        print(f"{'='*60}\n")