"""FID (Fréchet Inception Distance) metric implementation"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
from scipy import linalg
from typing import Union, List
import pandas as pd


class InceptionFeatureExtractor:
    """Extract features from images using InceptionV3"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Load InceptionV3 and remove final layers
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.fc = nn.Identity()
        inception.eval()
        self.model = inception.to(device)
        
        # InceptionV3 expects 299x299 images
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    @torch.no_grad()
    def extract_features(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
        """Extract features from list of image paths"""
        features_list = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load and preprocess images
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensor = self.transform(img)
                    images.append(img_tensor)
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")
                    continue
            
            if not images:
                continue
            
            # Stack into batch and move to device
            batch = torch.stack(images).to(self.device)
            
            # Extract features
            features = self.model(batch)
            features_list.append(features.cpu().numpy())
        
        if not features_list:
            raise ValueError("No valid images found")
        
        return np.vstack(features_list)


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray,
                               mu2: np.ndarray, sigma2: np.ndarray,
                               eps: float = 1e-6) -> float:
    """Calculate Fréchet distance between two Gaussian distributions"""
    
    # Calculate difference in means
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    # Calculate FID
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    return float(fid)


def compute_statistics(features: np.ndarray):
    """Compute mean and covariance of features"""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(ref_images: Union[str, Path, List[Path]], 
                gen_images: Union[str, Path, List[Path]],
                device: str = 'cuda',
                batch_size: int = 32) -> float:
    """
    Compute FID between reference and generated images
    
    Args:
        ref_images: Path to reference images directory or list of image paths
        gen_images: Path to generated images directory or list of image paths
        device: Device to use for computation
        batch_size: Batch size for feature extraction
    
    Returns:
        FID score (lower is better)
    """
    
    # Convert to list of paths
    if isinstance(ref_images, (str, Path)):
        ref_paths = sorted(Path(ref_images).rglob('*.png'))
    else:
        ref_paths = [Path(p) for p in ref_images]
    
    if isinstance(gen_images, (str, Path)):
        gen_paths = sorted(Path(gen_images).rglob('*.png'))
    else:
        gen_paths = [Path(p) for p in gen_images]
    
    if not ref_paths:
        raise ValueError(f"No reference images found")
    if not gen_paths:
        raise ValueError(f"No generated images found")
    
    print(f"Computing FID between {len(ref_paths)} reference and {len(gen_paths)} generated images...")
    
    # Extract features
    extractor = InceptionFeatureExtractor(device=device)
    ref_features = extractor.extract_features(ref_paths, batch_size)
    gen_features = extractor.extract_features(gen_paths, batch_size)
    
    # Compute statistics
    mu1, sigma1 = compute_statistics(ref_features)
    mu2, sigma2 = compute_statistics(gen_features)
    
    # Calculate FID
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    return fid


def compare_models_fid(output_dir: Union[str, Path],
                       reference_model: str = "FLUX2-Klein-FP16",
                       device: str = 'cuda',
                       batch_size: int = 32) -> pd.DataFrame:
    """
    Compare all models against reference model using FID
    
    Args:
        output_dir: Directory containing prompt folders with seed subfolders
        reference_model: Name of reference model (default: FLUX2-Klein-FP16)
        device: Device for computation
        batch_size: Batch size for feature extraction
    
    Returns:
        DataFrame with FID scores for each model
    """
    
    output_dir = Path(output_dir)
    
    # Find all models by looking in any seed folder
    prompt_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if not prompt_dirs:
        raise ValueError(f"No prompt directories found in {output_dir}")
    
    # Get first seed folder to find available models
    first_prompt = prompt_dirs[0]
    seed_dirs = [d for d in first_prompt.iterdir() if d.is_dir() and d.name.startswith('seed_')]
    if not seed_dirs:
        raise ValueError(f"No seed directories found in {first_prompt}")
    
    first_seed = seed_dirs[0]
    all_models = [p.stem for p in first_seed.glob('*.png')]
    
    if reference_model not in all_models:
        raise ValueError(f"Reference model {reference_model} not found. Available: {all_models}")
    
    # Get comparison models (all except reference)
    comparison_models = [m for m in all_models if m != reference_model]
    
    print(f"Found {len(all_models)} models: {all_models}")
    print(f"Computing FID for {len(comparison_models)} models against {reference_model}")
    
    # Collect all reference and comparison images
    results = []
    
    for model in comparison_models:
        ref_images = []
        gen_images = []
        
        # Collect images from all prompts and seeds
        for prompt_dir in prompt_dirs:
            for seed_dir in prompt_dir.glob('seed_*'):
                ref_img = seed_dir / f"{reference_model}.png"
                gen_img = seed_dir / f"{model}.png"
                
                if ref_img.exists() and gen_img.exists():
                    ref_images.append(ref_img)
                    gen_images.append(gen_img)
        
        if not ref_images:
            print(f"Warning: No images found for {model}, skipping")
            continue
        
        print(f"\n{model}: Evaluating {len(ref_images)} image pairs...")
        fid_score = compute_fid(ref_images, gen_images, device, batch_size)
        
        results.append({
            'model': model,
            'reference': reference_model,
            'fid_score': fid_score,
            'num_images': len(ref_images)
        })
        
        print(f"  FID: {fid_score:.2f}")
    
    df = pd.DataFrame(results)
    return df
