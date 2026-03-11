"""
Download FLUX.2 Klein 4B base model (optimized for fine-tuning).
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_flux2_klein(
    model_id="black-forest-labs/FLUX.2-klein-base-4B",  # Base = better for fine-tuning
    output_dir="models/flux2-klein-base-4b",
    hf_token=None
):
    """
    Download FLUX.2 Klein 4B base model.
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Model ID: {model_id}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"HF_HOME: {os.getenv('HF_HOME', 'default')}")
    print("")
    
    # Get token
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    
    if hf_token:
        print("✓ Using HuggingFace token")
    else:
        print("⚠ No token (may be required)")
    
    print("")
    print("Starting download...")
    print("This may take 20-40 minutes")
    print("")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(output_path),
            token=hf_token,
            max_workers=4,
            ignore_patterns=["*.md", "*.txt", ".git*"]
        )
        
        print("")
        print("="*70)
        print("✓ MODEL DOWNLOAD COMPLETE")
        print("="*70)
        
        # Verify
        verify_download(output_path)
        
        # Size
        total_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        print(f"Total size: {size_gb:.2f} GB")
        print(f"Location: {output_path.absolute()}")
        print("="*70)
        
        return str(output_path)
        
    except Exception as e:
        print("")
        print("="*70)
        print("✗ DOWNLOAD FAILED")
        print("="*70)
        print(f"Error: {e}")
        
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print("")
            print("Authentication required:")
            print("1. https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B")
            print("2. Accept license")
            print("3. Get token: https://huggingface.co/settings/tokens")
            print("4. export HF_TOKEN='your_token'")
        
        print("="*70)
        sys.exit(1)

def verify_download(model_path):
    """Verify essential files."""
    print("\nVerifying download...")
    
    # Klein has different structure than dev
    essential_files = [
        "model_index.json"
    ]
    
    all_found = True
    for file in essential_files:
        file_path = model_path / file
        exists = file_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_found = False
    
    if all_found:
        print("\n✓ All essential files present")
    else:
        print("\n⚠ Some files missing, but may still work")
    
    return all_found

def main():
    print("="*70)
    print("FLUX.2 KLEIN 4B BASE - DOWNLOAD")
    print("="*70)
    print("")
    
    download_flux2_klein()

if __name__ == "__main__":
    main()