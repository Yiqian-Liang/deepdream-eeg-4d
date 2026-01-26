import torch
import sys
import os
import gc
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

# Set allocator config to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.pipeline import EEG2DreamPipeline
from src.video_pipeline import EEG2VideoPipeline
from src.data_loader import ThingsEEGDataset, DreamDataset

def save_visualization(eeg_data, generated_image, epoch, stage_name, save_dir="results"):
    """
    保存 EEG 输入和生成的梦境图像
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot EEG (First Channel)
    plt.figure(figsize=(10, 4))
    # Handle tensor or numpy
    if isinstance(eeg_data, torch.Tensor):
        eeg_plot = eeg_data.cpu().numpy()
    else:
        eeg_plot = eeg_data
        
    plt.plot(eeg_plot[0, :], label="EEG Channel 0")
    plt.title(f"Input EEG Signal ({stage_name} - Epoch {epoch})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"eeg_input_{stage_name}_{epoch}.png"))
    plt.close()
    
    # 2. Save Dream Image
    # generated_image is numpy array (H, W, 3) normalized [0,1]
    img_uint8 = (generated_image * 255).astype(np.uint8)
    if img_uint8.shape[-1] == 1: # Grayscale check
        img_uint8 = np.repeat(img_uint8, 3, axis=-1)
        
    # Remove batch dim if present (1, H, W, 3) -> (H, W, 3)
    if img_uint8.ndim == 4:
        img_uint8 = img_uint8[0]
        
    Image.fromarray(img_uint8).save(os.path.join(save_dir, f"dream_output_{stage_name}_{epoch}.png"))
    print(f"Saved visualization to {save_dir}/")

def main():
    print("=== DeepDream-4D: Dual-Stage Dream Reconstruction ===")
    print("Strategy: Awake Visual Pre-training (THINGS) -> Dream Semantic Fine-tuning (DREAM)")
    
    # 1. Environment Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # --- Stage A: Visual Pre-training (THINGS-EEG) ---
    print("\n[Stage A] Visual Pre-training (THINGS-EEG - Awake Perception)...")
    # Initialize Dataset (Use dummy for MVP/Colab speed)
    things_dataset = ThingsEEGDataset(num_samples=100, dummy_fallback=True)
    things_loader = DataLoader(things_dataset, batch_size=2, shuffle=True)
    
    # Get dimensions from data to initialize model correctly
    sample_batch = next(iter(things_loader))
    eeg_channels = sample_batch['eeg'].shape[1]
    eeg_time_steps = sample_batch['eeg'].shape[2]
    print(f"Stage A Data Shape: {sample_batch['eeg'].shape}") 
    print(f"Detected Channels: {eeg_channels}, Time Steps: {eeg_time_steps}")

    # Initialize Pipeline (Phase 2 Video Capable)
    print("\n[Init] Initializing Pipeline...")
    pipeline = None
    try:
        pipeline = EEG2VideoPipeline(
            device=device,
            num_eeg_channels=eeg_channels,
            eeg_time_steps=eeg_time_steps
        )
    except Exception as e:
        print(f"Warning: Could not load Video Pipeline: {e}")
        print("Fallback to Base Pipeline...")
        pipeline = EEG2DreamPipeline(
            device=device,
            num_eeg_channels=eeg_channels,
            eeg_time_steps=eeg_time_steps
        )

    # Train Stage A (Image Targets)
    print("\nStarting Stage A Training (Visual Alignment)...")
    pipeline.train(things_loader, num_epochs=1, mode="pretrain")
    
    # --- Stage B: Dream Fine-tuning (DREAM Database) ---
    print("\n[Stage B] Dream Fine-tuning (DREAM Database - REM Sleep)...")
    # Initialize Dataset (Downloads from Tsinghua Mirror or generates synthetic)
    dream_dataset = DreamDataset(num_samples=50, use_tsinghua_mirror=True)
    dream_loader = DataLoader(dream_dataset, batch_size=2, shuffle=True)
    
    print("Starting Stage B Training (Semantic Alignment)...")
    # Note: We assume the pipeline/model handles the potentially different data distribution 
    # via the Adapter. The channel count must match Stage A or be padded.
    # Our synthetic data ensures matching (63 channels).
    pipeline.train(dream_loader, num_epochs=1, mode="finetune")
    
    # --- Generation ---
    print("\n[Inference] Generating Dream Content...")
    # Get a sample from Dream Dataset
    dream_sample = next(iter(dream_loader))
    eeg_input = dream_sample['eeg'].to(device)
    raw_text = dream_sample['raw_text'][0]
    print(f"Dream Report: '{raw_text}'")
    
    # Generate Image
    print("Generating Image...")
    images = pipeline.generate(eeg_input[0])
    
    # Save
    save_visualization(eeg_input[0], images, epoch="final", stage_name="dream_reconstruction")
    
    # --- Video Generation ---
    print("\n[Phase 2] Generating 4D Video...")
    if isinstance(pipeline, EEG2VideoPipeline):
        print("Stable Video Diffusion capability detected. Attempting video generation...")
        video_path = os.path.join("results", "dream_video.mp4")
        try:
            # Generate video using the same EEG sample
            steps = 2 if device == "cpu" else 25
            print(f"Running SVD with {steps} inference steps...")
            
            # Memory Cleanup before heavy video generation
            torch.cuda.empty_cache()
            gc.collect()
            
            saved_path = pipeline.generate_video(eeg_input[0], output_path=video_path, num_inference_steps=steps)
            if saved_path:
                print(f"SUCCESS: Video saved to {saved_path}")
            else:
                print("Video generation returned None (possibly due to memory or load failure).")
        except Exception as e:
            print(f"Error during video generation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping video generation (Pipeline is not EEG2VideoPipeline).")

if __name__ == "__main__":
    main()
