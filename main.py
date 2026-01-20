import torch
import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from src.pipeline import EEG2DreamPipeline
from src.video_pipeline import EEG2VideoPipeline
from src.data import get_dataloader

def save_visualization(eeg_data, generated_image, epoch, save_dir="results"):
    """
    保存 EEG 输入和生成的梦境图像
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot EEG (First Channel)
    plt.figure(figsize=(10, 4))
    plt.plot(eeg_data.cpu().numpy()[0, :], label="EEG Channel 0")
    plt.title(f"Input EEG Signal (Epoch {epoch})")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"eeg_input_epoch_{epoch}.png"))
    plt.close()
    
    # 2. Save Dream Image
    # generated_image is numpy array (H, W, 3) normalized [0,1]
    img_uint8 = (generated_image * 255).astype(np.uint8)
    if img_uint8.shape[-1] == 1: # Grayscale check
        img_uint8 = np.repeat(img_uint8, 3, axis=-1)
        
    # Remove batch dim if present (1, H, W, 3) -> (H, W, 3)
    if img_uint8.ndim == 4:
        img_uint8 = img_uint8[0]
        
    Image.fromarray(img_uint8).save(os.path.join(save_dir, f"dream_output_epoch_{epoch}.png"))
    print(f"Saved visualization to {save_dir}/")

def main():
    print("=== Phase 1: The Semantic Bridge (MVP) ===")
    
    # 1. Environment Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # 2. Data Loading (Load Real Data First to get Dimensions)
    print("\n[Step 1] Loading Real EEG Data (MOABB - BNCI2014001)...")
    # use_real_data=True will trigger MOABB download
    dataloader = get_dataloader(batch_size=2, use_real_data=True)
    sample_batch = next(iter(dataloader))
    
    # Get dimensions from data
    eeg_channels = sample_batch['eeg'].shape[1]
    eeg_time_steps = sample_batch['eeg'].shape[2]
    print(f"Loaded Data Shape: {sample_batch['eeg'].shape}") 
    print(f"Detected Channels: {eeg_channels}, Time Steps: {eeg_time_steps}")
    print(f"CLIP Target Shape: {sample_batch['clip_embed'].shape}") 

    # 3. Instantiate Pipeline with Correct Dimensions
    print("\n[Step 2] Initializing Pipeline (Phase 2: Video Capable)...")
    pipeline = None
    try:
        # Use EEG2VideoPipeline which extends EEG2DreamPipeline
        pipeline = EEG2VideoPipeline(
            device=device,
            num_eeg_channels=eeg_channels,
            eeg_time_steps=eeg_time_steps
        )
    except Exception as e:
        print(f"Warning: Could not download/load models: {e}")
        print("Retrying in DEBUG mode (using mock SD components)...")
        # Fallback to base pipeline in debug mode
        pipeline = EEG2DreamPipeline(
            device=device, 
            debug=True,
            num_eeg_channels=eeg_channels,
            eeg_time_steps=eeg_time_steps
        )

    # 4. Forward Pass Check (Before Training)
    print("\n[Step 3] Running Forward Pass Check...")
    if pipeline is not None:
        eeg_sample = sample_batch['eeg'].to(device)
        
        # Test Encoder
        encoded = pipeline.eeg_encoder(eeg_sample)
        print(f"EEG Encoder Output: {encoded.shape} (Expected: [B, 768])")
        
        # Test Adapter
        adapted = pipeline.dream_adapter(encoded)
        print(f"Dream Adapter Output: {adapted.shape} (Expected: [B, 77, 768])")
        
        # 5. Training Loop (Demo)
        print("\n[Step 4] Starting Demo Training (1 Epoch)...")
        pipeline.train(dataloader, num_epochs=1)
        
        # 6. Generation Check
        print("\n[Step 5] Generating Dream Image...")
        # Take the first sample from batch
        single_eeg = eeg_sample[0] 
        images = pipeline.generate(single_eeg)
        print(f"Generated Image Shape: {images.shape}")
        
        # Save Result
        save_visualization(single_eeg, images, epoch=1)

        # 7. Phase 2: Video Generation
        print("\n=== Phase 2: The 4D World (Video Generation) ===")
        if isinstance(pipeline, EEG2VideoPipeline) and pipeline.svd_pipeline is not None:
            print("Stable Video Diffusion is ready. Generating video from dream...")
            video_path = os.path.join("results", "dream_video.mp4")
            try:
                # Generate video using the same EEG sample
                # Use fewer steps for CPU demo to avoid long wait (default 25)
                steps = 2 if device == "cpu" else 25
                print(f"Running SVD with {steps} inference steps for demo...")
                saved_path = pipeline.generate_video(single_eeg, output_path=video_path, num_inference_steps=steps)
                if saved_path:
                    print(f"SUCCESS: Video saved to {saved_path}")
                else:
                    print("Video generation returned None.")
            except Exception as e:
                print(f"Error during video generation: {e}")
        else:
            print("Skipping video generation (SVD not loaded or pipeline in debug mode).")

if __name__ == "__main__":
    main()
