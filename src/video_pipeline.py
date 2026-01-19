import torch
import numpy as np
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from src.pipeline import EEG2DreamPipeline
import os

class EEG2VideoPipeline(EEG2DreamPipeline):
    """
    Phase 2: From 2D Dreams to 4D Worlds
    
    This pipeline extends the base image pipeline to generate video.
    It uses Stable Video Diffusion (SVD) conditioned on:
    1. The initial "Dream Image" (generated from EEG in Phase 1)
    2. Temporal dynamics extracted from the EEG signal (Future Work: Motion Bucket Control)
    """
    def __init__(self, 
                 image_model_id="runwayml/stable-diffusion-v1-5", 
                 video_model_id="stabilityai/stable-video-diffusion-img2vid-xt",
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 **kwargs):
        
        # Initialize Phase 1 (Image Generation) components
        super().__init__(model_id=image_model_id, device=device, **kwargs)
        
        self.video_model_id = video_model_id
        print(f"Loading Stable Video Diffusion components from {video_model_id}...")
        
        try:
            self.svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                video_model_id, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
                variant="fp16" if device == "cuda" else None
            )
            # Enable model cpu offload to save VRAM if on GPU
            if device == "cuda":
                self.svd_pipeline.enable_model_cpu_offload()
            else:
                self.svd_pipeline.to(device)
                
        except Exception as e:
            print(f"Warning: SVD load failed ({e}). Video generation will not work.")
            self.svd_pipeline = None

    @torch.no_grad()
    def generate_video(self, eeg_sample, output_path="dream_video.mp4"):
        """
        Full Pipeline: EEG -> Image -> Video
        """
        if self.svd_pipeline is None:
            print("Error: SVD Pipeline not loaded.")
            return None

        # Step 1: Generate Anchor Image (Phase 1)
        print("Generating Anchor Image from EEG...")
        # Note: self.generate returns numpy array (H, W, 3)
        image_np = self.generate(eeg_sample)
        
        # SVD expects PIL Image
        from PIL import Image
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        if image_pil.size != (1024, 576):
            image_pil = image_pil.resize((1024, 576)) # SVD preferred resolution
            
        # Step 2: Generate Video (Phase 2)
        print("Dreaming in 4D (Generating Video)...")
        frames = self.svd_pipeline(
            image_pil, 
            decode_chunk_size=8,
            generator=torch.manual_seed(42),
            motion_bucket_id=127, # Higher = more motion
            noise_aug_strength=0.1
        ).frames[0]

        # Step 3: Save
        print(f"Saving video to {output_path}...")
        export_to_video(frames, output_path, fps=7)
        return output_path
