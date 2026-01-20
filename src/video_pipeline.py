import torch
import numpy as np
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from src.pipeline import EEG2DreamPipeline
import os
import gc

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
        # Note: We do NOT load SVD here to save memory. 
        # SVD will be loaded lazily in generate_video AFTER destroying Phase 1 models.
        self.svd_pipeline = None

    def _load_svd_pipeline(self):
        print(f"Loading Stable Video Diffusion components from {self.video_model_id}...")
        try:
            self.svd_pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.video_model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
                variant="fp16" if self.device == "cuda" else None
            )
            if self.device == "cuda":
                # Use cpu offload to be safe even after clearing memory
                self.svd_pipeline.enable_model_cpu_offload()
            else:
                self.svd_pipeline.to(self.device)
            return True
        except Exception as e:
            print(f"Warning: SVD load failed ({e}). Video generation will not work.")
            self.svd_pipeline = None
            return False

    @torch.no_grad()
    def generate_video(self, eeg_sample, output_path="dream_video.mp4", num_inference_steps=25):
        """
        Full Pipeline: EEG -> Image -> Video
        """
        # Step 1: Generate Anchor Image (Phase 1)
        print("Generating Anchor Image from EEG...")
        # Note: self.generate returns numpy array (H, W, 3)
        image_np = self.generate(eeg_sample)
        
        # SVD expects PIL Image
        from PIL import Image
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        if image_pil.size != (1024, 576):
            image_pil = image_pil.resize((1024, 576)) # SVD preferred resolution
        
        # Step 2: Aggressive Memory Cleanup (Destroy Phase 1)
        print("Aggressively clearing Phase 1 memory to make room for SVD...")
        
        # Move everything to CPU first (just in case)
        self.vae.to("cpu")
        self.unet.to("cpu")
        self.text_encoder.to("cpu")
        self.eeg_encoder.to("cpu")
        self.dream_adapter.to("cpu")
        
        # Delete references
        del self.vae
        del self.unet
        del self.text_encoder
        del self.eeg_encoder
        del self.dream_adapter
        del self.optimizer
        
        if hasattr(self, 'scheduler'): del self.scheduler
        if hasattr(self, 'tokenizer'): del self.tokenizer
        
        # Force GC
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Memory cleared. CUDA allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Step 3: Load SVD (Lazy Load)
        if self.svd_pipeline is None:
            if not self._load_svd_pipeline():
                return None
            
        # Step 4: Generate Video (Phase 2)
        print(f"Dreaming in 4D (Generating Video, Steps={num_inference_steps})...")
        try:
            frames = self.svd_pipeline(
                image_pil, 
                decode_chunk_size=2, # Reduced chunk size to save memory
                generator=torch.manual_seed(42),
                motion_bucket_id=127, # Higher = more motion
                noise_aug_strength=0.1,
                num_inference_steps=num_inference_steps
            ).frames[0]

            # Step 5: Save
            print(f"Saving video to {output_path}...")
            export_to_video(frames, output_path, fps=7)
            return output_path
        except Exception as e:
            print(f"Error during video generation: {e}")
            return None
