import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import os

from src.models import EEGEncoder, DreamAdapter

class EEG2DreamPipeline:
    """
    EEG2Dream 端到端管道
    
    包含:
    1. 模型初始化 (SD, EEGEncoder, Adapter)
    2. 训练逻辑 (Alignment & Diffusion)
    3. 生成逻辑 (EEG -> Image)
    """
    def __init__(self, 
                 model_id="runwayml/stable-diffusion-v1-5", 
                 device="cuda" if torch.cuda.is_available() else "cpu", 
                 debug=False,
                 num_eeg_channels=32,
                 eeg_time_steps=512):
        self.device = device
        self.model_id = model_id
        
        print(f"Loading Stable Diffusion components from {model_id}...")
        
        if debug:
            print("DEBUG MODE: Skipping actual model loading, using mock layers.")
            self.vae = self._get_mock_vae()
            self.unet = self._get_mock_unet()
            self.scheduler = DDPMScheduler()
            self.text_encoder = nn.Linear(1, 1).to(device)
            self.tokenizer = None
        else:
            # 1. 加载冻结的 SD 组件
            # Strict mode: Do not fallback to mock if loading fails
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
            self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
            self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)

        if not debug and hasattr(self.vae, 'requires_grad_'):
            # 冻结参数
            self.vae.requires_grad_(False)
            self.unet.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
        
        # 2. 初始化可训练模块
        print(f"Initializing EEG Encoder (Channels={num_eeg_channels}, Time={eeg_time_steps}) & Dream Adapter...")
        self.eeg_encoder = EEGEncoder(num_channels=num_eeg_channels, time_steps=eeg_time_steps).to(device)
        self.dream_adapter = DreamAdapter().to(device)
        
        # 3. 优化器
        self.optimizer = torch.optim.AdamW(
            list(self.eeg_encoder.parameters()) + list(self.dream_adapter.parameters()),
            lr=1e-4
        )
    
    def _get_mock_unet(self):
        # Create a mock object that returns a .sample attribute with random tensor
        class MockUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('obj', (object,), {'in_channels': 4})
                self.dtype = torch.float32
            def forward(self, sample, timestep, encoder_hidden_states=None, **kwargs):
                return type('obj', (object,), {'sample': torch.randn_like(sample)})
        return MockUNet()

    def _get_mock_vae(self):
        class MockVAE(nn.Module):
            def __init__(self):
                super().__init__()
            def decode(self, latents):
                # Return a mock object with .sample
                # Image shape: (B, 3, H, W). Latents: (B, 4, H/8, W/8)
                B, C, H, W = latents.shape
                return type('obj', (object,), {'sample': torch.randn(B, 3, H*8, W*8)})
        return MockVAE()

        
    def train(self, dataloader, num_epochs=1):
        """
        训练循环
        
        Phase 1 重点:
        使用 CosineSimilarityLoss 对齐 EEG Embedding 和 CLIP Image Embedding。
        """
        self.eeg_encoder.train()
        self.dream_adapter.train()
        
        loss_fn = nn.CosineEmbeddingLoss()
        
        for epoch in range(num_epochs):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                eeg = batch['eeg'].to(self.device) # (B, Channels, Time)
                clip_target = batch['clip_embed'].to(self.device) # (B, 768)
                
                # Forward
                eeg_features = self.eeg_encoder(eeg) # (B, 768)
                
                # Loss Calculation
                # Target is 1.0 (maximize similarity)
                target = torch.ones(eeg.shape[0]).to(self.device)
                loss = loss_fn(eeg_features, clip_target, target)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                progress_bar.set_postfix({"loss": loss.item()})
                
        print("Training finished.")

    @torch.no_grad()
    def generate(self, eeg_sample, num_inference_steps=20):
        """
        生成梦境图像
        
        Args:
            eeg_sample: (Channels, Time)
        """
        self.eeg_encoder.eval()
        self.dream_adapter.eval()
        
        # 1. Prepare Condition
        eeg_input = eeg_sample.unsqueeze(0).to(self.device) # (1, C, T)
        eeg_embedding = self.eeg_encoder(eeg_input) # (1, 768)
        
        # Adapter: (1, 768) -> (1, 77, 768)
        cond_embeddings = self.dream_adapter(eeg_embedding)
        
        # Unconditional embeddings (for classifier-free guidance)
        # Usually we use empty string "" for uncond
        # Here we just use zero tensor or similar noise for simplicity in MVP
        uncond_embeddings = torch.zeros_like(cond_embeddings)
        
        # Concatenate
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        
        # 2. Initialize Latents
        height, width = 512, 512
        latents = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8),
            device=self.device,
            dtype=self.unet.dtype
        )
        
        # 3. Denoising Loop
        self.scheduler.set_timesteps(num_inference_steps)
        print("Denoising latents with EEG conditioning...")
        
        for t in self.scheduler.timesteps:
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            
            # Predict noise
            noise_pred = self.unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=text_embeddings
            ).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guidance_scale = 7.5
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
        # 4. Decode
        image = self.vae.decode(latents).sample
        
        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        
        return image[0] # (H, W, 3)
