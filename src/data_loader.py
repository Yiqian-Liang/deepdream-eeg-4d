import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
import zipfile
import shutil

# 引入我们刚写的预处理模块
from src.preprocessing import clean_dream_eeg, extract_rem_features

class ThingsEEGDataset(Dataset):
    """
    Stage A: 清醒视觉预训练数据集 (THINGS-EEG)
    目标：学习 "EEG -> Image Embedding" 的映射
    """
    def __init__(self, root_dir="./data/things_eeg", download=True, dummy_fallback=True, num_samples=100):
        self.root_dir = root_dir
        self.dummy_fallback = dummy_fallback
        self.data = []
        self.labels = [] # CLIP Image Embeddings
        
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            
        # 1. 尝试加载真实数据 (简化版逻辑)
        # 实际项目中，这里会使用 osfclient 下载并解析 BIDS 格式
        # 由于环境限制，我们默认使用 Dummy Data 策略来保证代码跑通
        self.use_dummy = True 
        if self.use_dummy and self.dummy_fallback:
            print(f"Notice: Using Dummy THINGS-EEG data for prototype verification ({num_samples} samples).")
            self._generate_dummy_data(num_samples)
            
    def _generate_dummy_data(self, num_samples=100):
        """生成符合视觉 EEG 特征的模拟数据"""
        # Shape: (Batch, Channels=63, Time=250) -> 1s @ 250Hz
        # 重点模拟枕叶 (Occipital) 通道的视觉诱发反应 (VEP)
        for _ in range(num_samples):
            # 1. 基础脑电 (随机噪声)
            eeg = np.random.randn(63, 250).astype(np.float32)
            
            # 2. 注入视觉诱发特征 (在 100-200ms 处的 P100/N170 波)
            # 假设后部通道是最后几个
            visual_channels = range(50, 63) 
            t = np.linspace(0, 1, 250)
            vep_signal = np.sin(2 * np.pi * 10 * t) * np.exp(-((t - 0.15)**2) / 0.01)
            eeg[visual_channels] += vep_signal * 2.0
            
            self.data.append(eeg)
            
            # 3. 生成对应的随机 CLIP Image Embedding (768维)
            # 模拟 "看到了一张图"
            clip_embed = np.random.randn(768).astype(np.float32)
            # 归一化
            clip_embed /= np.linalg.norm(clip_embed)
            self.labels.append(clip_embed)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        eeg = self.data[idx]
        clip_embed = self.labels[idx]
        
        # 预处理
        # 这里不做 clean_dream_eeg，因为这是清醒的高信噪比数据
        # 但为了格式统一，可以做基础的标准化
        eeg = (eeg - np.mean(eeg)) / (np.std(eeg) + 1e-6)
        
        return {
            "eeg": torch.tensor(eeg, dtype=torch.float32),
            "clip_target": torch.tensor(clip_embed, dtype=torch.float32),
            "type": "image" # 标记这是图像对齐任务
        }

class DreamDataset(Dataset):
    """
    Stage B: 梦境数据微调数据集 (DREAM Database)
    目标：学习 "EEG -> Text Embedding (Dream Report)" 的映射
    来源：清华镜像 (DREAM_mini.zip)
    """
    def __init__(self, root_dir="./data/dream", download=True, use_tsinghua_mirror=True, num_samples=50):
        self.root_dir = root_dir
        self.data = []
        self.reports = [] # 文本报告
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
        # 确保数据存在
        if download:
            self._download_from_mirror(use_tsinghua_mirror)
            
        # 加载数据 (模拟逻辑，实际需解析 CSV/EDF)
        # 这里我们生成基于 DREAM 统计特性的数据，展示如何处理文本标签
        self._load_or_generate_data(num_samples)
        
    def _download_from_mirror(self, use_tsinghua):
        if os.path.exists(os.path.join(self.root_dir, "DREAM_mini")):
            print("DREAM dataset already exists.")
            return
            
        url = "https://mirrors.tuna.tsinghua.edu.cn/dream-dataset/DREAM_mini.zip" if use_tsinghua else None
        if not url:
            print("No download URL provided.")
            return
            
        print(f"Downloading DREAM dataset from {url}...")
        # 实际代码中可以使用 wget 或 requests
        # os.system(f"wget {url} -P {self.root_dir}")
        # os.system(f"unzip {self.root_dir}/DREAM_mini.zip -d {self.root_dir}")
        print("Download simulated (to save bandwidth in this demo).")
        
    def _load_or_generate_data(self, num_samples=50):
        """
        加载真实数据失败时，生成合成数据作为 Fallback
        这对应了 Qwen 建议的 SyntheticDreamDataset
        """
        print("Loading DREAM data (Fallback to Synthetic)...")
        
        dummy_reports = [
            "I saw a flying cat in the sky.",
            "I was walking in a forest made of candy.",
            "A giant robot was chasing me.",
            "I was flying over a blue ocean.",
            "There was a talking dog."
        ]
        
        for i in range(num_samples):
            # 1. 生成梦境特征 EEG (Theta/Gamma 主导)
            # Shape: (Channels, Time)
            eeg = np.random.randn(63, 250).astype(np.float32)
            
            # 清洗 & 增强 (关键步骤！)
            # 模拟 REM 睡眠的特征
            eeg = clean_dream_eeg(eeg)
            
            self.data.append(eeg)
            
            # 2. 随机分配一个梦境报告
            report = dummy_reports[i % len(dummy_reports)]
            self.reports.append(report)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        eeg = self.data[idx]
        report = self.reports[idx]
        
        # 这里的关键差异：Stage B 的 Target 是文本
        # Pipeline 需要在运行时用 Text Encoder 编码它
        # 或者我们在这里预处理成 Token ID
        
        text_inputs = self.tokenizer(
            report, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "eeg": torch.tensor(eeg, dtype=torch.float32),
            "input_ids": text_inputs.input_ids.squeeze(0), # (77,)
            "type": "text", # 标记这是文本对齐任务
            "raw_text": report
        }

def get_dataloader(dataset_name="THINGS", batch_size=32):
    if dataset_name == "THINGS":
        dataset = ThingsEEGDataset()
    elif dataset_name == "DREAM":
        dataset = DreamDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
