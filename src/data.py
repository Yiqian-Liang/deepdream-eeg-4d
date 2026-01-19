import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from transformers import CLIPTokenizer, CLIPTextModel

class RealEEGDataset(Dataset):
    """
    真实 EEG 数据集 (BNCI2014001 via MOABB)
    映射: 运动想象类别 -> CLIP 文本 Embedding
    """
    def __init__(self, subject_id=1, time_steps=1001):
        print(f"Initializing RealEEGDataset for Subject {subject_id}...")
        
        # 1. Load Data using MOABB
        dataset = BNCI2014001()
        # fmin/fmax: typical motor imagery bands (4-38 Hz)
        paradigm = MotorImagery(fmin=4, fmax=38, n_classes=4, resample=None)
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
        
        # X shape: (Trials, Channels, Time) e.g., (576, 22, 1001)
        # y: labels ['left_hand', 'right_hand', 'feet', 'tongue']
        
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = y
        self.time_steps = X.shape[2]
        self.channels = X.shape[1]
        
        print(f"Loaded {len(self.X)} trials. EEG Shape: {self.X.shape}")
        
        # 2. Precompute CLIP Embeddings for labels
        print("Precomputing CLIP Text Embeddings for labels...")
        self.class_embeddings = self._precompute_class_embeddings()
        
    def _precompute_class_embeddings(self):
        # Semantic Mapping: Label -> Prompt
        label_map = {
            'left_hand': "A photo of a left hand",
            'right_hand': "A photo of a right hand",
            'feet': "A photo of human feet",
            'tongue': "A photo of a human tongue"
        }
        
        tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
        
        embeddings = {}
        for label, prompt in label_map.items():
            inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                # Get the pooled output (768,) or last hidden state
                # Here we use pooled_output which corresponds to the EOS token for global sentence representation
                # However, SD usually uses the last_hidden_state (B, 77, 768).
                # Our DreamAdapter maps EEG (B, 768) -> (B, 77, 768).
                # To align with our architecture (EEGEncoder outputs 1 vector), we should probably use the POOLED embedding as target
                # OR change the pipeline.
                
                # Let's verify what EEGEncoder outputs: (B, 768).
                # DreamAdapter inputs (B, 768) and outputs (B, 77, 768).
                # So the EEGEncoder is trying to learn the "Sentence Vector".
                # But for the Loss, we need to compare apples to apples.
                # If we train EEGEncoder -> DreamAdapter -> (77, 768) vs CLIP Text Encoder -> (77, 768), that works.
                # But simpler MVP: Align EEGEncoder output (768) with CLIP Pooled Output (768).
                
                outputs = text_encoder(**inputs)
                pooled_output = outputs.pooler_output # (1, 768)
                embeddings[label] = pooled_output.squeeze(0) # (768,)
                
        return embeddings

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        eeg_data = self.X[idx] # (Channels, Time)
        label = self.y[idx]
        
        # Get corresponding CLIP embedding
        clip_embed = self.class_embeddings[label] # (768,)
        
        return {
            "eeg": eeg_data,
            "clip_embed": clip_embed
        }

def get_dataloader(batch_size=8, use_real_data=True):
    # Enforce real data usage
    try:
        dataset = RealEEGDataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load Real EEG Dataset. {e}")
        raise e
