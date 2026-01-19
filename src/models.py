import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualAttentionBlock(nn.Module):
    """
    残差注意力模块 (Residual Attention Block)
    
    物理意义:
    脑电信号中包含大量背景噪声（如肌肉活动、眼动）。此模块通过注意力机制，
    自动加权重要的时序特征，抑制无关的背景噪声，类似于大脑在处理信息时的“聚焦”机制。
    """
    def __init__(self, channels):
        super().__init__()
        # 这里的卷积用于计算注意力权重
        self.attention_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        # 计算注意力图 (Attention Map)
        weights = self.attention_conv(x)
        # 残差连接：原始信号 + 加权后的信号
        # 物理意义：保留原始信息流的同时，增强关键特征
        return x + (x * weights)

class SubspaceProjection(nn.Module):
    """
    子空间投影层 (Subspace Projection / PCA-like Layer)
    
    物理意义:
    EEG信号是高维且冗余的。根据 BNN 论文思路，我们假设有效的神经编码位于一个低维流形上。
    此层强制模型将高维 EEG 信号投影到低维主成分空间，过滤掉非系统性的高频噪声。
    
    在实际训练前，可以用 sklearn.decomposition.PCA 计算训练集的变换矩阵，
    并初始化此层的权重，以加速收敛。
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):
        # x shape: (Batch, Features)
        return self.projection(x)

class EEGEncoder(nn.Module):
    """
    EEG 编码器 (EEG Encoder)
    
    功能: 提取时序 EEG 信号的深层语义特征。
    架构: 1D-CNN (时序特征) -> Attention (特征增强) -> Global Pooling -> Subspace Projection (去噪)
    """
    def __init__(self, num_channels=32, time_steps=512, feature_dim=1024, projection_dim=768):
        super().__init__()
        
        self.num_channels = num_channels
        
        # 1. 1D-CNN Feature Extraction
        # 物理意义：使用不同尺寸的卷积核捕捉不同频率的脑波震荡 (Alpha, Beta, Gamma 等)
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=15, stride=2, padding=7), # 捕捉慢波
            nn.GroupNorm(8, 64),
            nn.GELU(),
            
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4), # 捕捉中频波
            nn.GroupNorm(16, 128),
            nn.GELU(),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), # 捕捉快波
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )
        
        # 2. Residual Attention
        self.attention = ResidualAttentionBlock(256)
        
        # 3. Flatten & Projection
        # Use Adaptive Pooling to handle variable time lengths and reduce dimension
        # Target time dimension: 64 (preserves some temporal info but compresses)
        self.pool = nn.AdaptiveAvgPool1d(64)
        self.flatten_dim = 256 * 64
        
        # 4. Subspace Projection (PCA-like)
        # 将展平后的特征投影到潜在语义空间
        self.subspace_proj = SubspaceProjection(self.flatten_dim, projection_dim)
        
    def forward(self, x):
        # x shape: (Batch, Channels, Time) e.g. (B, 32, 512)
        
        # 提取时序特征
        feat = self.conv_blocks(x) # -> (B, 256, Time')
        
        # 注意力增强
        feat = self.attention(feat)
        
        # Adaptive Pooling (Fixes Time Dimension)
        feat = self.pool(feat) # -> (B, 256, 64)
        
        # 展平
        feat = feat.flatten(1) # -> (B, 256*64)
        
        # 子空间投影 (去噪 & 压缩)
        embedding = self.subspace_proj(feat) # -> (B, Projection_Dim)
        
        return embedding

class DreamAdapter(nn.Module):
    """
    梦境适配器 (Dream Adapter)
    
    功能: 将 EEG 语义向量映射到 Stable Diffusion 理解的 CLIP 文本嵌入空间。
    目标形状: (Batch, 77, 768) - 这是 CLIP Text Encoder 的输出形状。
    """
    def __init__(self, input_dim=768, seq_len=77, embed_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # 使用多层感知机 (MLP) 进行非线性映射
        # 物理意义：脑信号语义空间与视觉语义空间存在“模态鸿沟 (Modality Gap)”。
        # 需要通过非线性变换将 EEG 表征对齐到 CLIP 的流形上。
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, seq_len * embed_dim) # 扩展到序列长度
        )
        
        # 最后的 LayerNorm 保证数值稳定性，匹配 CLIP 的分布
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (Batch, Input_Dim) (来自 EEGEncoder)
        
        # 映射并重塑
        x = self.mapping(x) # -> (Batch, 77 * 768)
        x = x.view(-1, self.seq_len, self.embed_dim) # -> (Batch, 77, 768)
        
        x = self.norm(x)
        
        return x
