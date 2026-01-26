import numpy as np
import torch
import mne
from scipy.signal import welch

def clean_dream_eeg(eeg_data, sfreq=250):
    """
    对梦境 EEG 进行基础清洗：
    1. 简单去噪 (Bandpass filter 0.5 - 40 Hz)
    2. 幅值截断 (去除极端的运动/眼动伪影)
    
    Args:
        eeg_data: (Channels, Time) numpy array
        sfreq: Sampling frequency
    Returns:
        cleaned_eeg: (Channels, Time) numpy array
    """
    # 1. 简单的带通滤波 (模拟)
    # 注意：在实际 DataLoader 中通常使用 mne.filter，这里为了简单起见，
    # 如果输入是 numpy，我们假设外部已经做过基础滤波，或者这是一个轻量级的处理
    # 这里主要做幅度截断
    
    # 阈值截断 (Clip outliers)
    # 梦境 EEG 振幅通常较低，超过 100uV 通常是伪影
    threshold = 100.0  # uV
    cleaned_eeg = np.clip(eeg_data, -threshold, threshold)
    
    # 归一化 (Z-score per channel)
    # 保持数值稳定性，对深度学习模型很重要
    mean = np.mean(cleaned_eeg, axis=1, keepdims=True)
    std = np.std(cleaned_eeg, axis=1, keepdims=True)
    cleaned_eeg = (cleaned_eeg - mean) / (std + 1e-6)
    
    return cleaned_eeg

def extract_rem_features(eeg_data, sfreq=250):
    """
    提取 REM 睡眠相关的关键频段特征 (Theta & Gamma)
    作为原始波形的补充输入。
    
    Args:
        eeg_data: (Channels, Time) numpy array
    Returns:
        features: (Channels, 2) numpy array (Theta power, Gamma power)
    """
    # 使用 Welch 方法计算功率谱密度 (PSD)
    nperseg = min(eeg_data.shape[1], int(2 * sfreq)) # 2秒窗口
    freqs, psd = welch(eeg_data, fs=sfreq, nperseg=nperseg, axis=1)
    
    # 定义频段
    # Theta: 4-7 Hz (REM 睡眠特征波)
    # Gamma: 30-50 Hz (梦境视觉/意识相关)
    theta_idx = np.where((freqs >= 4) & (freqs <= 7))[0]
    gamma_idx = np.where((freqs >= 30) & (freqs <= 50))[0]
    
    # 计算平均功率
    theta_power = np.mean(psd[:, theta_idx], axis=1)
    gamma_power = np.mean(psd[:, gamma_idx], axis=1)
    
    # 堆叠特征
    # Shape: (Channels, 2)
    features = np.stack([theta_power, gamma_power], axis=1)
    
    # 对数变换 (Log power) 使分布更正态
    features = np.log1p(features)
    
    return torch.tensor(features, dtype=torch.float32)

def simulate_rem_bursts(eeg_data, sfreq=250):
    """
    数据增强：在清醒数据中模拟 REM 睡眠的特征 (Theta/Gamma Bursts)
    用于 Stage A -> Stage B 的过渡训练
    """
    n_channels, n_time = eeg_data.shape
    time = np.arange(n_time) / sfreq
    
    # 1. 注入 Theta 波 (4-7 Hz)
    theta_freq = np.random.uniform(4, 7)
    theta_wave = np.sin(2 * np.pi * theta_freq * time)
    # 随机振幅调制
    theta_amp = np.random.uniform(0.5, 1.5, size=(n_channels, 1))
    
    # 2. 注入 Gamma Bursts (30-50 Hz)
    gamma_freq = np.random.uniform(30, 50)
    gamma_wave = np.sin(2 * np.pi * gamma_freq * time)
    # Gamma 通常是短暂爆发，用高斯窗调制
    burst_center = np.random.uniform(0, n_time/sfreq)
    burst_width = 0.1 # seconds
    gamma_envelope = np.exp(-0.5 * ((time - burst_center) / burst_width)**2)
    
    # 混合
    augmented_eeg = eeg_data + 0.2 * theta_wave * theta_amp + 0.1 * gamma_wave * gamma_envelope
    
    return augmented_eeg
