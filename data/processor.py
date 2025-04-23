"""
音频特征处理器
"""

import torch
import torchaudio.transforms as T


class MFCCProcessor:
    """MFCC特征处理管道（含SpecAugment）
    
    Args:
        sample_rate (int): 采样率，默认16000Hz
        n_mfcc (int): MFCC特征维度，默认80
        win_length (int): 窗口长度，默认400 (25ms @16kHz)
        hop_length (int): 帧移大小，默认160 (10ms @16kHz)
        n_fft (int): FFT点数，默认512
        norm (str): 归一化方式，默认"ortho"
        audio_length (int): 处理音频的长度(秒)，默认为3秒
    """
    def __init__(self, 
                 sample_rate=16000,
                 n_mfcc=80,
                 win_length=400,    # 25ms @16kHz
                 hop_length=160,    # 10ms @16kHz
                 n_fft=512,
                 norm="ortho",
                 audio_length=3):
        # MFCC参数
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.win_length = win_length
        self.hop_length = hop_length
        self.audio_length = audio_length
        
        # MFCC转换器
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "n_mels": n_mfcc,
                "window_fn": torch.hamming_window,
            },
            norm=norm
        )
        
        # 滑动CMVN（仅均值归一）
        self.cmvn = T.SlidingWindowCmn(
            cmn_window=audio_length * 100,  # 秒数×100帧/秒
            min_cmn_window=100,   # 最小1秒
            norm_vars=False       # 仅均值归一化
        )
        
        # SpecAugment参数
        self.time_mask = T.TimeMasking(time_mask_param=5)  # 最大掩码5帧（50ms）
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10) # 最大掩码10通道

    def __call__(self, waveform, apply_augment=True):
        """处理流程：MFCC -> CMVN -> SpecAugment
        
        Args:
            waveform (Tensor): 波形数据，形状为(1, samples)
            apply_augment (bool): 是否应用SpecAugment增强，默认True
            
        Returns:
            Tensor: MFCC特征，形状为(n_mfcc, time)
        """
        try:
            # 检查波形是否有效
            if waveform.shape[1] == 0:
                # 返回空特征
                return torch.zeros((self.n_mfcc, 0))
                
            # 转换为MFCC
            mfcc = self.mfcc_transform(waveform)  # (1, n_mfcc, time)
            
            # 应用滑动CMVN（倒谱均值归一化）
            if mfcc.shape[2] > 0:  # 确保有时间帧
                mfcc = self.cmvn(mfcc)
            
            # 应用SpecAugment（训练时）
            if apply_augment and mfcc.shape[2] > 1:  # 确保至少有2个时间帧
                mfcc = self.time_mask(mfcc)
                mfcc = self.freq_mask(mfcc)
                
            return mfcc.squeeze(0)  # (n_mfcc, time)
        except Exception as e:
            print(f"MFCC处理错误: {e}")
            # 返回空特征
            return torch.zeros((self.n_mfcc, 0)) 