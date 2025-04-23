"""
分类器模型定义
"""

import torch
import torch.nn as nn
from .ecapa_tdnn import ECAPA_TDNN


class AudioClassifier(ECAPA_TDNN):
    """
    音频欺骗检测分类器，基于ECAPA-TDNN模型
    
    Args:
        input_size (int): 输入特征维度，默认80
        channels (int): 卷积通道数，默认512
        emb_size (int): 嵌入特征维度，默认192
    """
    def __init__(self, input_size=80, channels=512, emb_size=192):
        super().__init__(input_size, channels, emb_size)
        # 添加二分类输出层
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, 128),    # 扩展隐藏层
            nn.PReLU(),                  # 参数化ReLU
            nn.Dropout(0.2),             # 增强正则化
            nn.Linear(128, 2)            # 二分类输出
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch, input_size, time_steps)
        
        Returns:
            logits: 分类logits (batch, 2)
        """
        # 获取ECAPA嵌入特征
        emb = super().forward(x)  # (batch, emb_size)
        # 分类决策
        return self.classifier(emb)  # (batch, 2) 