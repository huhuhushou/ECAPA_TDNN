"""
ECAPA-TDNN 模型相关组件定义
包含：SEBlock, Res2Block, TDNNLayer, AttentiveStatsPool, ECAPA_TDNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力机制
    通过自适应学习通道权重增强重要特征通道
    
    Args:
        channels (int): 输入特征通道数
        bottleneck (int): 压缩中间层维度，默认128
    """
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            # 全局平均池化获取通道统计信息 (batch, channels, T) -> (batch, channels, 1)
            nn.AdaptiveAvgPool1d(1),
            # 降维全连接层(降低通道数)：channels -> bottleneck
            nn.Conv1d(channels, bottleneck, kernel_size=1),
            nn.ReLU(inplace=True),  # ReLU(x) = max(0, x)
            # 升维全连接层：bottleneck -> C
            nn.Conv1d(bottleneck, channels, kernel_size=1),
            # Sigmoid生成0-1的通道权重
            nn.Sigmoid()
        )

    def forward(self, x):
        """前向传播
        Input shape: (batch, channels, time_steps)
        Output shape: 与输入相同
        """
        # 原始特征乘以学习到的通道权重（广播机制）,每个通道的所有时间步乘以相同权重
        return x * self.se(x)  # (batch, C, T) * (batch, C, 1)


class Res2Block(nn.Module):
    """多尺度残差块（Res2Net结构）
    特征分组处理实现多尺度特征提取，结合SE模块
    1. 多粒度特征学习：将输入特征分为多个子组（scale个），在不同感受野下处理
    2. 层级残差连接：相邻子组间传递信息，增强梯度流动
    3. 通道注意力增强：通过SEBlock动态调整通道权重
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        scale (int): 分块数量，控制多尺度粒度，默认8
        dilation (int): 膨胀系数，扩大感受野，默认1
    """
    def __init__(self, in_channels, out_channels, scale=8, dilation=1):
        super().__init__()
        assert in_channels % scale == 0, f"{in_channels} must be divisible by scale"
        self.scale = scale  # 添加scale属性
        
        # 划分通道到各子组
        sub_channels = in_channels // scale
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(sub_channels, sub_channels, 
                         kernel_size=3, dilation=dilation, 
                         padding=dilation),
                nn.BatchNorm1d(sub_channels),
                nn.ReLU()
            ) for _ in range(scale-1)  # 保持scale-1个卷积块
        ])
        
        # 注意力机制
        self.se = SEBlock(in_channels)
        # 输出通道调整
        self.out_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 正确分割为scale块(在通道维度上划分)
        split_size = x.size(1) // self.scale
        x_split = torch.split(x, split_size, dim=1)
        
        # 多层级处理
        out = [x_split[0]]  # 第一个分块不处理
        for i in range(1, self.scale):  # 处理后续 scale-1 个分块
            
            # 卷积处理（仅对前scale-1块）
            if i-1 < len(self.blocks):
                # 残差连接：当前块输入 = 原始分割 + 前块输出
                residual = out[i-1] if i > 1 else 0
                processed = self.blocks[i-1](x_split[i] + residual)
                out.append(processed)
            else:
                out.append(x_split[i])
        
        # 合并 & SE加权
        x = torch.cat(out, dim=1)
        x = self.se(x)
        return self.out_conv(x)


class TDNNLayer(nn.Module):
    """TDNN网络层（封装Res2Block）
    包含膨胀卷积、批归一化和激活函数
    
    Args:
        in_channels (int): 输入通道
        out_channels (int): 输出通道
        scale (int): Res2分块数
        dilation (int): 膨胀系数
    """
    def __init__(self, in_channels, out_channels, scale=8, dilation=1):
        super().__init__()
        # 核心Res2Block结构
        self.res2block = Res2Block(in_channels, out_channels, scale, dilation)
        # 批归一化
        self.bn = nn.BatchNorm1d(out_channels)
        # 激活函数
        self.relu = nn.ReLU(inplace=True) # 可用PReLU

    def forward(self, x):
        """前向传播
        输入输出形状相同：(batch, C, T)
        """
        return self.relu(self.bn(self.res2block(x)))


class AttentiveStatsPool(nn.Module):
    """注意力统计池化层
    通过注意力机制计算时间维度加权均值和标准差
    
    Args:
        in_channels (int): 输入特征通道数
        hidden_size (int): 注意力网络隐藏层维度
    """
    def __init__(self, in_channels, hidden_size=128):
        super().__init__()
        # 上下文编码层（输入拼接均值/标准差）
        self.tdnn = nn.Sequential(
            nn.Conv1d(in_channels*3, hidden_size, kernel_size=1),  # 1x1卷积
            nn.ReLU(inplace=True)
        )
        # 注意力权重生成网络
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_size, in_channels, kernel_size=1),
            nn.Softmax(dim=2)  # 时间维度softmax
        )

    def forward(self, x):
        """前向传播
        Input shape: (batch, C, T)
        Output shape: (batch, 2*C)
        """
        # 计算时间维度均值和标准差
        mean = torch.mean(x, dim=2, keepdim=True)  # (batch, C, 1)
        std = torch.std(x, dim=2, keepdim=True).clamp(min=1e-5)
        
        # 拼接上下文信息：原始特征、均值、标准差
        context = torch.cat([x, mean.expand(-1,-1,x.size(2)), std.expand(-1,-1,x.size(2))], dim=1)
        
        # 通过TDNN层编码上下文
        context = self.tdnn(context)  # (batch, hidden, T)
        
        # 生成注意力权重矩阵
        attn = self.attention(context)  # (batch, C, T)
        
        # 计算加权统计量，用得到的注意力来加权，把时间维度上的T个值转为一个单一的值
        weighted_mean = torch.sum(x * attn, dim=2)  # (batch, C, 1)
        weighted_std = torch.sqrt(
            (torch.sum(x**2 * attn, dim=2) - weighted_mean**2).clamp(min=1e-5)
        )
        
        # 拼接均值标准差作为最终特征
        return torch.cat([weighted_mean, weighted_std], dim=1) # (batch, 2C, 1)


class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN 主网络
    论文：https://arxiv.org/abs/2005.07143
    
    Args:
        input_size (int): 输入特征维度（如80维）
        channels (int): 卷积通道基数，默认512
        emb_size (int): 说话人嵌入维度，默认192
    """
    def __init__(self, input_size=80, channels=512, emb_size=192):
        super().__init__()
        # 初始卷积层（5x1卷积）
        self.conv1 = nn.Conv1d(input_size, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 三个TDNN层（不同膨胀系数）
        self.layer1 = TDNNLayer(channels, channels, dilation=2)
        self.layer2 = TDNNLayer(channels, channels, dilation=3)
        self.layer3 = TDNNLayer(channels, channels, dilation=4)
        
        # 特征聚合层（1x1卷积）
        self.conv2 = nn.Conv1d(3*channels, 3*channels, kernel_size=1)
        
        # 注意力统计池化
        self.pooling = AttentiveStatsPool(3*channels)
        
        # 池化后处理
        self.bn2 = nn.BatchNorm1d(6*channels)  # 均值+标准差维度为2C
        
        # 全连接层生成嵌入
        self.fc = nn.Linear(6*channels, emb_size)
        self.bn3 = nn.BatchNorm1d(emb_size)  # 最终L2正则化

    def forward(self, x):
        """前向传播
        Input shape: (batch, feat_dim, time_steps)
        Output shape: (batch, emb_size)
        """
        # 初始卷积处理
        x = self.relu(self.bn1(self.conv1(x)))  # (B, C, T)
        
        # 通过三个TDNN层（含残差连接）
        x1 = self.layer1(x)                # 第一层输出
        x2 = self.layer2(x + x1)           # 第二层（含残差）
        x3 = self.layer3(x + x1 + x2)      # 第三层（累加残差）
        
        # 拼接多层级特征
        x = torch.cat([x1, x2, x3], dim=1) # (B, 3C, T)
        x = self.relu(self.conv2(x))       # (B, 3C, T)
        
        # 注意力统计池化
        x = self.pooling(x)                # (B, 6C)
        x = self.bn2(x)
        
        # 全连接层生成嵌入
        x = self.fc(x)                     # (B, emb_size)
        return self.bn3(x)                 # 标准化后的嵌入向量 