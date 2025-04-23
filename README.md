# ECAPA-TDNN 音频欺骗检测系统

基于ECAPA-TDNN架构的音频欺骗检测系统，专为区分真实人声和AI合成/伪造语音而设计。在使用了四个先进的TTS开源模型创建的开源数据集 [AudioSpoof](https://github.com/huhuhushou/AudioSpoof) 上进行了训练和测试，取得了较好的效果; 同时，模型较为轻量，且所需有效音频仅为3秒，便于实时监测; 提供了预训练模型于 [huggingface](https://huggingface.co/HuShou-ZMZN/ECAPA_TDNN) 。

## 项目介绍

随着AI语音合成技术的快速发展，区分真实人声和人工合成语音变得越来越困难。本项目基于ECAPA-TDNN（强调通道注意力、传播和聚合的时延神经网络）架构，开发了一个高效、准确的音频欺骗检测系统，可以有效识别当前主流的语音合成技术生成的伪造语音。

本系统可应用于：
- 语音验证系统的安全防护
- 音频内容审核
- 数字取证和真实性鉴别
- 声纹识别系统的防欺骗层

## 结果展示
在开源数据集 [AudioSpoof](https://github.com/huhuhushou/AudioSpoof) 上训练和预测的结果如下：

测试集评估结果:
- 准确率(ACC): 92.76%
- 精确率(P): 92.68%
- 召回率(R): 98.75%
- F1分数: 95.62%
- 等错误率(EER): 8.89%

## 模型架构

### 1. 基础模型 (ecapa_tdnn.py)

ECAPA-TDNN实现了一个强大的音频特征提取网络，包含以下关键组件：
- **SEBlock**：Squeeze-and-Excitation通道注意力机制，动态调整特征通道的重要性
- **Res2Block**：多尺度残差连接结构，在不同感受野下提取特征
- **TDNNLayer**：时延神经网络层，有效捕捉语音信号的时序关系
- **AttentiveStatsPool**：注意力统计池化，通过注意力机制聚合时间维度信息
- **ECAPA_TDNN**：整体架构，生成高质量的音频嵌入向量

### 2. 分类器模型 (classifier.py)

`AudioClassifier`类继承自`ECAPA_TDNN`，添加了用于欺骗检测的分类层：
- 输入特征默认维度为80
- 卷积通道数默认为512
- 嵌入特征维度默认为192
- 分类器包含一个隐藏层(192→128)和输出层(128→2)
- 使用PReLU激活函数和Dropout(0.2)正则化

### 3. 处理流程

1. 音频预处理：将原始音频转换为MFCC特征 (shape: batch, input_size, time_steps)
2. 特征提取：ECAPA-TDNN提取音频嵌入特征 (shape: batch, emb_size)
3. 分类决策：分类器将嵌入特征映射为二分类结果 (shape: batch, 2)

## 功能特点

- **高精度检测**：对多种语音合成技术（如SparkTTS、CosyVoice、F5TTS、NaturalSpeech3等）均有较高的识别率
- **鲁棒性增强**：采用特征增强和归一化技术，提高模型对噪声和通道变化的鲁棒性
- **灵活配置**：支持多种参数配置，适应不同应用场景
- **完整工具链**：提供从数据处理、模型训练到推理的完整流程
- **快速部署**：支持命令行操作和Python API调用，方便集成到现有系统
- **全面评估**：提供多种评估指标（EER、准确率、精确率、召回率等）

## 环境要求

- Python >= 3.7
- PyTorch >= 1.8.0
- torchaudio >= 0.8.0
- numpy >= 1.19.0
- scipy >= 1.6.0
- scikit-learn >= 0.24.0
- soundfile >= 0.10.0
- soxr >= 0.1.0
- tqdm >= 4.50.0

## 安装方法

### 从源码安装

```bash
git clone https://github.com/username/ECAPA_TDNN.git
```


## 快速开始

### 模型训练

#### 使用命令行

```python
# 方法1：使用python直接运行脚本
python train.py --data_dir /path/to/dataset --output_dir /path/to/save/model --epochs 50

# 方法2：在Python代码中调用
from train import main
import argparse

# 创建参数对象
args = argparse.Namespace(
    data_dir="/path/to/dataset",
    output_dir="/path/to/save/model",
    epochs=50,
    batch_size=16,
    lr=1e-4,
    channels=512,
    emb_size=192,
    device="auto",
    val_ratio=0.1,
    sample_rate=16000,
    audio_length=3,  # 处理3秒的音频
    spoof_models="sparktts,cosyvoice",  # 可选，指定欺骗模型
    weight_decay=1e-5,
    num_workers=4,
    seed=42
)

main(args)
```

#### 训练参数说明

| 参数           | 描述                           | 默认值                                  |
| -------------- | ------------------------------ | --------------------------------------- |
| --data_dir     | 数据集根目录                   | -                                       |
| --output_dir   | 模型保存目录                   | models/                                 |
| --batch_size   | 批次大小                       | 16                                      |
| --epochs       | 训练轮数                       | 50                                      |
| --lr           | 学习率                         | 1e-4                                    |
| --channels     | ECAPA-TDNN通道数               | 512                                     |
| --emb_size     | 嵌入特征维度                   | 192                                     |
| --device       | 计算设备(auto, cuda, cpu, mps) | auto                                    |
| --val_ratio    | 验证集比例                     | 0.1                                     |
| --audio_length | 处理音频的长度(秒)             | 3                                       |
| --sample_rate  | 音频采样率                     | 16000                                   |
| --spoof_models | 欺骗模型列表，逗号分隔         | sparktts,cosyvoice,f5tts,naturalspeech3 |

### 模型推理

#### 使用命令行

```python
# 方法1：使用python直接运行脚本处理单个文件
python infer.py /path/to/audio.wav --model_path /path/to/model.pth

# 方法2：在Python代码中调用
import argparse
from infer import main

# 创建参数对象
args = argparse.Namespace(
    input="/path/to/audio.wav",  # 输入音频文件或目录
    model_path="/path/to/model.pth",
    output="results.csv",  # 可选的输出CSV文件
    sample_rate=16000,
    audio_length=3,  # 处理3秒的音频
    device="auto"
)

main(args)
```

#### 推理参数说明

| 参数           | 描述                           | 默认值         |
| -------------- | ------------------------------ | -------------- |
| input          | 输入音频文件或目录             | -              |
| --model_path   | 模型文件路径                   | pretrained_models/run/best_model.pth |
| --output       | 输出CSV文件路径                | -              |
| --audio_length | 处理音频的长度(秒)             | 3              |
| --sample_rate  | 音频采样率                     | 16000          |
| --device       | 计算设备(auto, cuda, cpu, mps) | auto           |

## 项目结构

```
/
├── models/              # 模型定义
│   ├── __init__.py      
│   ├── ecapa_tdnn.py    # ECAPA-TDNN模型定义
│   └── classifier.py    # 分类器定义
├── data/                # 数据处理
│   ├── __init__.py
│   ├── processor.py     # MFCC特征处理
│   └── dataset.py       # 数据集类定义
├── utils/               # 工具函数
│   ├── __init__.py
│   ├── metrics.py       # 评估指标
│   ├── io.py            # 文件IO操作
│   ├── mp3-wav.ipynb    # MP3转WAV转换工具
│   └── vedio2wav.ipynb  # 视频提取音频工具
├── pretrained_models/   # 预训练模型目录
├── train.py             # 训练脚本
├── infer.py             # 推理脚本
├── __init__.py          # 包初始化文件
└── README.md            # 项目说明
```

## 工具脚本

### 音频格式转换 (mp3-wav.ipynb)

该Jupyter笔记本提供了将MP3文件批量转换为WAV格式的功能：
- 支持单个文件或整个目录的转换
- 保持目录结构
- 自动处理采样率转换
- 进度条显示处理状态

使用方法：
1. 在笔记本中设置输入目录和输出目录
2. 运行所有单元格
3. 转换完成后会显示成功转换的文件数量和失败的文件列表

### 视频提取音频 (vedio2wav.ipynb)

该Jupyter笔记本用于从视频文件中提取音频并转换为WAV格式：
- 支持多种视频格式（mp4, avi, mkv等）
- 批量处理功能
- 可设置输出音频的采样率和声道数
- 自动处理文件名冲突

使用方法：
1. 在笔记本中设置视频源目录和音频输出目录
2. 配置采样率和声道数（默认16kHz, 单声道）
3. 运行所有单元格
4. 处理完成后会生成提取报告

## 数据集格式

训练数据集应按以下格式组织：

```
dataset/
├── wav/
│   ├── train/               # 真实语音训练集
│   ├── dev/                 # 真实语音开发集
│   ├── test/                # 真实语音测试集
│   ├── train-sparktts/      # SparkTTS生成的虚假语音
│   ├── train-cosyvoice/     # CosyVoice生成的虚假语音
│   ├── train-f5tts/         # F5TTS生成的虚假语音
│   ├── train-naturalspeech3/# NaturalSpeech3生成的虚假语音
│   └── ... (类似的dev和test目录)
```
**注意：dataset 可以更换为任意名字，下面的文件夹必须严格按照要求命名，wav 文件夹必须存在**

系统会自动处理以下内容：
- 收集所有真实和虚假音频
- 过滤无效文件
- 按指定比例划分训练集和验证集
- 支持所有主流音频格式（自动重采样）

### 添加新的欺骗模型

如需添加新的欺骗模型，您可以：

1. 创建相应的数据目录：`train-[新模型名]`、`dev-[新模型名]`和`test-[新模型名]`
2. 在训练时通过`--spoof_models`参数指定要包含的模型数据集：
   ```bash
   python train.py --data_dir /path/to/dataset --spoof_models sparktts,cosyvoice,f5tts,naturalspeech3,新模型名
   ```

系统会自动检测目录是否存在，如果不存在会给出警告但不会中断训练。

## 评估指标

系统使用以下指标评估模型性能：

- **等错误率(EER)**: 评估二分类器在假阳率等于假阴率时的性能
- **准确率(ACC)**: 正确分类的样本比例
- **精确率(Precision)**: 在预测为虚假的样本中，真正虚假的比例
- **召回率(Recall)**: 被正确识别的虚假样本占所有虚假样本的比例
- **F1分数**: 精确率和召回率的调和平均值
- **AUC**: ROC曲线下面积，反映分类器的整体性能

## 开发者指南

### 扩展模型

如需添加新的特征提取器或分类器，可以继承现有类并重写相关方法：

```python
from models import ECAPA_TDNN

class MyCustomClassifier(ECAPA_TDNN):
    def __init__(self, input_size=80, channels=512, emb_size=192):
        super().__init__(input_size, channels, emb_size)
        # 添加自定义层...
    
    def forward(self, x):
        # 自定义前向传播逻辑...
        pass
```

### 自定义数据预处理

可以通过继承MFCCProcessor类来实现自定义的特征提取：

```python
from data import MFCCProcessor

class MyProcessor(MFCCProcessor):
    def __init__(self, sample_rate=16000, n_mfcc=80):
        super().__init__(sample_rate, n_mfcc)
        # 添加自定义处理步骤...
    
    def __call__(self, waveform, apply_augment=True):
        # 自定义特征提取流程...
        pass
```


