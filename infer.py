#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
声纹伪造检测模型推理脚本
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import soundfile
import soxr
import numpy as np

# 添加当前目录到Python路径，确保可以正确导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AudioClassifier
from data import MFCCProcessor


def load_model(model_path, device):
    """
    加载模型
    
    Args:
        model_path: 模型路径
        device: 计算设备
        
    Returns:
        模型实例
    """
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 初始化模型
    model = AudioClassifier(input_size=80, channels=512, emb_size=192).to(device)
    
    # 直接加载模型状态字典
    model.load_state_dict(torch.load(model_path))
    print(f"从文件加载模型权重: {model_path}")
    
    model.eval()
    return model


def preprocess_audio(audio_path, processor, audio_length=3):
    """
    预处理音频文件
    
    Args:
        audio_path: 音频文件路径
        processor: MFCC处理器
        audio_length: 处理音频的长度(秒)，默认为3秒
        
    Returns:
        处理后的特征
    """
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    # 读取音频
    waveform, sr = soundfile.read(audio_path, dtype='float32')
    
    # 确保波形不为空
    if waveform.size == 0:
        raise ValueError(f"音频文件为空: {audio_path}")
    
    # 重采样至目标采样率
    if sr != processor.sample_rate:
        waveform = soxr.resample(waveform, sr, processor.sample_rate)
    
    # 转换为tensor
    waveform = torch.from_numpy(waveform).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)
    else:
        # 如果是多通道，取平均
        waveform = waveform.mean(axis=1, keepdim=True)
    
    # 生成MFCC特征
    mfcc = processor(waveform, apply_augment=False)  # (n_mfcc, time)
    
    # 调整特征长度
    target_frames = audio_length * 100  # 帧数 (100帧/秒)
    
    # 如果特征太长,取中间部分
    if mfcc.shape[1] > target_frames:
        start = (mfcc.shape[1] - target_frames) // 2
        mfcc = mfcc[:, start:start+target_frames]
    # 如果特征太短,重复填充
    elif mfcc.shape[1] < target_frames:
        repeats = target_frames // mfcc.shape[1] + 1
        mfcc = mfcc.repeat(1, repeats)[:, :target_frames]
    
    # 添加批次维度
    mfcc = mfcc.unsqueeze(0)  # (1, n_mfcc, time)
    
    return mfcc


def predict(model, features, device):
    """
    使用模型进行预测
    
    Args:
        model: 模型实例
        features: 输入特征
        device: 计算设备
        
    Returns:
        预测结果和概率
    """
    features = features.to(device)
    
    with torch.no_grad():
        outputs = model(features)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        fake_prob = probs[0, 1].item()
    
    return pred_class, fake_prob


def main(args):
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 
                             'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 直接使用指定的模型路径
    model_path = args.model_path
    print(f"使用模型: {model_path}")
    
    # 加载模型
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 初始化MFCC处理器
    processor = MFCCProcessor(sample_rate=args.sample_rate, audio_length=args.audio_length)
    
    # 处理单个文件还是目录
    if os.path.isfile(args.input):
        # 处理单个文件
        try:
            audio_paths = [args.input]
        except Exception as e:
            print(f"处理音频文件失败: {e}")
            return
    elif os.path.isdir(args.input):
        # 处理目录
        audio_paths = []
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.endswith('.wav'):
                    audio_paths.append(os.path.join(root, file))
        
        if not audio_paths:
            print(f"目录中没有找到WAV文件: {args.input}")
            return
        
        print(f"找到 {len(audio_paths)} 个WAV文件")
    else:
        print(f"输入路径不存在: {args.input}")
        return
    
    # 处理所有音频文件
    results = []
    for audio_path in audio_paths:
        try:
            # 预处理音频
            features = preprocess_audio(audio_path, processor, args.audio_length)
            
            # 预测
            pred_class, fake_prob = predict(model, features, device)
            
            # 解释结果
            if pred_class == 0:
                result = "真人"
                confidence = 1 - fake_prob
            else:
                result = "虚假"
                confidence = fake_prob
            
            # 保存结果
            results.append({
                'file': audio_path,
                'result': result,
                'confidence': confidence,
                'fake_prob': fake_prob
            })
            
            # 打印结果
            print(f"文件: {os.path.basename(audio_path)}")
            print(f"鉴别结果: {result}")
            print(f"置信度: {confidence*100:.2f}%")
            print()
            
        except Exception as e:
            print(f"处理文件 {audio_path} 时出错: {e}")
            print()
    
    # 打印汇总信息
    if len(results) > 1:
        real_count = sum(1 for r in results if r['result'] == '真人')
        fake_count = sum(1 for r in results if r['result'] == '虚假')
        print(f"\n汇总结果: 共 {len(results)} 个文件，真人: {real_count}，虚假: {fake_count}")
    
    # 如果指定了输出文件，保存结果
    if args.output:
        import csv
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['文件', '结果', '置信度', '虚假概率'])
            for r in results:
                writer.writerow([r['file'], r['result'], f"{r['confidence']*100:.2f}%", f"{r['fake_prob']:.4f}"])
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="语音伪造检测模型推理脚本")
    
    parser.add_argument("input", type=str, 
                        help="输入音频文件或目录")
    parser.add_argument("--model_path", type=str, 
                        default="pretrained_models/run_20250423_170726/best_model.pth",
                        help="模型文件路径")
    parser.add_argument("--output", type=str, default="",
                        help="输出CSV文件路径")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="音频采样率")
    parser.add_argument("--audio_length", type=int, default=3,
                        help="处理音频的长度(秒)")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备 (auto, cuda, cpu, mps)")
    
    args = parser.parse_args()
    
    main(args) 