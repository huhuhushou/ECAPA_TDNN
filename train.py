#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
声纹伪造检测模型训练脚本
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import time
import json
from datetime import datetime

# 添加当前目录到Python路径，确保可以正确导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import AudioClassifier
from data import MFCCProcessor, MFCCDataset, prepare_dataset
from utils import compute_eer, compute_metrics


def set_seed(seed):
    """设置随机种子，保证实验可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        
    Returns:
        dict: 训练指标
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="训练", leave=False)
    for specs, labels in pbar:
        specs = specs.to(device)
        labels = labels.long().to(device)
        
        # 前向传播
        outputs = model(specs)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        _, predicted = torch.max(outputs, 1)
        batch_correct = (predicted == labels).sum().item()
        batch_size = labels.size(0)
        
        correct += batch_correct
        total += batch_size
        total_loss += loss.item() * batch_size
        
        # 更新进度条
        pbar.set_postfix({
            "损失": f"{loss.item():.4f}",
            "准确率": f"{100 * batch_correct / batch_size:.1f}%"
        })
    
    return {
        "loss": total_loss / total,
        "accuracy": 100 * correct / total
    }


def evaluate(model, dataloader, criterion, device):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        
    Returns:
        dict: 评估指标
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for specs, labels in tqdm(dataloader, desc="评估", leave=False):
            specs = specs.to(device)
            labels = labels.long().to(device)
            
            # 前向传播
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            # 统计
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 取第1类（虚假）的概率
            
            total_loss += loss.item() * labels.size(0)
    
    # 计算各项指标
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / len(dataloader.dataset)
    
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """保存模型状态字典"""
    # 仅保存模型状态字典，不保存优化器状态和其他信息
    torch.save(model.state_dict(), filename)
    print(f"模型权重已保存到: {filename}")


def main(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建一个唯一的模型保存目录，基于当前时间
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    if args.run_name:
        run_name = f"{args.run_name}_{timestamp}"
    
    # 完整的输出目录路径
    full_output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(full_output_dir, exist_ok=True)
    print(f"模型和日志将保存在: {full_output_dir}")
    
    # 保存训练配置
    config_path = os.path.join(full_output_dir, "args.json")
    with open(config_path, "w", encoding="utf-8") as f:
        # 将args转换为dict并保存
        config = vars(args)
        config["timestamp"] = timestamp
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    # 选择设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             'mps' if torch.backends.mps.is_available() else 
                             'cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 解析欺骗模型列表
    spoof_models = None
    if args.spoof_models:
        spoof_models = args.spoof_models.split(',')
    
    # 准备数据集
    print("准备数据集...")
    train_files, train_labels, val_files, val_labels, test_files, test_labels = \
        prepare_dataset(args.data_dir, val_ratio=args.val_ratio, random_seed=args.seed,
                       spoof_models=spoof_models)
    
    # 初始化MFCC处理器
    processor = MFCCProcessor(sample_rate=args.sample_rate, audio_length=args.audio_length)
    
    # 创建数据集
    train_dataset = MFCCDataset(train_files, train_labels, processor, train_mode=True, audio_length=args.audio_length)
    val_dataset = MFCCDataset(val_files, val_labels, processor, train_mode=False, audio_length=args.audio_length)
    test_dataset = MFCCDataset(test_files, test_labels, processor, train_mode=False, audio_length=args.audio_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化模型
    model = AudioClassifier(
        input_size=processor.n_mfcc, 
        channels=args.channels, 
        emb_size=args.emb_size
    ).to(device)
    
    # 打印模型信息
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 创建日志文件
    log_file = os.path.join(full_output_dir, "training_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练配置: {json.dumps(vars(args), ensure_ascii=False)}\n\n")
        f.write(f"训练样本数: {len(train_dataset)}\n")
        f.write(f"验证样本数: {len(val_dataset)}\n")
        f.write(f"测试样本数: {len(test_dataset)}\n\n")
    
    # 训练循环
    best_val_eer = float('inf')
    best_epoch = 0
    
    print(f"开始训练 {args.epochs} 个epochs...")
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 在验证集上评估
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 训练日志
        log_message = f"Epoch {epoch+1}/{args.epochs} - " \
                     f"训练损失: {train_metrics['loss']:.4f} - " \
                     f"训练准确率: {train_metrics['accuracy']:.2f}% - " \
                     f"验证损失: {val_metrics['loss']:.4f} - " \
                     f"验证准确率: {val_metrics['accuracy']*100:.2f}% - " \
                     f"验证EER: {val_metrics['eer']*100:.2f}%"
        
        # 打印当前epoch结果
        print(log_message)
        
        # 保存到日志文件
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
        
        # 更新学习率
        scheduler.step(val_metrics['loss'])
        
        # 保存最佳模型
        if val_metrics['eer'] < best_val_eer:
            best_val_eer = val_metrics['eer']
            best_epoch = epoch + 1
            # 保存最佳模型（仅状态字典）
            best_model_path = os.path.join(full_output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            best_message = f"【保存最佳模型】当前验证EER: {best_val_eer*100:.2f}%"
            print(best_message)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(best_message + "\n")
    
    # 保存最终模型（仅状态字典）
    final_model_path = os.path.join(full_output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    
    final_message = f"\n训练完成! 最佳模型（Epoch {best_epoch}）验证EER: {best_val_eer*100:.2f}%"
    print(final_message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(final_message + "\n")
    
    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    # 直接加载模型状态字典
    model.load_state_dict(torch.load(os.path.join(full_output_dir, "best_model.pth")))
    
    # 在测试集上评估
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # 测试结果
    test_results = [
        "\n测试集评估结果:",
        f"准确率(ACC): {test_metrics['accuracy']*100:.2f}%",
        f"精确率(P): {test_metrics['precision']*100:.2f}%",
        f"召回率(R): {test_metrics['recall']*100:.2f}%",
        f"F1分数: {test_metrics['f1']*100:.2f}%",
        f"等错误率(EER): {test_metrics['eer']*100:.2f}%",
        f"EER对应阈值: {test_metrics['threshold']:.4f}"
    ]
    
    # 打印测试结果
    for line in test_results:
        print(line)
    
    # 保存测试结果
    with open(log_file, "a", encoding="utf-8") as f:
        for line in test_results:
            f.write(line + "\n")
        f.write(f"\n训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="语音伪造检测模型训练脚本")
    
    # 数据集参数
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="数据集根目录")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--sample_rate", type=int, default=16000,
                        help="音频采样率")
    parser.add_argument("--audio_length", type=int, default=3,
                        help="处理音频的长度(秒)")
    parser.add_argument("--spoof_models", type=str, default="",
                        help="欺骗模型列表，逗号分隔，例如'sparktts,cosyvoice,f5tts'，默认为全部")
    
    # 模型参数
    parser.add_argument("--channels", type=int, default=512,
                        help="ECAPA-TDNN通道数")
    parser.add_argument("--emb_size", type=int, default=192,
                        help="嵌入特征维度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载器工作线程数")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备 (auto, cuda, cpu, mps)")
    parser.add_argument("--output_dir", type=str, default="pretrained_models",
                        help="输出目录")
    parser.add_argument("--run_name", type=str, default="",
                        help="运行名称前缀，默认使用时间戳")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()
    
    main(args) 