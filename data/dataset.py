"""
数据集类和数据处理函数
"""

import os
import random
import torch
import numpy as np
import soundfile
import soxr
from torch.utils.data import Dataset
from tqdm import tqdm


def collect_wav_files(dirs):
    """
    从目录列表中递归收集所有WAV文件
    
    Args:
        dirs (list): 目录路径列表
        
    Returns:
        list: 收集到的WAV文件路径列表
    """
    all_files = []
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            print(f"警告: 目录不存在 {dir_path}")
            continue
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.wav'):
                    all_files.append(os.path.join(root, file))
    return all_files


def filter_invalid_files(file_list):
    """
    过滤掉无效的音频文件
    
    Args:
        file_list (list): 文件路径列表
        
    Returns:
        list: 有效文件路径列表
    """
    valid_files = []
    invalid_count = 0
    
    for file_path in tqdm(file_list, desc="检查有效性"):
        try:
            # 检查文件是否存在且大小合适
            if not os.path.exists(file_path) or os.path.getsize(file_path) < 44:
                invalid_count += 1
                continue
            
            # 尝试读取音频
            waveform, sr = soundfile.read(file_path, dtype='float32')
            if len(waveform) == 0:
                invalid_count += 1
                continue
                
            valid_files.append(file_path)
        except Exception as e:
            print(f"文件无效: {file_path}, 原因: {e}")
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"已过滤 {invalid_count} 个无效文件")
    
    return valid_files


def prepare_dataset(base_dir, test_ratio=0.1, val_ratio=0.1, random_seed=42, 
                    spoof_models=None):
    """
    准备数据集，分为训练集、验证集和测试集
    
    Args:
        base_dir (str): 数据集根目录
        test_ratio (float): 测试集比例
        val_ratio (float): 验证集比例
        random_seed (int): 随机种子
        spoof_models (list): 欺骗模型列表，默认为["sparktts", "cosyvoice", "f5tts", "naturalspeech3"]
        
    Returns:
        tuple: (train_files, train_labels, val_files, val_labels, test_files, test_labels)
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 数据集路径配置
    wav_dir = os.path.join(base_dir, 'wav')
    
    # 定义真实音频和虚假音频目录
    real_dirs = [
        os.path.join(wav_dir, 'train'),
        os.path.join(wav_dir, 'dev')
    ]
    
    # 如果未指定欺骗模型，使用默认列表
    if spoof_models is None:
        spoof_models = ['sparktts', 'cosyvoice', 'f5tts', 'naturalspeech3']
    
    # 打印使用的欺骗模型
    print(f"使用的欺骗模型: {', '.join(spoof_models)}")
    
    spoof_dirs = []
    for model in spoof_models:
        for subset in ['train', 'dev']:
            model_dir = os.path.join(wav_dir, f'{subset}-{model.lower()}')
            if os.path.exists(model_dir):
                spoof_dirs.append(model_dir)
            else:
                print(f"警告: 欺骗模型目录不存在 {model_dir}")
    
    test_real_dir = os.path.join(wav_dir, 'test')
    test_spoof_dirs = []
    for model in spoof_models:
        test_model_dir = os.path.join(wav_dir, f'test-{model.lower()}')
        if os.path.exists(test_model_dir):
            test_spoof_dirs.append(test_model_dir)
        else:
            print(f"警告: 测试集欺骗模型目录不存在 {test_model_dir}")
    
    # 收集音频文件
    print("收集真实音频文件...")
    real_files = collect_wav_files(real_dirs)
    print("收集虚假音频文件...")
    spoof_files = collect_wav_files(spoof_dirs)
    print("收集测试集真实音频文件...")
    test_real_files = collect_wav_files([test_real_dir])
    print("收集测试集虚假音频文件...")
    test_spoof_files = collect_wav_files(test_spoof_dirs)
    
    # 过滤无效文件
    print("过滤无效文件...")
    real_files = filter_invalid_files(real_files)
    spoof_files = filter_invalid_files(spoof_files)
    test_real_files = filter_invalid_files(test_real_files)
    test_spoof_files = filter_invalid_files(test_spoof_files)
    
    # 打乱数据顺序
    random.shuffle(real_files)
    random.shuffle(spoof_files)
    random.shuffle(test_real_files)
    random.shuffle(test_spoof_files)
    
    # 从训练集中分出验证集
    real_val_size = int(len(real_files) * val_ratio)
    spoof_val_size = int(len(spoof_files) * val_ratio)
    
    val_real = real_files[:real_val_size]
    train_real = real_files[real_val_size:]
    
    val_spoof = spoof_files[:spoof_val_size]
    train_spoof = spoof_files[spoof_val_size:]
    
    # 合并数据并创建标签
    def combine_data(real_data, spoof_data):
        combined = real_data + spoof_data
        labels = [0]*len(real_data) + [1]*len(spoof_data)
        combined_data = list(zip(combined, labels))
        random.shuffle(combined_data)
        files, labels = zip(*combined_data) if combined_data else ([], [])
        return list(files), list(labels)
    
    train_files, train_labels = combine_data(train_real, train_spoof)
    val_files, val_labels = combine_data(val_real, val_spoof)
    test_files, test_labels = combine_data(test_real_files, test_spoof_files)
    
    # 打印数据集信息
    print(f"训练集: {len(train_files)} 个样本 (真实: {len(train_real)}, 虚假: {len(train_spoof)})")
    print(f"验证集: {len(val_files)} 个样本 (真实: {len(val_real)}, 虚假: {len(val_spoof)})")
    print(f"测试集: {len(test_files)} 个样本 (真实: {len(test_real_files)}, 虚假: {len(test_spoof_files)})")
    
    return train_files, train_labels, val_files, val_labels, test_files, test_labels


        
class MFCCDataset(Dataset):
    """MFCC特征数据集处理流程（含SpecAugment）"""
    def __init__(self, file_list, labels, processor, train_mode=True, audio_length=3):
        
        """
        Args:
            file_list (list): 音频文件路径列表
            labels (list): 对应标签列表
            processor (MFCCProcessor): 预处理实例
            train_mode (bool): 训练模式启用数据增强
        """
        self.files = file_list
        self.labels = labels
        self.processor = processor
        self.train_mode = train_mode
        self.target_frames = audio_length * 100  # 转换为帧数 (100帧/秒)

    def __len__(self):
        return len(self.files)

    def _random_crop(self, spec):
        """随机裁剪3秒片段（训练模式专用）"""
        if spec.shape[-1] > self.target_frames:
            start = torch.randint(0, spec.shape[-1] - self.target_frames, (1,))
            return spec[..., start:start+self.target_frames]
        return spec

    def _repeat_pad(self, spec):
        """重复填充短于目标长度的样本"""
        repeats = self.target_frames // spec.shape[-1] + 1
        spec = spec.repeat(1, 1, repeats)
        return spec[..., :self.target_frames]

    def __getitem__(self, idx):
        # 加载音频
        waveform, sr = soundfile.read(self.files[idx], dtype='float32')
        
        # 重采样至目标采样率
        if sr != self.processor.sample_rate:
            waveform = soxr.resample(waveform, sr, self.processor.sample_rate)

        waveform = torch.from_numpy(waveform).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        else:
            waveform = waveform.T[0] # (channels, samples)
            

        # 生成MFCC特征
        mfcc = self.processor(waveform, apply_augment=self.train_mode)  # (n_mfcc, time)

        # 训练模式下的处理
        if self.train_mode:
            # 随机裁剪
            mfcc = self._random_crop(mfcc)
            
            # 随机通道噪声（可选增强）
            if torch.rand(1) < 0.3:
                mfcc += torch.randn_like(mfcc) * 0.01
        else:
            # 验证/测试模式：中心裁剪或填充
            if mfcc.shape[-1] > self.target_frames:
                start = (mfcc.shape[-1] - self.target_frames) // 2
                mfcc = mfcc[..., start:start+self.target_frames]
            else:
                mfcc = self._repeat_pad(mfcc.unsqueeze(0)).squeeze(0)

        # 确保长度固定
        mfcc = mfcc[..., :self.target_frames] if mfcc.shape[-1] > self.target_frames else mfcc
        mfcc = self._repeat_pad(mfcc.unsqueeze(0)).squeeze(0)

        return mfcc, self.labels[idx]