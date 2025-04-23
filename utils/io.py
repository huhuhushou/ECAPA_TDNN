"""
文件IO操作相关函数
"""

import os
import soundfile
from tqdm import tqdm


def scan_audio_files(base_dir, remove_invalid=False):
    """
    扫描音频文件，检测损坏或空的文件
    
    Args:
        base_dir (str): 音频文件的基础目录
        remove_invalid (bool): 是否删除无效文件
        
    Returns:
        tuple: (valid_files, invalid_files) - 有效和无效文件列表
    """
    print(f"开始扫描目录: {base_dir} 下的音频文件...")
    valid_files = []
    invalid_files = []
    total_files = 0
    
    # 要检查的WAV属性
    min_valid_size = 44  # WAV头部至少44字节
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.wav'):
                total_files += 1
                file_path = os.path.join(root, file)
                is_valid = True
                error_reason = ""
                
                # 检查1: 文件存在且大小合适
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size == 0:
                        is_valid = False
                        error_reason = "空文件"
                    elif file_size < min_valid_size:
                        is_valid = False
                        error_reason = f"文件过小 ({file_size} 字节)"
                except Exception as e:
                    is_valid = False
                    error_reason = f"检查文件大小错误: {str(e)}"
                
                # 检查2: 尝试读取音频内容
                if is_valid:
                    try:
                        # 使用soundfile尝试读取
                        wave_data, sr = soundfile.read(file_path, dtype='float32')
                        
                        # 检查波形是否为空
                        if len(wave_data) == 0:
                            is_valid = False
                            error_reason = "波形数据为空"
                        # 检查采样率是否合理
                        elif sr <= 0:
                            is_valid = False
                            error_reason = f"采样率异常: {sr}Hz"
                    except Exception as e:
                        is_valid = False
                        error_reason = f"读取音频错误: {str(e)}"
                
                # 整理结果
                if is_valid:
                    valid_files.append(file_path)
                else:
                    invalid_files.append((file_path, error_reason))
                    print(f"发现无效文件: {file_path} - 原因: {error_reason}")
                    
                    # 如果需要，删除无效文件
                    if remove_invalid:
                        try:
                            os.remove(file_path)
                            print(f"已删除无效文件: {file_path}")
                        except Exception as e:
                            print(f"删除文件失败 {file_path}: {e}")
    
    # 打印统计信息
    print("\n音频文件扫描完成:")
    print(f"总文件数: {total_files}")
    print(f"有效文件: {len(valid_files)}")
    print(f"无效文件: {len(invalid_files)}")
    
    if invalid_files:
        print("\n无效文件列表:")
        for path, reason in invalid_files:
            print(f"- {path} (原因: {reason})")
    
    return valid_files, invalid_files 