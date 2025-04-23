"""
评估指标相关函数
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def compute_eer(y_true, y_score):
    """
    计算等错误率(Equal Error Rate, EER)
    
    Args:
        y_true (array-like): 真实标签(0=真实,1=虚假)
        y_score (array-like): 预测为虚假的概率(0-1)
        
    Returns:
        tuple: (eer, threshold) - EER值和对应的阈值
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    # 用插值找到假阳率(FPR)=漏检率(1-TPR)的点，即EER
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # 用插值找到EER对应的阈值
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def compute_metrics(y_true, y_pred, y_score):
    """
    计算常用评估指标
    
    Args:
        y_true (array-like): 真实标签(0=真实,1=虚假)
        y_pred (array-like): 预测标签(0=真实,1=虚假)
        y_score (array-like): 预测为虚假的概率(0-1)
        
    Returns:
        dict: 包含各项指标的字典
    """
    # 计算精确率、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # 计算准确率
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    
    # 计算EER
    eer, threshold = compute_eer(y_true, y_score)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'eer': eer,
        'threshold': threshold
    } 