# utils.py
import os
import json
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from config import config


def setup_directories():
    """创建必要的目录"""
    os.makedirs(config.results_path, exist_ok=True)
    os.makedirs(config.models_path, exist_ok=True)
    os.makedirs(config.tensorboard_path, exist_ok=True)


def get_tensorboard_writer(model_name, optimizer_name):
    """获取TensorBoard写入器"""
    timestamp = int(time.time())
    log_dir = f"{config.tensorboard_path}/{model_name}_{optimizer_name}_{timestamp}"
    return SummaryWriter(log_dir)


def save_model(model, model_name, optimizer_name):
    """保存模型"""
    filename = f"{config.models_path}/{model_name}_{optimizer_name}_cifar10.pth"
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def convert_to_serializable(obj):
    """将对象转换为JSON可序列化的类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


def save_results(results, filename=None):
    """保存结果到JSON文件"""
    if filename is None:
        filename = f"{config.results_path}/experiment_results_{int(time.time())}.json"

    # 使用转换函数确保所有数据都可序列化
    json_results = convert_to_serializable(results)

    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved to {filename}")
    return filename


def load_results(filename):
    """从JSON文件加载结果"""
    with open(filename, 'r') as f:
        return json.load(f)