# config.py
import torch
import os


# 实验配置
class Config:
    # 设备设置 - 先设置环境变量，再创建device
    device_id = "5"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据配置
    batch_size = 128
    num_workers = 4

    # 训练配置
    epochs = 50
    learning_rate = 0.0001

    # 模型列表
    models = ['vgg16', 'resnet50', 'densenet121']

    # 优化器列表
    optimizers = ['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad']

    # 数据路径
    data_path = './data'

    # 结果保存路径
    results_path = './results'
    models_path = './saved_models'

    # TensorBoard路径
    tensorboard_path = './runs'


# 创建配置实例
config = Config()