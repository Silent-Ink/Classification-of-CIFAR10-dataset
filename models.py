import torch.nn as nn
from torchvision.models import vgg16, resnet50, densenet121
from config import config


def get_model(model_name, num_classes=10):
    """
    获取指定模型，适配CIFAR-10的32x32输入
    """
    if model_name == 'vgg16':
        model = vgg16(pretrained=False)
        # 修改分类器
        model.classifier[6] = nn.Linear(4096, num_classes)

    elif model_name == 'resnet50':
        model = resnet50(pretrained=False)
        # 修改最后的全连接层
        model.fc = nn.Linear(2048, num_classes)

    elif model_name == 'densenet121':
        model = densenet121(pretrained=False)
        # 修改分类器
        model.classifier = nn.Linear(1024, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(config.device)


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)