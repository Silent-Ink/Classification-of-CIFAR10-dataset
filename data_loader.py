import torch
import torchvision
import torchvision.transforms as transforms
from config import config


def get_cifar10_data_loaders(batch_size=None):
    """
    获取CIFAR-10数据加载器
    """
    if batch_size is None:
        batch_size = config.batch_size

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),# ← 数据增强：随机裁剪
        transforms.RandomHorizontalFlip(),# ← 数据增强：随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载训练集
    trainset = torchvision.datasets.CIFAR10(
        root=config.data_path, train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers)

    # 加载测试集
    testset = torchvision.datasets.CIFAR10(
        root=config.data_path, train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=config.num_workers)

    return trainloader, testloader


def get_cifar10_classes():
    """返回CIFAR-10类别名称"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']