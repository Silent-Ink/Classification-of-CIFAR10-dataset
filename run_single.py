import argparse
from data_loader import get_cifar10_data_loaders
from models import get_model, count_parameters
from trainers import CIFAR10Trainer
from utils import setup_directories, save_model, save_results, get_tensorboard_writer


def run_single_experiment(model_name, optimizer_name, epochs=None):
    """
    运行单个模型和优化器的实验
    """
    # 设置目录
    setup_directories()

    # 获取数据
    trainloader, testloader = get_cifar10_data_loaders()

    # 获取模型
    model = get_model(model_name)
    print(f"Model: {model_name}")
    print(f"Number of parameters: {count_parameters(model):,}")

    # 创建训练器
    trainer = CIFAR10Trainer(model, trainloader, testloader, optimizer_name, model_name)

    # 训练模型
    history = trainer.train(epochs)

    # 获取最终指标
    final_metrics = trainer.get_final_metrics()
    history['final_metrics'] = final_metrics

    # 保存模型
    save_model(model, model_name, optimizer_name)

    # 保存结果
    result_key = f"{model_name}_{optimizer_name}"
    results = {result_key: history}
    results_file = save_results(results)

    print(f"\nExperiment completed: {model_name} with {optimizer_name}")
    print(f"Final Test Accuracy: {final_metrics['final_accuracy']:.2f}%")
    print(f"Best Test Accuracy: {final_metrics['best_accuracy']:.2f}% (epoch {final_metrics['best_epoch']})")

    return results_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run single CIFAR-10 experiment')
    parser.add_argument('--model', type=str, required=True,
                        choices=['vgg16', 'resnet50', 'densenet121'],
                        help='Model architecture')
    parser.add_argument('--optimizer', type=str, required=True,
                        choices=['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad'],
                        help='Optimizer')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')

    args = parser.parse_args()

    run_single_experiment(args.model, args.optimizer, args.epochs)