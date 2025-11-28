# run_all.py
import os
import json
from data_loader import get_cifar10_data_loaders
from models import get_model, count_parameters
from trainers import CIFAR10Trainer
from utils import setup_directories, save_model, save_results
from config import config


def run_all_experiments():
    """
    运行所有模型和优化器的组合实验
    """
    # 设置目录
    setup_directories()

    # 存储所有结果
    all_results = {}

    for model_name in config.models:
        for optimizer_name in config.optimizers:
            try:
                print(f"\n{'=' * 80}")
                print(f"Running experiment: {model_name} with {optimizer_name}")
                print(f"{'=' * 80}")

                # 获取数据
                trainloader, testloader = get_cifar10_data_loaders()

                # 获取模型
                model = get_model(model_name)
                print(f"Model: {model_name}, Parameters: {count_parameters(model):,}")

                # 创建训练器
                trainer = CIFAR10Trainer(model, trainloader, testloader, optimizer_name, model_name)

                # 训练模型
                history = trainer.train()

                # 获取最终指标
                final_metrics = trainer.get_final_metrics()
                history['final_metrics'] = final_metrics

                # 保存模型
                save_model(model, model_name, optimizer_name)

                # 存储结果
                result_key = f"{model_name}_{optimizer_name}"
                all_results[result_key] = history

                print(f"Completed: {result_key}")
                print(f"Final Accuracy: {final_metrics['final_accuracy']:.2f}%")

            except Exception as e:
                print(f"Error in {model_name}_{optimizer_name}: {str(e)}")
                continue

    # 保存所有结果
    if all_results:
        results_file = save_results(all_results)

        print(f"\n{'=' * 80}")
        print("ALL EXPERIMENTS COMPLETED!")
        print(f"Results saved to: {results_file}")
        print(f"{'=' * 80}")

        return results_file
    else:
        print("No experiments completed successfully!")
        return None


if __name__ == "__main__":
    run_all_experiments()