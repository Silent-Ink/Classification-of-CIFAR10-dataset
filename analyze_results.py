import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import load_results


def analyze_results(results_file):
    """
    分析实验结果并生成报告
    """
    # 加载结果
    results = load_results(results_file)

    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 100)

    # 按模型分组显示结果
    models = set([key.split('_')[0] for key in results.keys()])

    summary = {}

    for model in models:
        print(f"\n{model.upper()} Results:")
        print("-" * 80)
        print(
            f"{'Optimizer':<12} {'Final Acc':<10} {'Best Acc':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Best Epoch':<12}")
        print("-" * 80)

        model_results = {k: v for k, v in results.items() if k.startswith(model)}
        model_summary = []

        for config, metrics in sorted(model_results.items(),
                                      key=lambda x: x[1]['final_metrics']['final_accuracy'], reverse=True):
            optimizer_name = config.split('_')[1]
            final_acc = metrics['final_metrics']['final_accuracy']
            best_acc = metrics['final_metrics']['best_accuracy']
            precision = metrics['final_metrics']['final_precision']
            recall = metrics['final_metrics']['final_recall']
            f1 = metrics['final_metrics']['final_f1']
            best_epoch = metrics['final_metrics']['best_epoch']

            model_summary.append({
                'optimizer': optimizer_name,
                'final_accuracy': final_acc,
                'best_accuracy': best_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'best_epoch': best_epoch
            })

            print(
                f"{optimizer_name:<12} {final_acc:<10.2f} {best_acc:<10.2f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {best_epoch:<12}")

        summary[model] = model_summary

    # 绘制比较图表
    plot_comparison_charts(results, summary)

    return summary


def plot_comparison_charts(results, summary):
    """
    绘制比较图表
    """
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 颜色配置
    colors = plt.cm.Set3(np.linspace(0, 1, len(summary)))

    # 图表1: 最终准确率比较
    models = list(summary.keys())
    x = np.arange(len(models))
    width = 0.15

    for i, optimizer in enumerate(['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad']):
        accuracies = []
        for model in models:
            model_opt = next((item for item in summary[model] if item['optimizer'] == optimizer), None)
            if model_opt:
                accuracies.append(model_opt['final_accuracy'])
            else:
                accuracies.append(0)

        ax1.bar(x + i * width, accuracies, width, label=optimizer.upper())

    ax1.set_xlabel('Model Architecture')
    ax1.set_ylabel('Final Accuracy (%)')
    ax1.set_title('Final Test Accuracy by Model and Optimizer')
    ax1.set_xticks(x + 2 * width)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图表2: 训练曲线比较 (以第一个模型为例)
    if models:
        first_model = models[0]
        model_results = {k: v for k, v in results.items() if k.startswith(first_model)}

        for config, metrics in model_results.items():
            optimizer_name = config.split('_')[1]
            test_accuracies = metrics['test_accuracy']
            epochs = range(1, len(test_accuracies) + 1)
            ax2.plot(epochs, test_accuracies, label=optimizer_name.upper(), linewidth=2)

        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title(f'{first_model.upper()} - Test Accuracy Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 图表3: 最佳准确率比较
    for i, model in enumerate(models):
        optimizers_data = summary[model]
        accuracies = [item['best_accuracy'] for item in optimizers_data]
        optimizer_names = [item['optimizer'].upper() for item in optimizers_data]

        bars = ax3.bar([f"{model}\n{opt}" for opt in optimizer_names],
                       accuracies, color=colors[i], alpha=0.7)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    ax3.set_xlabel('Model and Optimizer')
    ax3.set_ylabel('Best Accuracy (%)')
    ax3.set_title('Best Test Accuracy Achieved')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 图表4: F1分数比较
    optimizers = ['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad']
    x = np.arange(len(optimizers))

    for i, model in enumerate(models):
        f1_scores = []
        for optimizer in optimizers:
            model_opt = next((item for item in summary[model] if item['optimizer'] == optimizer), None)
            if model_opt:
                f1_scores.append(model_opt['f1'])
            else:
                f1_scores.append(0)

        ax4.plot(x, f1_scores, marker='o', label=model.upper(), linewidth=2)

    ax4.set_xlabel('Optimizer')
    ax4.set_ylabel('F1-Score')
    ax4.set_title('F1-Score by Optimizer')
    ax4.set_xticks(x)
    ax4.set_xticklabels([opt.upper() for opt in optimizers])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimizer_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results-file', type=str, required=True,
                        help='Path to results JSON file')

    args = parser.parse_args()

    analyze_results(args.results_file)