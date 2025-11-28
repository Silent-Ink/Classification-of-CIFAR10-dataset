# trainers.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from config import config
from utils import get_tensorboard_writer


class CIFAR10Trainer:
    def __init__(self, model, trainloader, testloader, optimizer_name, model_name):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer_name = optimizer_name
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()

        # 获取优化器
        self.optimizer = self._get_optimizer(optimizer_name)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs)

        # TensorBoard写入器
        self.writer = get_tensorboard_writer(model_name, optimizer_name)

        # 存储训练历史
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }

    def _get_optimizer(self, optimizer_name):
        """获取指定优化器"""
        if optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=config.learning_rate,
                             momentum=0.9, weight_decay=5e-4)
        elif optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=config.learning_rate,
                              weight_decay=5e-4)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=config.learning_rate,
                               weight_decay=5e-4)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=config.learning_rate,
                                 weight_decay=5e-4)
        elif optimizer_name == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=config.learning_rate,
                                 weight_decay=5e-4)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.trainloader, desc=f'{self.model_name}_{self.optimizer_name} - Epoch {epoch}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = running_loss / len(self.trainloader)
        accuracy = 100. * correct / total

        self.history['train_loss'].append(avg_loss)
        self.history['train_accuracy'].append(accuracy)

        # 记录到TensorBoard
        self.writer.add_scalar('Train/Loss', avg_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', accuracy, epoch)

        return avg_loss, accuracy

    def test_epoch(self, epoch):
        """测试一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(config.device), targets.to(config.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = running_loss / len(self.testloader)
        accuracy = 100. * correct / total

        # 计算其他指标
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        # 存储结果
        self.history['test_loss'].append(avg_loss)
        self.history['test_accuracy'].append(accuracy)
        self.history['precision'].append(precision)
        self.history['recall'].append(recall)
        self.history['f1_score'].append(f1)

        # 记录到TensorBoard
        self.writer.add_scalar('Test/Loss', avg_loss, epoch)
        self.writer.add_scalar('Test/Accuracy', accuracy, epoch)
        self.writer.add_scalar('Test/Precision', precision, epoch)
        self.writer.add_scalar('Test/Recall', recall, epoch)
        self.writer.add_scalar('Test/F1-Score', f1, epoch)

        # 记录学习率
        current_lr = self.scheduler.get_last_lr()[0]
        self.writer.add_scalar('Learning_Rate', current_lr, epoch)

        return avg_loss, accuracy, precision, recall, f1

    def train(self, epochs=None):
        """完整训练过程"""
        if epochs is None:
            epochs = config.epochs

        print(f"\n{'=' * 60}")
        print(f"Training {self.model_name} with {self.optimizer_name}")
        print(f"{'=' * 60}")

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch + 1)

            # 测试
            test_loss, test_acc, precision, recall, f1 = self.test_epoch(epoch + 1)

            # 更新学习率
            self.scheduler.step()

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
            print(f"  Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print("-" * 50)

        # 关闭TensorBoard写入器
        self.writer.close()

        return self.history

    def get_final_metrics(self):
        """获取最终的性能指标"""
        return {
            'final_accuracy': self.history['test_accuracy'][-1],
            'final_precision': self.history['precision'][-1],
            'final_recall': self.history['recall'][-1],
            'final_f1': self.history['f1_score'][-1],
            'best_accuracy': max(self.history['test_accuracy']),
            'best_epoch': int(np.argmax(self.history['test_accuracy']) + 1)  # 确保是Python int
        }