# CIFAR-10 Optimizer Comparison Experiment

A comprehensive PyTorch-based deep learning project that systematically compares the performance of different optimizers on the CIFAR-10 image classification task.

## 📋 Project Overview

This project empirically evaluates 5 popular optimizers (SGD, Adam, AdamW, RMSprop, Adagrad) across 3 classic CNN architectures (VGG16, ResNet50, DenseNet121) to provide evidence-based insights for optimizer selection in computer vision tasks.

### Key Features
- 🔬 **Multi-Model Comparison**: VGG16, ResNet50, DenseNet121
- ⚡ **Multi-Optimizer Testing**: SGD, Adam, AdamW, RMSprop, Adagrad
- 📊 **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- 📈 **Visualization Tools**: Automatic comparison charts and training curves
- 💾 **Experiment Management**: Automatic saving of models, results, and TensorBoard logs
- 🔄 **Flexible Execution**: Run single experiments or batch comparisons

## 🗂️ Project Structure
├── config.py # Experiment configuration parameters  
├── data_loader.py # Data loading and preprocessing  
├── models.py # Model definitions and loading  
├── trainers.py # Training and testing logic  
├── utils.py # Utility functions  
├── run_single.py # Run single experiment  
├── run_all.py # Run all experiment combinations  
├── analyze_results.py # Results analysis and visualization  
├── requirements.txt # Python dependencies  
└── README.md # Project documentation  


## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- CUDA (recommended for GPU acceleration)

### Install Dependencies

```bash  
pip install -r requirements.txt
```
## ⚙️ Configuration

### Key parameters in config.py:

```bash
# Device settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters
batch_size = 128
epochs = 50
learning_rate = 0.0001

# Models and optimizers to test
models = ['vgg16', 'resnet50', 'densenet121']
optimizers = ['sgd', 'adam', 'adamw', 'rmsprop', 'adagrad']
```

## 🚀 Quick Start
### 1. Run Single Experiment
- Test a specific model-optimizer combination:
```bash
python run_single.py --model vgg16 --optimizer adam --epochs 50
```
- Available models: vgg16, resnet50, densenet121
- Available optimizers: sgd, adam, adamw, rmsprop, adagrad

### 2. Run All Experiments
Automatically run all model-optimizer combinations:
```bash
python run_all.py
```

### 3. Analyze Results
Generate comprehensive reports and visualizations:
```bash
python analyze_results.py --results-file ./results/experiment_results_1234567890.json
```

## 📊 Output & Results
### The experiment generates:
- Trained Models: Saved in ./saved_models/ directory
- Experiment Results: JSON format in ./results/ directory
- TensorBoard Logs: Training metrics in ./runs/ directory
- Visualization Charts: Comprehensive comparison plots

### Generated Charts Include:
- Final Accuracy Comparison: Bar charts across all combinations
- Training Curves: Test accuracy evolution over epochs
- Best Accuracy Analysis: Highest achieved performance
- F1-Score Comparison: Detailed metric analysis by optimizer

## 🎯 Experimental Setup
### Default Configuration:
- Dataset: CIFAR-10 (10 classes, 32×32 RGB images)
- Training Epochs: 50
- Batch Size: 128
- Learning Rate: 0.0001
- Data Augmentation: Random cropping, horizontal flipping, rotation

### Optimizer Settings:
- All optimizers use the same learning rate and weight decay (5e-4)
- SGD uses momentum (0.9)
- Cosine annealing learning rate scheduler applied to all

## 🔧 Customization
### Modify Models
- Edit models.py to add new architectures:
```bash
def get_model(model_name, num_classes=10):
    if model_name == 'your_model':
        # Your implementation
        pass
```

### Add Optimizers
- Extend trainers.py:
```bash
def _get_optimizer(self, optimizer_name):
    if optimizer_name == 'your_optimizer':
        return optim.YourOptimizer(self.model.parameters(), lr=config.learning_rate)
```

### Adjust Data Augmentation
- Modify data_loader.py:
```bash
transform_train = transforms.Compose([
    # Add your custom transforms
    transforms.YourTransform(),
])
```

## 📈 Expected Results
Based on typical CIFAR-10 benchmarks:
- ResNet50 generally achieves highest accuracy
- Adam/AdamW often show fastest convergence
- SGD with momentum can achieve competitive final performance
- Results may vary based on hyperparameter tuning

## ⚠️ Notes & Tips
- First Run: Automatically downloads CIFAR-10 dataset (~163MB)
- Storage: Ensure sufficient disk space for models and results
- GPU Recommended: Significant speedup over CPU training
- Time Considerations: Full experiment suite may take several hours
- Reproducibility: Set random seeds for consistent results

## 🤝 Contributing
Contributions are welcome! Please feel free to submit:
- Bug reports
- Feature requests
- Performance improvements
- Additional models or optimizers

## 📄 License
This project is intended for educational and research purposes. Please cite appropriately if used in academic work.







