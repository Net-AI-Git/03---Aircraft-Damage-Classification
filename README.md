# Aircraft Damage Classification System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Net-AI-Git/03---Aircraft-Damage-Classification/graphs/commit-activity)

A comprehensive deep learning system for classifying aircraft damage into two categories: **dent** vs **crack**. This production-grade implementation leverages multiple CNN architectures with advanced training strategies, featuring intelligent caching, memory optimization, and ensemble learning for optimal performance in aircraft maintenance and safety applications.

## 📋 Table of Contents

- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Architectures & Strategies](#-model-architectures--strategies)
- [Results & Evaluation](#-results--evaluation)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)

## 🚀 Features

- 🧠 **4 State-of-the-art CNN Architectures**: VGG16, ResNet50, EfficientNet-B0, DenseNet121
- 🎯 **5 Training Strategies**: Partial fine-tuning, feature extraction (2 variants), progressive fine-tuning (2 variants)
- 🔄 **Advanced Data Augmentation**: Albumentations-powered pipeline optimized for aircraft damage detection
- 🏆 **Ensemble Learning**: Weighted ensemble from top-performing models using softmax weighting
- 📊 **Comprehensive Visualization**: Training progress plots, individual experiment analysis, and comparative visualizations
- ⚡ **Performance Optimization**: Intelligent caching system, memory-efficient sequential execution
- 🛠️ **Production-Ready**: Error handling, logging, resource monitoring, and modular architecture
- 🎨 **ImageNet Normalization**: Optimal transfer learning setup for pre-trained models

## 🛠️ Technologies Used

**Core Framework:**
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) Python 3.8+
- ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=TensorFlow&logoColor=white) TensorFlow 2.10+
- ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=Keras&logoColor=white) Keras (via tf.keras)

**Data Processing & Augmentation:**
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) NumPy
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) Pandas
- Albumentations

**Visualization & Analysis:**
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white) Matplotlib
- ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white) Seaborn

**System Monitoring:**
- psutil (optional)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA & cuDNN for GPU acceleration

### Step-by-Step Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Net-AI-Git/03---Aircraft-Damage-Classification.git
cd 03---Aircraft-Damage-Classification
```

2. **Create virtual environment (recommended):**
```bash
python -m venv aircraft_damage_env

# Activate virtual environment
# Windows:
aircraft_damage_env\Scripts\activate
# Linux/Mac:
source aircraft_damage_env/bin/activate
```

3. **Install required packages:**
```bash
pip install -r requirements.txt
```

4. **Verify installation:**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}'); print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\"))}')"
```

## 🚀 Usage

### Basic Usage

Run the complete experimental pipeline with default configuration:

```python
python aircraft_damage_optimized.py
```

### Advanced Configuration

Customize experiment parameters by modifying the configuration:

```python
from aircraft_damage_optimized import ExperimentConfiguration, OptimizedExperimentOrchestrator

# Create custom configuration
config = ExperimentConfiguration(
    batch_size=64,
    initial_epochs=25,
    learning_rate=0.0005,
    image_height=256,
    image_width=256
)

# Run experiments with custom config
orchestrator = OptimizedExperimentOrchestrator(config)
orchestrator.run_comprehensive_experiments()
```

### Individual Model Training

Train a specific model with a particular strategy:

```python
from aircraft_damage_optimized import *

# Initialize components
config = ExperimentConfiguration()
data_repo = OptimizedDataRepository(config)
model_factory = OptimizedModelFactory(config)

# Get data generators
dataset_path = data_repo.download_and_extract_dataset()
train_gen, val_gen, test_gen = data_repo.get_data_generators(dataset_path)

# Train specific model
strategy = PartialFineTuningStrategy(config, model_factory)
test_accuracy, history, model = strategy.execute_experiment('ResNet50', train_gen, val_gen, test_gen)

print(f"Test Accuracy: {test_accuracy:.4f}")
```

## 📁 Project Structure

```
aircraft-damage-classification/
│
├── aircraft_damage_optimized.py    # Main implementation file
├── requirements.txt                 # Package dependencies
├── README.md                       # Project documentation
├── LICENSE                         # MIT License
│
├── results/                        # Generated results and visualizations
│   ├── experiment_results.csv      # Comprehensive results table
│   ├── *_training_progress.png     # Training progress plots
│   ├── *_all_experiments.png       # Individual experiment analysis
│   └── test_accuracy_comparison.png # Comparative analysis
│
├── aircraft_damage_dataset_v1/     # Dataset directory (auto-downloaded)
│   ├── train/                      # Training data
│   │   ├── crack/                  # Crack images
│   │   └── dent/                   # Dent images
│   ├── valid/                      # Validation data
│   └── test/                       # Test data
│
└── aircraft_damage_experiments.log # Detailed execution logs
```

## 🏗️ Model Architectures & Strategies

### CNN Architectures
- **VGG16**: Deep architecture with small filters, excellent feature extraction
- **ResNet50**: Residual connections for training very deep networks
- **EfficientNet-B0**: Compound scaling for optimal accuracy-efficiency trade-off
- **DenseNet121**: Dense connections for feature reuse and parameter efficiency

### Training Strategies
1. **Partial Fine-tuning**: Unfreezes last 4 layers from the beginning
2. **Feature Extraction (Arch1)**: ReLU → Dropout → BatchNorm pattern
3. **Feature Extraction (Arch2)**: ReLU → BatchNorm → Dropout pattern
4. **Progressive Fine-tuning (Arch1)**: Two-stage training with gradual unfreezing
5. **Progressive Fine-tuning (Arch2)**: Enhanced progressive approach

## 📊 Results & Evaluation

### Experimental Setup
- **Total Experiments**: 20 (4 models × 5 strategies)
- **Dataset Split**: 70% train, 15% validation, 15% test
- **Image Size**: 224×224 pixels
- **Batch Size**: 32
- **Data Augmentation**: Rotation, flips, brightness/contrast adjustment, Gaussian blur

<!-- ADD TRAINING PROGRESS PLOTS HERE -->
*Training Progress Comparison for All Models:*

![VGG16 Training Progress](results/VGG16_training_progress.png)
![ResNet50 Training Progress](results/ResNet50_training_progress.png)
![EfficientNet-B0 Training Progress](results/EfficientNet-B0_training_progress.png)
![DenseNet121 Training Progress](results/DenseNet121_training_progress.png)

<!-- ADD INDIVIDUAL EXPERIMENT PLOTS HERE -->
*Individual Experiment Analysis:*

![VGG16 All Experiments](results/VGG16_all_experiments.png)
![ResNet50 All Experiments](results/ResNet50_all_experiments.png)
![EfficientNet-B0 All Experiments](results/EfficientNet-B0_all_experiments.png)
![DenseNet121 All Experiments](results/DenseNet121_all_experiments.png)

<!-- ADD COMPARISON PLOT HERE -->
*Test Accuracy Comparison:*

![Test Accuracy Comparison](results/test_accuracy_comparison.png)

### Key Findings

<!-- PLACEHOLDER: ADD YOUR ACTUAL RESULTS HERE -->
```
[TO BE FILLED: Add your actual experimental results here]
Example format:
- Best performing model: [Model Name] - [Strategy] with [X.XXX] accuracy
- Ensemble accuracy: [X.XXX]
- Most effective strategy: [Strategy Name]
- Training time comparison: [Details]
```

### Performance Metrics
- **Primary Metric**: Binary classification accuracy on test set
- **Ensemble Method**: Softmax-weighted combination of top 4 models
- **Validation Strategy**: Early stopping with patience of 3 epochs
- **Learning Rate Scheduling**: ReduceLROnPlateau with factor 0.5

## 🔮 Future Work

- [ ] **Extended Dataset**: Incorporate additional damage types (corrosion, fatigue cracks)
- [ ] **Real-time Inference**: Deploy optimized model for mobile/edge devices
- [ ] **Explainable AI**: Integrate Grad-CAM for damage localization visualization
- [ ] **Data Efficiency**: Implement few-shot learning for rare damage types
- [ ] **3D Analysis**: Extend to depth/stereo image analysis for damage assessment
- [ ] **Integration**: Develop API endpoints for maintenance system integration

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 style guide
- Add comprehensive docstrings for new functions
- Include unit tests for new features
- Update documentation as needed

## 📧 Contact

**Netanel Itzhak**
- 📧 Email: [ntitz19@gmail.com](mailto:ntitz19@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/netanelitzhak](https://www.linkedin.com/in/netanelitzhak)
- 🐙 GitHub: [github.com/Net-AI-Git](https://github.com/Net-AI-Git)

## 🙏 Acknowledgments

- Dataset provided by IBM Cloud Object Storage
- TensorFlow team for the excellent deep learning framework
- Albumentations library for advanced data augmentation capabilities
- Open source community for the foundational CNN architectures
- Aircraft maintenance professionals for domain expertise insights

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
