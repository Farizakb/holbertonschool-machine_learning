# Machine Learning Curriculum - From Fundamentals to Transformers & MLOps

A comprehensive machine learning educational repository covering foundational mathematics through advanced deep learning — including attention, transformers, GANs, and autoencoders — and closing with MLOps practices for taking models to production.

## 📚 Project Overview

This repository contains a structured progression of machine learning projects and implementations, starting from mathematical foundations and advancing through supervised learning, unsupervised learning, specialized deep learning architectures, and finally MLOps.

**Learning Path:** Mathematical Foundations → Data Processing → Supervised Learning → Unsupervised Learning → Advanced Deep Learning → MLOps

## 📁 Repository Structure

### 1. **math/** - Mathematical Foundations
Essential mathematical concepts for machine learning:
- **linear_algebra/** - Vectors, matrices, and linear transformations
- **advanced_linear_algebra/** - Determinants, inverses, eigenvalues, and definiteness
- **calculus/** - Derivatives, gradients, and optimization fundamentals
- **probability/** - Probability theory and distributions
- **bayesian_prob/** - Bayesian statistics and inference
- **convolutions_and_pooling/** - Convolution and pooling operations from scratch
- **plotting/** - Data visualization with Matplotlib

### 2. **pipeline/** - Data Processing & Preparation
Real-world data handling and preprocessing:
- **pandas/** - Data manipulation and analysis with pandas
- **data_augmentation/** - Techniques for augmenting training datasets

### 3. **supervised_learning/** - Core ML Algorithms & Deep Learning
Predictive models and supervised techniques:

#### Classical ML
- **classification/** - Binary and multi-class classification, neural nets from scratch
- **decision_tree/** - Decision trees, random forests, and isolation forests
- **optimization/** - Optimization algorithms (momentum, RMSProp, Adam, batch norm)
- **regularization/** - L2, dropout, early stopping, and data augmentation
- **error_analysis/** - Confusion matrices, bias/variance, and model diagnostics

#### Deep Learning - Neural Networks
- **keras/** - Building neural networks with Keras/TensorFlow
- **cnn/** - Convolutional Neural Networks from scratch
- **deep_cnns/** - Inception, ResNet, DenseNet, and other deep architectures

#### Sequence Models & NLP
- **RNNs/** - RNN, GRU, LSTM, bidirectional, and deep RNN cells
- **word_embeddings/** - Word2Vec, GloVe, FastText, and embedding layers
- **attention/** - Attention mechanisms and the Transformer architecture
- **transformer_apps/** - Applied transformers (tokenization, training pipelines)
- **qa_bot/** - Question-answering bot using pre-trained transformer models
- **time_series/** - Time series forecasting with recurrent models

#### Computer Vision - Specialized
- **transfer_learning/** - Transfer learning and fine-tuning pre-trained models
- **object_detection/** - YOLO and object detection pipelines
- **neural_style_transfer/** - Artistic style transfer using deep learning

### 4. **unsupervised_learning/** - Pattern Discovery
Learning structure from unlabeled data:
- **dimensionality_reduction/** - PCA, t-SNE, and manifold methods
- **clustering/** - K-means, GMM, EM, and hierarchical clustering
- **hyperparameter_tuning/** - Gaussian processes and Bayesian optimization
- **autoencoders/** - Vanilla, sparse, convolutional, and variational autoencoders
- **gan/** - Generative Adversarial Networks

### 5. **MLOps** - Production Machine Learning 🔜
Taking models from notebook to production (upcoming final phase):
- Experiment tracking & model registry
- Model packaging, serving, and APIs
- Containerization & orchestration
- CI/CD for ML pipelines
- Monitoring, drift detection, and retraining
- Reproducibility & data/model versioning

## 🎯 Learning Objectives

- **Foundation:** Master mathematical concepts essential for ML (linear algebra, calculus, probability)
- **Data Processing:** Prepare, clean, and augment datasets for training
- **Supervised Learning:** Implement and understand classical and deep learning algorithms
- **Sequence Modeling & NLP:** Build RNNs, embeddings, attention, and transformers
- **Unsupervised Learning:** Discover patterns and generate data with clustering, autoencoders, and GANs
- **MLOps:** Deploy, monitor, and maintain models in production

## 🚀 Key Topics Covered

✅ Linear Algebra & Advanced Matrix Operations
✅ Calculus & Optimization
✅ Probability & Bayesian Methods
✅ Data Preprocessing & Augmentation
✅ Classification & Regression
✅ Decision Trees & Random Forests
✅ Neural Networks (from scratch and with Keras)
✅ Convolutional Neural Networks & Deep CNN Architectures
✅ Transfer Learning
✅ Object Detection
✅ Neural Style Transfer
✅ RNNs, GRUs & LSTMs
✅ Word Embeddings
✅ Attention & Transformers
✅ Question Answering Bots
✅ Time Series Forecasting
✅ Dimensionality Reduction & Clustering
✅ Hyperparameter Tuning (Bayesian Optimization)
✅ Autoencoders & Variational Autoencoders
✅ Generative Adversarial Networks
✅ Model Optimization & Regularization
✅ Error Analysis & Evaluation
🔜 MLOps & Production Deployment

## 📝 Project Status

🔄 **In Development** - Math, pipeline, supervised, and unsupervised modules complete through transformers and GANs. MLOps modules are the remaining final phase.

## 💡 Getting Started

1. Clone the repository
2. Start with the **math/** folder to understand foundational concepts
3. Progress through **supervised_learning/** and **unsupervised_learning/** sequentially
4. Explore specialized architectures (attention, transformers, GANs) once core concepts are mastered
5. Finish with **MLOps** to put models into production

## 🛠️ Technologies & Libraries

- Python 3
- NumPy - Numerical computing
- Pandas - Data manipulation
- TensorFlow/Keras - Deep learning framework
- Scikit-learn - Machine learning algorithms
- Matplotlib/Seaborn - Data visualization

## 📖 Curriculum Flow

```
Mathematical Foundations (math/)
    ↓
Data Processing (pipeline/)
    ↓
Supervised Learning (supervised_learning/)
    ├── Classical ML (Classification, Decision Trees, Optimization)
    ├── Deep Learning (Keras, CNN, Deep CNNs)
    ├── Sequence & NLP (RNNs, Word Embeddings, Attention, Transformers, QA Bot)
    └── Vision (Transfer Learning, Object Detection, Neural Style Transfer)
    ↓
Unsupervised Learning (unsupervised_learning/)
    ├── Dimensionality Reduction, Clustering
    ├── Hyperparameter Tuning
    └── Autoencoders, GANs
    ↓
MLOps (production, deployment, monitoring)
```

## 📌 Next Steps

- [ ] Experiment tracking (MLflow / Weights & Biases)
- [ ] Model serving & REST APIs (FastAPI, TF Serving)
- [ ] Containerization & orchestration (Docker, Kubernetes)
- [ ] CI/CD pipelines for ML
- [ ] Model monitoring & drift detection
- [ ] Data & model versioning (DVC)
- [ ] Create end-to-end production projects

## 📄 License

This project is part of the Holberton School curriculum.

---

**Note:** This curriculum is continuously evolving. Each module contains practical implementations and exercises to reinforce learning.
