# Monkeypox Epitope Prediction and Classification

This project leverages advanced machine learning models to predict immunogenic epitopes of the Monkeypox virus. By extracting key bioinformatics features from peptide sequences and using both graph- and convolution-based neural networks, we achieve high-accuracy classification that can inform vaccine design and therapeutic development.

## Key Features

### Comprehensive Feature Extraction
Calculates 14 peptide-level properties:

- **Chou–Fasman**
- **Emini**
- **Kolaskar–Tongaonkar**
- **Parker**
- **Isoelectric Point**
- **Aromaticity**
- **Hydrophobicity**
- **Stability**
- **Charge Distribution**
- **Flexibility**
- **Solvent Accessibility**
- **BLOSUM Substitution Score**
- **PTM Sites**
- **Interaction Energy**

### Dual Modeling Approach

- **Graph Neural Network (GNN)**: Constructs a k‑NN graph over peptides and performs two- or three-layer GCN convolutions to capture relational patterns.
  
- **Convolutional Neural Network (CNN)**: Treats the 14-feature vector as a 1D "image" and applies convolutional layers to learn spatial hierarchies.

### Robust Evaluation Logs
Accuracy metrics captured via Python logging for reproducibility and comparison.

## Model Performance

### Graph Neural Network (GNN)
Logged test accuracies:

2025-04-17 21:27:13 INFO: Test Accuracy: 92.17% 2025-04-17 21:27:37 INFO: Test Accuracy: 93.09% 2025-04-17 21:28:41 INFO: Test Accuracy: 89.86% 2025-04-17 21:31:28 INFO: Test Accuracy: 93.55%
2025-04-17 22:40:19 INFO: Test Accuracy: 95.39% 2025-04-17 22:40:42 INFO: Test Accuracy: 96.31%

## Methodology

### Data Preprocessing

1. Load `input_train_dataset.csv` with 14 numeric features and binary target.
2. Standardize features using `StandardScaler`.
3. For GNN: build a k‑NN adjacency graph (k=5).
4. For CNN: reshape the 14-dimensional feature vector into a 1×14×1 tensor.

### Model Architectures

- **GNN**: Two- or three-layer `GCNConv` network with ReLU and Dropout.
- **CNN**: Two 2D convolutional layers (3×1 kernels) followed by two fully connected layers.

### Training & Logging

1. Use **Adam optimizer** and **CrossEntropyLoss**.
2. Train for 100–1000 epochs, logging loss every 20 epochs (GNN) or every 10 epochs (CNN).
3. Evaluate on held-out 20% test set, logging final accuracy.

## Technologies Used

- **Python & PyTorch** for model implementation.
- **PyTorch Geometric** for graph-based learning.
- **scikit-learn** for preprocessing and k‑NN graph construction.
- **Logging module** to record training metrics.

## Future Work

- **Hyperparameter Tuning**: Optimize learning rate, hidden sizes, and graph construction parameters.
- **Graph Enhancements**: Incorporate domain-driven edges (e.g., sequence similarity, structural data).
- **Model Variants**: Explore Graph Attention Networks (GAT) or deeper residual GCNs.
- **Cross-Validation**: Implement k-fold to obtain robust performance estimates.

## Conclusion

This repository demonstrates two complementary deep learning approaches—GNN and CNN—for Monkeypox epitope prediction. The CNN achieved up to 96.31% test accuracy, while the GNN consistently performed above 89%, showcasing the potential of both graph- and convolution-based methods in computational immunology.
