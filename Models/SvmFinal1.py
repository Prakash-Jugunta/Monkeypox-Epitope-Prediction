import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from mpl_toolkits.mplot3d import Axes3D
import time

# Load datasets
bcell_data = pd.read_excel('input_F13_train.xlsx')
unseen_data = pd.read_excel('input_F13_test.xlsx')  # Unseen dataset

# Handle missing values
# Separate numeric and non-numeric columns
numeric_columns = bcell_data.select_dtypes(include=[np.number]).columns
non_numeric_columns = bcell_data.select_dtypes(exclude=[np.number]).columns

# Fill missing values for numeric columns with the mean
bcell_data[numeric_columns] = bcell_data[numeric_columns].fillna(bcell_data[numeric_columns].mean())

# Feature-target separation
X_bcell = bcell_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq', 'target'])
y_bcell = bcell_data['target']

# Address multicollinearity (correlated features)
correlation_matrix = X_bcell.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_correlation_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]
X_bcell = X_bcell.drop(columns=high_correlation_features)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_bcell, y_bcell, test_size=0.2, random_state=42, stratify=y_bcell
)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality reduction using PCA
pca = PCA(n_components=3, random_state=42)  # Keep 3 components for visualization
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Hyperparameter optimization using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train_pca, y_train)

best_svm = grid_search.best_estimator_

# Cross-validation
cv_scores = cross_val_score(best_svm, X_train_pca, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Model evaluation on test set
y_pred_test = best_svm.predict(X_test_pca)
y_prob_test = best_svm.predict_proba(X_test_pca)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred_test)
precision_pos = precision_score(y_test, y_pred_test, pos_label=1)
precision_neg = precision_score(y_test, y_pred_test, pos_label=0)
recall_pos = recall_score(y_test, y_pred_test, pos_label=1)
recall_neg = recall_score(y_test, y_pred_test, pos_label=0)
auc = roc_auc_score(y_test, y_prob_test)

print("\nEvaluation Metrics (Test Set):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Positive - Epitope): {precision_pos:.4f}")
print(f"Precision (Negative - Non-Epitope): {precision_neg:.4f}")
print(f"Recall (Positive - Epitope): {recall_pos:.4f}")
print(f"Recall (Negative - Non-Epitope): {recall_neg:.4f}")
print(f"AUC: {auc:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['Non-Epitope', 'Epitope']))

# --- Visualization Section ---
# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Epitope', 'Epitope'],
            yticklabels=['Non-Epitope', 'Epitope'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# 3. 2D Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_test_pca[:, 0][y_test == 0], X_test_pca[:, 1][y_test == 0],
            c='blue', label='Non-Epitope', alpha=0.6)
plt.scatter(X_test_pca[:, 0][y_test == 1], X_test_pca[:, 1][y_test == 1],
            c='red', label='Epitope', alpha=0.6)
plt.title('2D Scatter Plot: Epitope vs Non-Epitope')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# 4. 3D Scatter Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test_pca[:, 0][y_test == 0], X_test_pca[:, 1][y_test == 0],
           X_test_pca[:, 2][y_test == 0], c='blue', label='Non-Epitope', alpha=0.6)
ax.scatter(X_test_pca[:, 0][y_test == 1], X_test_pca[:, 1][y_test == 1],
           X_test_pca[:, 2][y_test == 1], c='red', label='Epitope', alpha=0.6)
ax.set_title('3D Scatter Plot: Epitope vs Non-Epitope')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
ax.legend()
plt.show()
