import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
bcell_data = pd.read_excel('input_F13_train.xlsx')
covid_data = pd.read_excel('input_F13_test.xlsx')

# Prepare B-Cell data
X_bcell = bcell_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq', 'target'])
y_bcell = bcell_data['target']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_bcell, y_bcell, test_size=0.2, random_state=42, stratify=y_bcell)

# Balance the data using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Scale the data
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for SVM using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 'scale'],
    'kernel': ['rbf', 'poly']
}
grid_search = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_balanced_scaled, y_train_balanced)

# Train the optimized model
best_svm = grid_search.best_estimator_
best_svm.fit(X_train_balanced_scaled, y_train_balanced)

# Predictions and probabilities on the test set
y_pred_test_opt = best_svm.predict(X_test_scaled)
y_prob_test_opt = best_svm.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics on test set
accuracy_opt = accuracy_score(y_test, y_pred_test_opt)
precision_opt = precision_score(y_test, y_pred_test_opt, pos_label=1)
recall_opt = recall_score(y_test, y_pred_test_opt, pos_label=1)
auc_opt = roc_auc_score(y_test, y_prob_test_opt)

print("\nOptimized Evaluation Metrics (Test Set):")
print(f"Accuracy: {accuracy_opt:.4f}")
print(f"Precision (Epitope): {precision_opt:.4f}")
print(f"Recall (Epitope): {recall_opt:.4f}")
print(f"AUC: {auc_opt:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test_opt, target_names=['Non-Epitope', 'Epitope']))

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred_test_opt)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Epitope', 'Epitope'], yticklabels=['Non-Epitope', 'Epitope'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve Visualization
fpr, tpr, _ = roc_curve(y_test, y_prob_test_opt)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Test Set AUC = {auc_opt:.2f}")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)
X_train_pca = pca.fit_transform(X_train_balanced_scaled)

# 2D Scatter Plot with Decision Boundary
plt.figure(figsize=(8, 6))
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = best_svm.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, style=y_pred_test_opt, palette='viridis')
plt.title('Epitope vs Non-Epitope Decision Boundary')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title="Legend", loc='best')
plt.show()

# 3D Scatter Plot
pca_3d = PCA(n_components=3)
X_test_3d = pca_3d.fit_transform(X_test_scaled)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=y_pred_test_opt, cmap='coolwarm', alpha=0.7)
plt.title('3D Scatter Plot of Epitope Predictions')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.colorbar(scatter, label='Prediction (0 = Non-Epitope, 1 = Epitope)')
plt.show()

# Predict on COVID data
X_covid = covid_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq'], errors='ignore')
X_covid_scaled = scaler.transform(X_covid)
covid_predictions = best_svm.predict(X_covid_scaled)

# Separate Epitope and Non-Epitope Predictions
covid_epitopes = covid_data[covid_predictions == 1]
covid_non_epitopes = covid_data[covid_predictions == 0]

# Save results to files
covid_epitopes.to_csv('covid_epitopes.csv', index=False)
covid_non_epitopes.to_csv('covid_non_epitopes.csv')

print("Epitope predictions have been saved to 'covid_epitopes.csv'.")
print("Non-epitope predictions have been saved to 'covid_non_epitopes.csv'.")
