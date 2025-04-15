import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Load datasets
bcell_data = pd.read_excel('input_F13_train.xlsx')
covid_data = pd.read_excel('input_F13_test.xlsx')

# Prepare B-Cell data
X_bcell = bcell_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq', 'target'])
y_bcell = bcell_data['target']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_bcell, y_bcell, test_size=0.2, random_state=42, stratify=y_bcell
)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model with hyperparameter tuning
svm = SVC(C=100, gamma='scale', kernel='rbf', probability=True)
svm.fit(X_train_scaled, y_train)

# Predictions and probabilities on the test set
y_pred_test = svm.predict(X_test_scaled)
y_prob_test = svm.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics on the test set
accuracy = accuracy_score(y_test, y_pred_test)
precision_pos = precision_score(y_test, y_pred_test, pos_label=1)
precision_neg = precision_score(y_test, y_pred_test, pos_label=0)
recall_pos = recall_score(y_test, y_pred_test, pos_label=1)
recall_neg = recall_score(y_test, y_pred_test, pos_label=0)
auc = roc_auc_score(y_test, y_prob_test)

# Print evaluation metrics
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

# 3. 2D Scatter Plot for Epitope and Non-Epitope Regions
plt.figure(figsize=(8, 6))
plt.scatter(X_test_scaled[:, 0][y_test == 0], X_test_scaled[:, 1][y_test == 0],
            c='blue', label='Non-Epitope', alpha=0.6)
plt.scatter(X_test_scaled[:, 0][y_test == 1], X_test_scaled[:, 1][y_test == 1],
            c='red', label='Epitope', alpha=0.6)
plt.title('2D Scatter Plot: Epitope vs Non-Epitope')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# 4. 3D Scatter Plot for Epitope and Non-Epitope Regions
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test_scaled[:, 0][y_test == 0], X_test_scaled[:, 1][y_test == 0],
           X_test_scaled[:, 2][y_test == 0], c='blue', label='Non-Epitope', alpha=0.6)
ax.scatter(X_test_scaled[:, 0][y_test == 1], X_test_scaled[:, 1][y_test == 1],
           X_test_scaled[:, 2][y_test == 1], c='red', label='Epitope', alpha=0.6)
ax.set_title('3D Scatter Plot: Epitope vs Non-Epitope')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.show()

# Predict on COVID data
X_covid = covid_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq'], errors='ignore')
X_covid_scaled = scaler.transform(X_covid)
covid_predictions = svm.predict(X_covid_scaled)

# Separate Epitope and Non-Epitope Predictions
covid_epitopes = covid_data[covid_predictions == 1]
covid_non_epitopes = covid_data[covid_predictions == 0]

# Save results to files
covid_epitopes.to_csv('covid_epitopes.csv', index=False)
covid_non_epitopes.to_csv('covid_non_epitopes.csv', index=False)

# 5. Distribution of Predictions
plt.figure(figsize=(6, 4))
sns.countplot(x=covid_predictions, palette='viridis')
plt.xticks([0, 1], ['Non-Epitope', 'Epitope'])
plt.title('Distribution of Predictions on COVID Data')
plt.xlabel('Prediction')
plt.ylabel('Count')
plt.show()

print("\nEpitope predictions saved to 'covid_epitopes.csv'.")
print("Non-epitope predictions saved to 'covid_non_epitopes.csv'.")
