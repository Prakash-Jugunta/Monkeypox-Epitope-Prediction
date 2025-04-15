import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import shap

# Indicate that code execution has started
print("Starting execution...")

# Step 1: Load Datasets
print("Loading datasets...")
bcell_data = pd.read_excel('input_F13_train.xlsx')
covid_data = pd.read_excel('input_F13_test.xlsx')
print("Datasets loaded successfully!")

# Step 2: Prepare B-Cell Data
print("Preparing B-Cell data...")
X_bcell = bcell_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq', 'target'])
y_bcell = bcell_data['target']

# Step 3: Train-Test Split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_bcell, y_bcell, test_size=0.2, random_state=42, stratify=y_bcell
)

# Step 4: Apply SMOTE
print("Balancing data using SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Class distribution after SMOTE: {np.bincount(y_train_balanced)}")

# Step 5: Scale Data
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Step 6: Dimensionality Reduction (PCA)
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Step 7: Hyperparameter Tuning with GridSearchCV
print("Starting hyperparameter tuning...")
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}
grid_search = GridSearchCV(
    SVC(probability=True, class_weight='balanced'), param_grid, cv=3, scoring='roc_auc'
)
with tqdm(total=len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])) as pbar:
    def update_progress(*args):
        pbar.update(1)
    grid_search.fit(X_train_pca, y_train_balanced)
    
best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Step 8: Train the Final Model
print("Training the final model...")
best_model.fit(X_train_pca, y_train_balanced)

# Step 9: Predictions and Evaluation
print("Evaluating the model on the test set...")
y_pred_test = best_model.predict(X_test_pca)
y_prob_test = best_model.predict_proba(X_test_pca)[:, 1]

# Metrics Calculation
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

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Epitope', 'Epitope'], yticklabels=['Non-Epitope', 'Epitope'])
plt.title('Confusion Matrix: Epitope vs Non-Epitope')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()

# ROC Curve Visualization
fpr, tpr, _ = roc_curve(y_test, y_prob_test)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Test Set AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Step 10: SHAP Analysis
print("Performing SHAP analysis...")
explainer = shap.Explainer(best_model, X_train_pca)
shap_values = explainer(X_test_pca)

# Summary Plot
print("Creating SHAP summary plot...")
shap.summary_plot(shap_values, X_test_pca, feature_names=[f"PC{i+1}" for i in range(X_test_pca.shape[1])])

# Force Plot for a Single Prediction
print("Creating SHAP force plot for the first prediction...")
shap.force_plot(explainer.expected_value[0], shap_values[0].values, X_test_pca[0], feature_names=[f"PC{i+1}" for i in range(X_test_pca.shape[1])], matplotlib=True)

# Save Predictions for COVID Data
print("Predicting on COVID data...")
X_covid = covid_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq'], errors='ignore')
X_covid_scaled = scaler.transform(X_covid)
X_covid_pca = pca.transform(X_covid_scaled)
covid_predictions = best_model.predict(X_covid_pca)

covid_epitopes = covid_data[covid_predictions == 1]
covid_non_epitopes = covid_data[covid_predictions == 0]

covid_epitopes.to_csv('covid_epitopes.csv', index=False)
covid_non_epitopes.to_csv('covid_non_epitopes.csv', index=False)

print("Epitope predictions saved to 'covid_epitopes.csv'.")
print("Non-epitope predictions saved to 'covid_non_epitopes.csv'.")
print("Execution completed successfully!")