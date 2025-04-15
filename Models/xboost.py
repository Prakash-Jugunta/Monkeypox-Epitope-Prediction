import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb  # Import XGBoost

# Indicate that code execution has started
print("Starting execution...")

# Step 1: Load Datasets
print("Loading datasets...")
bcell_data = pd.read_excel('input_F13_train.xlsx')
mokeypox_data = pd.read_excel('input_F13_test.xlsx')
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

# Step 4: Scale Data
print("Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Dimensionality Reduction (PCA)
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Step 6: Cross-Validation and Hyperparameter Tuning with XGBoost
print("Starting cross-validation and hyperparameter tuning...")
param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0]
}
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)

grid_search = GridSearchCV(
    xgb_model, param_grid, cv=3, scoring='roc_auc', verbose=1
)
grid_search.fit(X_train_pca, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Cross-Validation Score
print("Performing cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_pca, y_train, cv=cv, scoring='roc_auc')
print(f"Cross-Validation AUC Scores: {cv_scores}")
print(f"Mean CV AUC Score: {np.mean(cv_scores):.4f}")

# Step 7: Train the Final Model
print("Training the final model...")
best_model.fit(X_train_pca, y_train)

# Step 8: Predictions and Evaluation
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
plt.title('Original MPOX data - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
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

# Save Predictions for MPOX Data
print("Predicting on MPOX data...")
X_covid = mokeypox_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq'], errors='ignore')
X_covid_scaled = scaler.transform(X_covid)
X_covid_pca = pca.transform(X_covid_scaled)
covid_predictions = best_model.predict(X_covid_pca)

covid_epitopes = mokeypox_data[covid_predictions == 1]
covid_non_epitopes = mokeypox_data[covid_predictions == 0]

covid_epitopes.to_csv('covid_epitopes.csv', index=False)
covid_non_epitopes.to_csv('covid_non_epitopes.csv', index=False)

print("Epitope predictions saved to 'Mpox_epitopes.csv'.")
print("Non-epitope predictions saved to 'Mpox_non_epitopes.csv'.")
print("Execution completed successfully!")
