import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, 
    precision_score, recall_score, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Indicate execution has started
print("Execution started...")

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

# Step 6: Feature Importance with Random Forest
print("Using Random Forest for feature selection...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
)
rf.fit(X_train_scaled, y_train_balanced)
feature_importances = rf.feature_importances_
important_features_indices = np.argsort(feature_importances)[::-1][:10]  # Top 10 features

X_train_rf = X_train_scaled[:, important_features_indices]
X_test_rf = X_test_scaled[:, important_features_indices]

print(f"Top features selected: {important_features_indices}")

# Step 7: Dimensionality Reduction (PCA)
print("Applying PCA to reduced data...")
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_rf)
X_test_pca = pca.transform(X_test_rf)

# Step 8: Train SVM Model with Hyperparameter Tuning
print("Starting SVM training with GridSearchCV...")
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm = SVC(probability=True, random_state=42)
grid_search_svm = GridSearchCV(
    svm, param_grid_svm, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1
)
grid_search_svm.fit(X_train_pca, y_train_balanced)

best_svm = grid_search_svm.best_estimator_
print(f"Best parameters for SVM: {grid_search_svm.best_params_}")

# Step 9: Cross-Validation with the Hybrid Model
print("Performing cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_svm, X_train_pca, y_train_balanced, cv=cv, scoring='roc_auc')
print(f"Cross-Validation AUC Scores: {cv_scores}")
print(f"Mean CV AUC Score: {np.mean(cv_scores):.4f}")

# Step 10: Final Model Training
print("Training the final SVM model...")
best_svm.fit(X_train_pca, y_train_balanced)

# Step 11: Predictions and Evaluation
print("Evaluating the hybrid model on the test set...")
y_pred_test = best_svm.predict(X_test_pca)
y_prob_test = best_svm.predict_proba(X_test_pca)[:, 1]

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

# Step 12: Save COVID Data Predictions
print("Predicting on COVID data...")
X_covid = covid_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq'], errors='ignore')
X_covid_scaled = scaler.transform(X_covid)
X_covid_rf = X_covid_scaled[:, important_features_indices]
X_covid_pca = pca.transform(X_covid_rf)
covid_predictions = best_svm.predict(X_covid_pca)

covid_epitopes = covid_data[covid_predictions == 1]
covid_non_epitopes = covid_data[covid_predictions == 0]

covid_epitopes.to_csv('covid_epitopes_hybrid.csv', index=False)
covid_non_epitopes.to_csv('covid_non_epitopes_hybrid.csv', index=False)

print("Epitope predictions saved to 'covid_epitopes_hybrid.csv'.")
print("Non-epitope predictions saved to 'covid_non_epitopes_hybrid.csv'.")
print("Execution completed successfully!")