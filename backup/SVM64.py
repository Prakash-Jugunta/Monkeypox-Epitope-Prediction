import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate

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

# Initialize models
print("Initializing models...")
models = {
    "SVM": SVC(probability=True, class_weight='balanced', random_state=42, C=10, gamma='scale', kernel='rbf'),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
    "SVM-RF Hybrid": None,  # Placeholder for the hybrid approach
    "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate models
results = []
best_model = None
best_auc = 0

for model_name, model in models.items():
    print(f"Training {model_name}...")
   
    if model_name == "SVM-RF Hybrid":
        # Train SVM
        svm = models["SVM"]
        svm.fit(X_train_pca, y_train_balanced)
        svm_proba = svm.predict_proba(X_train_pca)[:, 1]
       
        # Use SVM's probabilities as input to Random Forest
        rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
        rf.fit(svm_proba.reshape(-1, 1), y_train_balanced)
        model = rf  # Update model to RF for testing phase
        X_test_hybrid = svm.predict_proba(X_test_pca)[:, 1].reshape(-1, 1)
        y_prob_test = rf.predict_proba(X_test_hybrid)[:, 1]
        y_pred_test = rf.predict(X_test_hybrid)
    else:
        # Train the model
        model.fit(X_train_pca, y_train_balanced)
        y_prob_test = model.predict_proba(X_test_pca)[:, 1]
        y_pred_test = model.predict(X_test_pca)
   
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred_test)
    precision_pos = precision_score(y_test, y_pred_test, pos_label=1)
    recall_pos = recall_score(y_test, y_pred_test, pos_label=1)
    auc = roc_auc_score(y_test, y_prob_test)
   
    results.append([model_name, accuracy, precision_pos, recall_pos, auc])
   
    # Update best model
    if auc > best_auc:
        best_auc = auc
        best_model = model

# Display results
print("\nModel Evaluation Metrics:")
print(tabulate(results, headers=["Model", "Accuracy", "Precision (Epitope)", "Recall (Epitope)", "AUC"], tablefmt="grid"))

# Step 7: Predict on unseen COVID data using the best model
print(f"\nBest model: {results[np.argmax([row[4] for row in results])][0]}")
print("Predicting on COVID data...")
X_covid = covid_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq'], errors='ignore')
X_covid_scaled = scaler.transform(X_covid)
X_covid_pca = pca.transform(X_covid_scaled)

if best_model == models["SVM-RF Hybrid"]:
    # Handle hybrid model separately
    svm = models["SVM"]
    X_covid_hybrid = svm.predict_proba(X_covid_pca)[:, 1].reshape(-1, 1)
    covid_predictions = best_model.predict(X_covid_hybrid)
else:
    covid_predictions = best_model.predict(X_covid_pca)

# Save predictions
covid_epitopes = covid_data[covid_predictions == 1]
covid_non_epitopes = covid_data[covid_predictions == 0]

covid_epitopes.to_csv('covid_epitopes.csv', index=False)
covid_non_epitopes.to_csv('covid_non_epitopes.csv', index=False)

print("Epitope predictions saved to 'covid_epitopes.csv'.")
print("Non-epitope predictions saved to 'covid_non_epitopes.csv'.")
print("Execution completed successfully!")