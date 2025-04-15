import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
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

# Step 4: Apply SMOTE
print("Balancing data using SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Class distribution after SMOTE: {np.bincount(y_train_balanced)}")
# print(bcell_data.shape)
# Convert balanced data (after SMOTE) back to DataFrame for plotting
balanced_data = pd.DataFrame(X_train_balanced, columns=X_train.columns)
balanced_data['target'] = y_train_balanced
# print(balanced_data.shape)
# Separate epitopes and non-epitopes after SMOTE
epitopes_smote = balanced_data[balanced_data['target'] == 1]
non_epitopes_smote = balanced_data[balanced_data['target'] == 0]
print(epitopes_smote.shape)
print(non_epitopes_smote.shape)

# Scatter plot for epitopes and non-epitopes after SMOTE
plt.figure(figsize=(8, 6))

# Plot epitopes (red circles)
plt.scatter(
    epitopes_smote['chou_fasman'], epitopes_smote['kolaskar_tongaonkar'], 
    c='red', marker='o', label='Epitopes'
)

# Plot non-epitopes (blue crosses)
plt.scatter(
    non_epitopes_smote['chou_fasman'], non_epitopes_smote['kolaskar_tongaonkar'], 
    c='blue', marker='o',  label='Non-Epitopes'
)

# Add labels, legend, and title
plt.xlabel('Chou Fasman')
plt.ylabel('Kolaskar Tongaonkar')
plt.title('Scatter Plot After SMOTE: Epitope vs Non-Epitope')
plt.legend()
plt.show()


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

# Step 7: Cross-Validation and Hyperparameter Tuning
print("Starting cross-validation and hyperparameter tuning...")
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}
grid_search = GridSearchCV(
    SVC(probability=True, class_weight='balanced'), 
    param_grid, cv=3, scoring='roc_auc', verbose=1
)
grid_search.fit(X_train_pca, y_train_balanced)

best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Cross-Validation Score
print("Performing cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_train_pca, y_train_balanced, cv=cv, scoring='roc_auc')
print(f"Cross-Validation AUC Scores: {cv_scores}")
print(f"Mean CV AUC Score: {np.mean(cv_scores):.4f}")

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
plt.title('SMOTE MPOX data - SVM')
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

# 3D Scatter Plot
print("Creating 3D scatter plot...")
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], 
    c=y_pred_test, cmap='viridis', alpha=0.8
)
ax.set_title('3D Scatter Plot of Epitope Predictions')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
legend1 = ax.legend(
    *scatter.legend_elements(), 
    title="Predicted Labels", 
    loc='upper left'
)
ax.add_artist(legend1)
plt.show()



# Save Predictions for COVID Data
print("Predicting on MPOX data...")
X_covid = mokeypox_data.drop(columns=['parent_protein_id', 'protein_seq', 'peptide_seq'], errors='ignore')
X_covid_scaled = scaler.transform(X_covid)
X_covid_pca = pca.transform(X_covid_scaled)
covid_predictions = best_model.predict(X_covid_pca)

covid_epitopes = mokeypox_data[covid_predictions == 1]
covid_non_epitopes = mokeypox_data[covid_predictions == 0]


covid_epitopes.to_csv('Mpox_epitopes.csv', index=False)
covid_non_epitopes.to_csv('Mpox_non_epitopes.csv', index=False)

plt.figure(figsize=(8, 6))

# Plot epitopes with one color (e.g., red)
plt.scatter(covid_epitopes['chou_fasman'], covid_epitopes['kolaskar_tongaonkar'], 
            c='red', label='Epitopes', alpha=0.5,marker='o')

# Plot non-epitopes with another color (e.g., blue)
plt.scatter(covid_non_epitopes['chou_fasman'], covid_non_epitopes['kolaskar_tongaonkar'], 
            c='blue', label='Non-Epitopes',alpha=0.5,marker='x')

# Add labels, legend, and title
plt.xlabel('Chou Fasman')
plt.ylabel('Kolaskar Tongaonkar')
plt.title('Monkeypox Epitopes vs Non-Epitopes')
plt.legend()  # Display the legend to differentiate points

# Show the plot
plt.show()

print("Epitope predictions saved to 'MPOX_epitopes.csv'.")
print("Non-epitope predictions saved to 'MPOX_non_epitopes.csv'.")
print("Execution completed successfully!")