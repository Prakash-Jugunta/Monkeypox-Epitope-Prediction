# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_curve, auc
)
from mpl_toolkits.mplot3d import Axes3D

# 2. Load your data
df = pd.read_csv('../dataset/input_train_dataset.csv') 

# 3. Select the same feature columns and target
feature_cols = [
    'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker', 'isoelectric_point',
    'aromaticity', 'hydrophobicity', 'stability', 'charge', 'flexibility',
    'solvent_accessibility', 'blosum_score', 'ptm_sites'
]
X = df[feature_cols].values
y = df['target'].values

# 4. Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Train the SVM
model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True
)
model.fit(X_train, y_train)

# 7. Evaluate on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%\n")

# 8. Detailed classification report
print(classification_report(y_test, y_pred, digits=4))

# 9. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 10. AUC-ROC Curve (for binary classification)
if len(np.unique(y)) == 2:
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
else:
    print("AUC-ROC is only applicable for binary classification.")

# 11. 3D Scatter Plot using 3 selected features
feat1, feat2, feat3 = 'charge', 'flexibility', 'hydrophobicity'
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df[feat1], df[feat2], df[feat3], 
    c=df['target'], cmap='viridis', edgecolor='k'
)
ax.set_xlabel(feat1)
ax.set_ylabel(feat2)
ax.set_zlabel(feat3)
plt.title('3D Scatter Plot by Target Class')
plt.legend(*scatter.legend_elements(), title="Classes")
plt.show()
