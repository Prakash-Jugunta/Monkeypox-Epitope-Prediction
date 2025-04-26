# 1. Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 2. Load your data
df = pd.read_csv('../dataset/input_train_dataset.csv') 

# 3. Select the same feature columns and target
feature_cols = [
    'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker', 'isoelectric_point',
    'aromaticity', 'hydrophobicity', 'stability', 'charge', 'flexibility',
    'solvent_accessibility', 'blosum_score', 'ptm_sites', 'interaction_energy'
]
X = df[feature_cols].values        # shape: [n_samples, n_features]
y = df['target'].values            # shape: [n_samples]

# 4. Split into train & test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y         # preserve the ratio of classes in both sets
)

# 5. Standardize features (important for SVMs!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 6. Instantiate & train the SVM
model = SVC(
    kernel='rbf',     # radial‑basis‑function kernel
    C=1.0,            # regularization strength
    gamma='scale',    # kernel coefficient
    probability=True  # if you want probability estimates
)
model.fit(X_train, y_train)

# 7. Evaluate on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy*100:.2f}%\n")

# 8. Detailed classification report
print(classification_report(y_test, y_pred, digits=4))
