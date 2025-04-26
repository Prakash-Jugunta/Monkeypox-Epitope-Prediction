import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import logging

# Configure logging (append mode)
logging.basicConfig(
    filename='../Results/Logs/Hybrid_XGBoost_training.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load CSV into DataFrame
df = pd.read_csv('../dataset/input_train_dataset.csv')

# Select features and target (excluding interaction_energy)
feature_cols = [
    'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker', 'isoelectric_point',
    'aromaticity', 'hydrophobicity', 'stability', 'charge', 'flexibility',
    'solvent_accessibility', 'blosum_score', 'ptm_sites'
]
X = df[feature_cols].values
y = df['target'].values.astype(int)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN: [samples, channels, height, width]
X_scaled = X_scaled.reshape(-1, 1, 13, 1)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# DataLoaders
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long))
test_ds  = TensorDataset(torch.tensor(X_test,  dtype=torch.float32),
                         torch.tensor(y_test,  dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

# Define the CNN Model for Feature Extraction
class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1))
        self.fc1 = nn.Linear(64 * 9, 128)  # Adjusted for 13 features

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))

# Instantiate the CNN model
cnn_model = CNNFeatureExtractor()

# Train the CNN model
cnn_model.eval()  # Switch to evaluation mode since we're using CNN for feature extraction

# Extract features from the training data
train_features = []
train_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        features = cnn_model(images)
        train_features.append(features)
        train_labels.append(labels)

train_features = torch.cat(train_features).numpy()
train_labels = torch.cat(train_labels).numpy()

# Extract features from the test data
test_features = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        features = cnn_model(images)
        test_features.append(features)
        test_labels.append(labels)

test_features = torch.cat(test_features).numpy()
test_labels = torch.cat(test_labels).numpy()

# Define the Ensemble Model: XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the Ensemble Model (XGBoost) on features extracted by CNN
xgb_model.fit(train_features, train_labels)

# Evaluate the XGBoost model
y_pred = xgb_model.predict(test_features)
accuracy = (y_pred == test_labels).mean() * 100

msg = f'Test Accuracy (XGBoost): {accuracy:.2f}%'
print(msg)
logging.info(msg)

# Confusion Matrix
cm = confusion_matrix(test_labels, y_pred)
logging.info(f'Confusion Matrix:\n{cm}')

# Visualize Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (XGBoost)")
plt.savefig("../Results/Confusion_matrix/Hybrid_XGBoost_confusion_matrix.png")