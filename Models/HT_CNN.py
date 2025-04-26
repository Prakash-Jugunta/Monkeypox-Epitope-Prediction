# 1) Imports
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import logging

# 2) Logging configuration
logging.basicConfig(
    filename='../Results/Logs/cnn_HTT_training.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 3) Load dataset
df = pd.read_csv('../dataset/input_train_dataset.csv')

# 4) Feature and target selection
feature_cols = [
    'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker', 'isoelectric_point',
    'aromaticity', 'hydrophobicity', 'stability', 'charge', 'flexibility',
    'solvent_accessibility', 'blosum_score', 'ptm_sites', 'interaction_energy'
]
X = df[feature_cols].values
y = df['target'].values.astype(int)

# 5) Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7) Reshape input for CNN
X_scaled = X_scaled.reshape(-1, 1, 14, 1)

# 8) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 9) DataLoaders
train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                         torch.tensor(y_train, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                        torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# 10) Define CNN Model with increased dropout & reduced FC size
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1))
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        self.fc1 = nn.Linear(64 * 10, 64)  # Reduced FC size
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# 11) Instantiate model, loss, optimizer (with weight decay)
model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

# 12) Training loop (reduced epochs)
model.train()
for epoch in range(1, 61):  # Reduced to 60 epochs
    total_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logging.info(f'Epoch {epoch:03d} - Loss: {total_loss:.4f}')

# 13) Evaluation
model.eval()
correct, total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 14) Print metrics
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average='binary')
rec = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')
report = classification_report(all_labels, all_preds)

# Display
print(f"\nEvaluation Metrics:")
print(f"Accuracy       : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"\nClassification Report:\n{report}")

# Log metrics
logging.info(f"Test Accuracy       : {acc:.4f}")
logging.info(f"Test Precision      : {prec:.4f}")
logging.info(f"Test Recall         : {rec:.4f}")
logging.info(f"Test F1 Score       : {f1:.4f}")
logging.info(f"\nClassification Report:\n{report}")

# 15) Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
logging.info(f'Confusion Matrix:\n{cm}')

# 16) Visualize Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("../Results/Confusion_matrix/CNN_HTT_confusion_matrix.png")
plt.show()
