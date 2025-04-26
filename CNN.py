import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import logging

# Configure logging (append mode)
logging.basicConfig(
    filename='Results/Logs/boom_training.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load CSV into DataFrame
df = pd.read_csv('input_train_dataset.csv')

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

# Define the CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,1))
        self.fc1   = nn.Linear(64 * 9, 128)  # Adjusted for 13 features
        self.fc2   = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate model, loss, optimizer
model   = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.CrossEntropyLoss()

# Training loop with logging
model.train()
for epoch in range(1, 101):
    total_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logging.info(f'Epoch {epoch:03d} - Loss: {total_loss:.4f}')

# Evaluation with accuracy and confusion matrix
model.eval()
correct, total = 0, 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
msg = f'Test Accuracy: {accuracy:.2f}%'
print(msg)
logging.info(msg)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
logging.info(f'Confusion Matrix:\n{cm}')

# Visualize Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("Results/Confusion_matrix/boom_confusion_matrix.png")