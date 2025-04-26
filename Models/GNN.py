# 1) Imports
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 2) Set up logging
logging.basicConfig(
    filename='../Results/Logs/gnn_training.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 3) Load your CSV into a pandas DataFrame
df = pd.read_csv('input_train_dataset.csv')

# 4) Select features and target
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

# 6) Build the k-NN graph
A = kneighbors_graph(X_scaled, n_neighbors=5, mode='connectivity', include_self=False)
coo = A.tocoo()
edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)

# 7) Convert to torch tensors
x = torch.tensor(X_scaled, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# 8) Train/test mask
num_nodes = x.size(0)
perm = torch.randperm(num_nodes)
train_size = int(0.8 * num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[perm[:train_size]] = True
test_mask = ~train_mask

# 9) PyG Data object
data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.test_mask = test_mask

# 10) Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# 11) Instantiate model, loss, optimizer
model = GCN(in_channels=x.size(1), hidden_channels=16, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 12) Training loop
model.train()
for epoch in range(1, 1001):  # 1000 epochs
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# 13) Evaluation and confusion matrix
model.eval()
with torch.no_grad():
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    total = data.test_mask.sum().item()
    accuracy = 100 * correct / total

    # Log accuracy
    logging.info(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Confusion Matrix
    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()
    cm = confusion_matrix(y_true, y_pred)
    logging.info(f'Confusion Matrix:\n{cm}')

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues')
    plt.title("GCN Confusion Matrix")
    plt.savefig("../Results/Confusion_matrix/GCN_confusion_matrix.png")
    plt.show()
