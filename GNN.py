import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import logging

# 1) Set up logging
logging.basicConfig(
    filename='training.log',  # Log will be saved to this file
    filemode='a',             # Overwrite each time; use 'a' to append
    level=logging.INFO,       # Log all levels INFO and above
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 2) Load your CSV into a pandas DataFrame
df = pd.read_csv('input_train_dataset.csv')

# 3) Select features (only the specified columns) and target
feature_cols = [
    'chou_fasman', 'emini', 'kolaskar_tongaonkar', 'parker', 'isoelectric_point',
    'aromaticity', 'hydrophobicity', 'stability', 'charge', 'flexibility',
    'solvent_accessibility', 'blosum_score', 'ptm_sites', 'interaction_energy'
]
X = df[feature_cols].values
y = df['target'].values.astype(int)

# 4) Normalize features (important for GCNs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5) Build the k‑NN graph (e.g., k=5)
A = kneighbors_graph(X_scaled, n_neighbors=5, mode='connectivity', include_self=False)
coo = A.tocoo()
edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long)

# 6) Convert features and labels to torch.Tensors
x = torch.tensor(X_scaled, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# 7) Create train/test masks (80% train, 20% test)
num_nodes = x.size(0)
perm = torch.randperm(num_nodes)
train_size = int(0.8 * num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[perm[:train_size]] = True
test_mask = ~train_mask

# 8) Create PyG Data object
data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.test_mask = test_mask

# 9) Define the GCN Model
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

# 10) Instantiate the model, loss function, and optimizer
model = GCN(in_channels=x.size(1), hidden_channels=16, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 11) Training loop with logging for accuracy
model.train()
for epoch in range(1, 1000):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


# 12) Evaluation with accuracy logging
model.eval()
with torch.no_grad():
    out = model(data)
    pred = out.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
    total = data.test_mask.sum().item()
    accuracy = 100 * correct / total
    logging.info(f'Test Accuracy: {accuracy:.2f}%')  # Log the accuracy
    print(f'Test Accuracy: {accuracy:.2f}%')
