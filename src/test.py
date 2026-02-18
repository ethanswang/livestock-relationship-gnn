import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Define graph dataset
edge_index = torch.tensor([
    [0, 1, 0, 2, 0, 4, 2, 4],
    [1, 0, 2, 0, 4, 0, 4, 2],
], dtype=torch.long)

# Define data features
node_features = torch.tensor([
    [25, 1],  # Person 0 (25 years old, likes sports)
    [30, 0],  # Person 1 (30 years old, does not like sports)
    [22, 1],  # Person 2 (22 years old, likes sports)
    [35, 0],  # Person 3 (35 years old, does not like sports)
    [27, 1],  # Person 4 (27 years old, likes sports)
], dtype=torch.float)
node_features = F.normalize(node_features, dim=0)
# Define dataset labels
num_friends = torch.tensor([3, 1, 2, 0, 3])
labels = (num_friends >= 2).long()

# Mask for separating training and testing data
train_mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
data = Data(x=node_features, edge_index=edge_index, y=labels, train_mask=train_mask)

# Define model
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim) 
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # Activation function
        x = self.conv2(x, edge_index)
        return x

# Instantiate model
model = GNN(input_dim=2, hidden_dim=4, output_dim=2)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Test model
model.eval()
with torch.no_grad():
    predictions = model(data).argmax(dim=1)

print("\nFinal Predictions (1=Popular, 0=Not Popular):", predictions.tolist())