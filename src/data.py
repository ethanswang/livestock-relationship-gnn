import pandas as pd
import torch
from torch_geometric.utils import to_torch_csr_tensor

# Load raw edge list
df = pd.read_csv("./data/data.csv")

# Drop self-loops and edges below relationship threshold
df = df[df["id1"] != df["id2"]]
df = df[df["relationship"] >= 0.25]

# Remap node IDs to contiguous range starting at 0
all_ids = pd.concat([df["id1"], df["id2"]]).unique()
id_map = {old: new for new, old in enumerate(sorted(all_ids))}
df["id1"] = df["id1"].map(id_map)
df["id2"] = df["id2"].map(id_map)
num_nodes = len(id_map)

src = torch.tensor(df["id1"].values, dtype=torch.long)
dst = torch.tensor(df["id2"].values, dtype=torch.long)
edge_attr = torch.tensor(df["relationship"].values, dtype=torch.float)

edge_index = torch.stack([src, dst], dim=0)  # shape [2, num_edges]

# Sparse adjacency matrix (CSR)
adj = to_torch_csr_tensor(edge_index, edge_attr=edge_attr, size=num_nodes)

torch.save({
    "edge_index": edge_index,
    "edge_attr": edge_attr,
    "num_nodes": num_nodes,
    "adj": adj,
}, "./data/data.pt")

print(f"Nodes (remapped) : {num_nodes}  (contiguous 0..{num_nodes - 1})")
print(f"Edges : {edge_index.shape[1]}")
print(f"edge_index : {edge_index.shape}  dtype={edge_index.dtype}")
print(f"edge_attr  : {edge_attr.shape}   dtype={edge_attr.dtype}")
print(f"adj (CSR)  : {adj.shape}")
print("Saved → data/data.pt")
