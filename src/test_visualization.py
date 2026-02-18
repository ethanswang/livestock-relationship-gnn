import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# Load the processed graph object
saved = torch.load("./data/data.pt", weights_only=True)
data = Data(
    edge_index=saved["edge_index"],
    edge_attr=saved["edge_attr"],
    num_nodes=saved["num_nodes"],
)

# Build a weighted NetworkX graph from edge_index and edge_attr
G = nx.Graph()
G.add_nodes_from(range(data.num_nodes))

edges_with_weights = [
    (src.item(), dst.item(), {"weight": w.item()})
    for (src, dst), w in zip(data.edge_index.t(), data.edge_attr)
]
G.add_edges_from(edges_with_weights)

# Layout and draw
pos = nx.spring_layout(G, seed=42)
weights = [G[u][v]["weight"] for u, v in G.edges()]

plt.figure(figsize=(10, 10))
nx.draw_networkx(
    G, pos,
    with_labels=True,
    node_color="lightblue",
    edge_color=weights,
    edge_cmap=plt.cm.Blues,
    width=2,
    node_size=500,
    font_size=7,
)
plt.title(f"  ({data.num_nodes} nodes, {data.edge_index.shape[1]} edges, threshold ≥ 0.25)")
plt.axis("off")
plt.tight_layout()
plt.show()
