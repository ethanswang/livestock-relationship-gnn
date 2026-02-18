# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

A Python virtual environment is located at `env/`. Activate it before running scripts:

```bash
source env/bin/activate
```

## Running Scripts

```bash
# Train the GNN model (100 epochs, prints predictions)
python src/test.py

# Visualize the social network graph
python src/test_visualization.py

# Inspect the CSV data
python src/data.py
```

## Architecture

This is a Graph Neural Network (GNN) experiment for node classification on a social network.

**Data flow:**
- `data/data.csv` — raw relationship data (id1, id2, relationship strength)
- `src/data.py` — loads CSV with pandas (currently independent of the model)
- `src/test.py` — defines a synthetic 5-node graph with hardcoded features/edges, implements a 2-layer GCN using PyTorch Geometric, trains with Adam (lr=0.01) for 100 epochs to classify nodes as "Popular" (≥2 friends) or not
- `src/test_visualization.py` — imports model and graph data from `test.py`, converts `edge_index` tensor to NetworkX format, and renders with Matplotlib

**Model:** `GCN` class in `src/test.py` — two `GCNConv` layers (input=2, hidden=4, output=2) with ReLU between them.

## Key Dependencies

| Package | Version |
|---|---|
| PyTorch | 2.8.0 |
| PyTorch Geometric | 2.6.1 |
| NetworkX | 3.2.1 |
| Matplotlib | 3.9.4 |
| Pandas | 2.3.3 |

Note: `data/data.csv` is not yet wired into the GNN model — `src/test.py` uses hardcoded synthetic graph data instead.
