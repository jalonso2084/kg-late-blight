import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
from torch_geometric.data import Data
from gnn.train_gnn_advanced import TGCN 

# --- CONFIGURATION ---
CONFIG = {
    "num_node_features": 2, 
    "gcn_hidden_channels": 64,
    "lstm_hidden_channels": 32,
    "num_classes": 2,
    "pytorch_model_path": "gnn/gsage_v1.pt",
    "torchscript_model_path": "gnn/gsage_v1.ts"
}

def export_to_torchscript():
    """Loads a trained GNN and exports it to TorchScript format."""

    print(f"Loading trained PyTorch model from {CONFIG['pytorch_model_path']}...")

    model = TGCN(
        node_features=CONFIG["num_node_features"],
        gcn_hidden=CONFIG["gcn_hidden_channels"],
        lstm_hidden=CONFIG["lstm_hidden_channels"],
        num_classes=CONFIG["num_classes"]
    )
    model.load_state_dict(torch.load(CONFIG["pytorch_model_path"]))
    model.eval()

    # --- THIS IS THE FIX: Create a dummy input that matches the new model signature ---
    print("Creating a dummy input for tracing the model...")
    dummy_x_list = [torch.randn(24, CONFIG["num_node_features"]) for _ in range(7)]
    dummy_edge_list = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long) for _ in range(7)]
    dummy_batch_list = [torch.zeros(24, dtype=torch.long) for _ in range(7)]
    dummy_input = (dummy_x_list, dummy_edge_list, dummy_batch_list)

    # We can now use the simpler 'trace' method
    print(f"Exporting model to TorchScript at {CONFIG['torchscript_model_path']}...")
    traced_model = torch.jit.trace(model, dummy_input)

    traced_model.save(CONFIG["torchscript_model_path"])

    print("\nModel successfully exported to TorchScript.")
    print(f"File saved to: {CONFIG['torchscript_model_path']}")

    print("\nVerifying the TorchScript model...")
    loaded_ts_model = torch.jit.load(CONFIG["torchscript_model_path"])
    with torch.no_grad():
        output = loaded_ts_model(*dummy_input)
    print(f"Verification successful. Model loaded and produced output of shape: {output.shape}")

if __name__ == "__main__":
    export_to_torchscript()