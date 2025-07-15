import os
import glob
import torch
import random
import wandb
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphSAGE, global_mean_pool # <-- ADDED global_mean_pool
from sklearn.metrics import f1_score

# --- CONFIGURATION ---
CONFIG = {
    "snapshot_path": "data/snapshots",
    "num_node_features": 2, # Temp, Humidity
    "num_classes": 2, # Low risk (0), High risk (1)
    "hidden_channels": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "project_name": "kg-late-blight-gnn",
    "run_name": "sprint_3_2_baseline_sage",
    "output_model_path": "gnn/gsage_v0.pt"
}

# --- GNN MODEL DEFINITION ---
class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GraphSAGE(in_channels, hidden_channels, num_layers=1, aggr='mean')
        self.conv2 = GraphSAGE(hidden_channels, out_channels, num_layers=1, aggr='mean')
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

def train():
    # --- DATA PREPARATION ---
    print("Loading graph snapshots...")
    all_snapshots = [torch.load(f, weights_only=False) for f in glob.glob(os.path.join(CONFIG["snapshot_path"], '*.pt'))]
    random.shuffle(all_snapshots)

    split_idx = int(len(all_snapshots) * 0.7)
    train_data = all_snapshots[:split_idx]
    val_data = all_snapshots[split_idx:]

    if not train_data:
        print("Error: No training data found. Make sure snapshots exist.")
        return

    train_loader = DataLoader(train_data, batch_size=2)
    val_loader = DataLoader(val_data, batch_size=2)
    print(f"Loaded {len(all_snapshots)} snapshots. Training with {len(train_data)}, validating with {len(val_data)}.")

    all_labels = torch.cat([data.y for data in train_data])
    class_counts = torch.bincount(all_labels)
    if len(class_counts) < CONFIG["num_classes"]:
        print("Warning: One class is missing in the training data. Using uniform weights.")
        class_weights = torch.ones(CONFIG["num_classes"])
    else:
        class_weights = 1. / class_counts.float()
        class_weights = class_weights / class_weights.sum()

    # --- MODEL SETUP ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GNNModel(
        in_channels=CONFIG["num_node_features"],
        hidden_channels=CONFIG["hidden_channels"],
        out_channels=CONFIG["num_classes"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)

    # --- TRAINING & VALIDATION LOOP ---
    best_val_f1 = 0.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            # --- THIS IS THE FIX: Use global_mean_pool for correct graph-level prediction ---
            graph_level_prediction = global_mean_pool(out, data.batch)
            loss = criterion(graph_level_prediction, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index)
                # --- THIS IS THE FIX: Use global_mean_pool here as well ---
                graph_level_prediction = global_mean_pool(out, data.batch)
                pred = graph_level_prediction.argmax(dim=1)
                all_preds.append(pred)
                all_true.append(data.y)

        if not all_true:
            print("Warning: No validation data to evaluate.")
            continue

        val_f1 = f1_score(torch.cat(all_true).cpu(), torch.cat(all_preds).cpu(), average='macro', zero_division=0)
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch: {epoch+1:02d}, Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}")
        wandb.log({"epoch": epoch, "loss": avg_loss, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), CONFIG["output_model_path"])
            print(f"  -> New best model saved with F1: {best_val_f1:.4f}")

    wandb.finish()
    print("\nTraining complete.")
    print(f"Best validation F1 was: {best_val_f1:.4f}")

if __name__ == "__main__":
    train()