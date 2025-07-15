import os
import glob
import torch
import random
import wandb
import collections
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import f1_score

# --- CONFIGURATION ---
CONFIG = {
    "snapshot_path": "data/snapshots",
    "num_node_features": 2, # Temp, Humidity
    "num_classes": 2, # Low risk (0), High risk (1)
    "gcn_hidden_channels": 64,
    "lstm_hidden_channels": 32,
    "days_in_snapshot": 7,
    "epochs": 100,
    "learning_rate": 0.001,
    "project_name": "kg-late-blight-gnn",
    "run_name": "sprint_3_3_T-GCN",
    "output_model_path": "gnn/gsage_v1.pt"
}

# --- T-GCN MODEL DEFINITION (with simplified input) ---
class TGCN(torch.nn.Module):
    def __init__(self, node_features, gcn_hidden, lstm_hidden, num_classes):
        super().__init__()
        self.gcn1 = GCNConv(node_features, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.lstm = torch.nn.LSTM(gcn_hidden, lstm_hidden, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(lstm_hidden, num_classes)

    def forward(self, x_list: list[torch.Tensor], edge_index_list: list[torch.Tensor], batch_list: list[torch.Tensor]):
        graph_embeddings = []
        # The loop now iterates over simple tensors
        for i in range(len(x_list)):
            x, edge_index, batch = x_list[i], edge_index_list[i], batch_list[i]
            x = self.gcn1(x, edge_index).relu()
            x = self.gcn2(x, edge_index).relu()
            graph_embedding = global_mean_pool(x, batch)
            graph_embeddings.append(graph_embedding)

        x = torch.cat(graph_embeddings, dim=0).unsqueeze(0)
        x, (hn, cn) = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

def create_sequences(all_snapshots, sequence_length=7):
    snapshots_by_field = collections.defaultdict(list)
    for s in all_snapshots:
        filename = os.path.basename(s.path)
        field_id = '_'.join(filename.split('_')[:-1])
        snapshots_by_field[field_id].append(s)

    sequences = []
    for field_id, snapshots in snapshots_by_field.items():
        snapshots.sort(key=lambda s: os.path.basename(s.path).split('_')[-1], reverse=True)
        if len(snapshots) >= sequence_length:
            for i in range(len(snapshots) - sequence_length + 1):
                sequence = snapshots[i:i + sequence_length]
                label = sequence[-1].y
                sequences.append({"data": sequence, "label": label})
    return sequences

def train_advanced_model():
    print("Loading graph snapshots for sequencing...")
    snapshot_files = glob.glob(os.path.join(CONFIG["snapshot_path"], '*.pt'))
    all_snapshots = [torch.load(f, weights_only=False) for f in snapshot_files]

    for i, f_path in enumerate(snapshot_files):
        all_snapshots[i].path = f_path

    sequences = create_sequences(all_snapshots, sequence_length=CONFIG["days_in_snapshot"])
    random.shuffle(sequences)

    if not sequences:
        print("Not enough snapshots to create sequences. Need at least 7 per field.")
        return

    split_idx = int(len(sequences) * 0.7)
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]
    print(f"Created {len(sequences)} sequences. Training with {len(train_sequences)}, validating with {len(val_sequences)}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TGCN(
        node_features=CONFIG["num_node_features"],
        gcn_hidden=CONFIG["gcn_hidden_channels"],
        lstm_hidden=CONFIG["lstm_hidden_channels"],
        num_classes=CONFIG["num_classes"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()

    wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)

    print("Starting T-GCN training...")
    best_val_f1 = 0.0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        total_loss = 0
        for seq_item in train_sequences:
            seq_data = seq_item["data"]
            label = seq_item["label"].to(device)

            # --- THIS IS THE FIX: Unpack data into simple lists of tensors ---
            x_list = [s.x.to(device) for s in seq_data]
            edge_index_list = [s.edge_index.to(device) for s in seq_data]
            batch_list = [s.batch.to(device) if s.batch is not None else torch.zeros(s.x.shape[0], dtype=torch.long, device=device) for s in seq_data]

            optimizer.zero_grad()
            out = model(x_list, edge_index_list, batch_list)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_sequences) if train_sequences else 0

        model.eval()
        all_preds, all_true = [], []
        if val_sequences:
            with torch.no_grad():
                for seq_item in val_sequences:
                    seq_data = seq_item["data"]
                    label = seq_item["label"].to(device)

                    x_list = [s.x.to(device) for s in seq_data]
                    edge_index_list = [s.edge_index.to(device) for s in seq_data]
                    batch_list = [s.batch.to(device) if s.batch is not None else torch.zeros(s.x.shape[0], dtype=torch.long, device=device) for s in seq_data]

                    out = model(x_list, edge_index_list, batch_list)
                    pred = out.argmax(dim=1)
                    all_preds.append(pred)
                    all_true.append(label)

            val_f1 = f1_score(torch.cat(all_true).cpu(), torch.cat(all_preds).cpu(), average='macro', zero_division=0)
        else:
            val_f1 = 0

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
    train_advanced_model()