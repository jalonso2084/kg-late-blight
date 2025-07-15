import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import torch
import time
import psutil
from torch_geometric.data import Data

# --- CONFIGURATION ---
CONFIG = {
    "torchscript_model_path": "gnn/gsage_v1.ts",
    "num_node_features": 2,
    "num_runs": 100,
    "output_file": "benchmark_gnn.txt"
}

def benchmark_gnn_model():
    """Loads and benchmarks the TorchScript GNN model."""

    # --- BENCHMARK MEMORY ---
    # Get memory usage before loading the model
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    print(f"Loading TorchScript model: {CONFIG['torchscript_model_path']}")
    loaded_ts_model = torch.jit.load(CONFIG['torchscript_model_path'])

    mem_after = process.memory_info().rss / (1024 * 1024)
    model_ram_mb = mem_after - mem_before

    # Create a dummy input for inference
    dummy_x_list = [torch.randn(24, CONFIG["num_node_features"]) for _ in range(7)]
    dummy_edge_list = [torch.tensor([[0, 1], [1, 0]], dtype=torch.long) for _ in range(7)]
    dummy_batch_list = [torch.zeros(24, dtype=torch.long) for _ in range(7)]
    dummy_input = (dummy_x_list, dummy_edge_list, dummy_batch_list)

    # --- BENCHMARK LATENCY ---
    print(f"Running latency test ({CONFIG['num_runs']} iterations)...")
    # Warm-up run
    with torch.no_grad():
        loaded_ts_model(*dummy_input)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(CONFIG['num_runs']):
            loaded_ts_model(*dummy_input)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / CONFIG['num_runs']) * 1000

    # --- SAVE RESULTS ---
    # Get file size
    file_size_mb = os.path.getsize(CONFIG['torchscript_model_path']) / (1024 * 1024)

    results = (
        f"--- GNN Benchmark Results ---\n"
        f"Model: {CONFIG['torchscript_model_path']}\n"
        f"File Size: {file_size_mb:.2f} MB\n"
        f"Average Latency: {avg_latency_ms:.2f} ms / sequence\n"
        f"Model RAM Usage: {model_ram_mb:.2f} MB\n"
    )

    print(results)

    with open(CONFIG['output_file'], 'w') as f:
        f.write(results)
    print(f"Results saved to {CONFIG['output_file']}")


if __name__ == "__main__":
    benchmark_gnn_model()