import time
import numpy as np
import onnxruntime as ort
import psutil
import os

# --- CONFIGURATION ---
CONFIG = {
    "onnx_model_path": "vision/mosvit_blite_int8.onnx",
    "image_size": 224,
    "num_runs": 100, # Number of times to run inference for averaging
    "output_file": "benchmark.txt"
}

def benchmark_model():
    """Loads and benchmarks the quantized ONNX model for speed and memory."""

    print(f"Loading quantized model: {CONFIG['onnx_model_path']}")

    # Set session options for CPU execution
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Create inference session
    ort_session = ort.InferenceSession(CONFIG['onnx_model_path'], sess_options=options)
    input_name = ort_session.get_inputs()[0].name

    # Create a dummy input tensor (1 image, 3 color channels, 224x224 size)
    dummy_input = np.random.randn(1, 3, CONFIG['image_size'], CONFIG['image_size']).astype(np.float32)

    # --- BENCHMARK LATENCY ---
    print(f"Running latency test ({CONFIG['num_runs']} iterations)...")
    # Warm-up run (the first run is often slower)
    ort_session.run(None, {input_name: dummy_input})

    start_time = time.perf_counter()
    for _ in range(CONFIG['num_runs']):
        ort_session.run(None, {input_name: dummy_input})
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / CONFIG['num_runs']) * 1000

    # --- BENCHMARK MEMORY ---
    process = psutil.Process(os.getpid())
    peak_ram_mb = process.memory_info().rss / (1024 * 1024) # Convert bytes to MB

    # --- SAVE RESULTS ---
    results = (
        f"--- Benchmark Results ---\n"
        f"Model: {CONFIG['onnx_model_path']}\n"
        f"Average Latency: {avg_latency_ms:.2f} ms / image\n"
        f"Peak RAM Usage: {peak_ram_mb:.2f} MB\n"
    )

    print(results)

    with open(CONFIG['output_file'], 'w') as f:
        f.write(results)
    print(f"Results saved to {CONFIG['output_file']}")

if __name__ == "__main__":
    benchmark_model()