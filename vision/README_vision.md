\# Phase 2 - Vision Model



This directory contains the scripts related to training, optimizing, and using the computer vision model for potato leaf disease classification.



\## Scripts



\-   `train.py`: Loads the curated dataset from `data/labels.csv` and fine-tunes a `vit\_tiny\_patch16\_224` model. Outputs the best model to `mosvit\_blite.pt`.

\-   `export\_model.py`: Converts the trained PyTorch model (`.pt`) into the standard ONNX format (`.onnx`).

\-   `quantize\_model.py`: Takes the `.onnx` model and creates a smaller, faster INT8 quantized version (`\_int8.onnx`).

\-   `benchmark.py`: Measures the latency and RAM usage of the final quantized model.

\-   `leaf\_ingest.py` (in `etl/vision`): Uses the quantized model to analyze new images from the `data/images/inbox` folder and writes the results to Neo4j.

