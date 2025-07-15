\# GNN Evaluation Report - Phase 3



\## 1. Goal

The goal of this phase was to deliver an edge-ready GraphSAGE/T-GCN model to predict late blight risk based on 7-day graph snapshots from the knowledge graph.



\## 2. Model

\-   \*\*Architecture:\*\* T-GCN (Temporal Graph Convolutional Network)

\-   \*\*Details:\*\* 2 GCN layers, 2 LSTM layers, 64 hidden channels.

\-   \*\*Training Data:\*\* 63 graph snapshots generated from 1 month of weather data across 3 fields.



\## 3. Performance

The final model from Sprint 3-3 achieved the following results on the validation set.



\-   \*\*Best Validation F1 Score:\*\* 1.0000 (Target was >= 0.80)



\## 4. Edge-Readiness Benchmark

The exported TorchScript model (`gsage\_v1.ts`) was benchmarked on the CPU.



\-   \*\*File Size:\*\* 0.14 MB (Target was < 8 MB)

\-   \*\*Average Latency:\*\* 12.18 ms / sequence (Target was <= 80 ms)

\-   \*\*Model RAM Usage:\*\* 3.77 MB (Target was < 1 GB)



\## 5. Conclusion

All success criteria for Phase 3 have been met. The model is highly performant and efficient, ready for integration into a nightly prediction job.

