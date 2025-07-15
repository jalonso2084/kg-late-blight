\# SmartField Blight Advisor

# SmartField Blight Advisor

**By: Jorge Luis Alonso G. | [LinkedIn](https://www.linkedin.com/in/jorgeluisalonso/)**

This project is a complete, AI-driven decision-support tool for managing potato late blight, developed as a comprehensive portfolio project. It integrates a knowledge graph, a computer vision model, a graph neural network (GNN), and a large language model (LLM) to provide actionable risk assessments.



\## Features



\* \*\*Knowledge Graph Core:\*\* Uses Neo4j to build a connected graph of fields, weather data, disease outbreaks, and sensor readings.

\* \*\*Computer Vision:\*\* A fine-tuned Vision Transformer (`vit\_tiny\_patch16\_224`) classifies potato leaf images as 'healthy' or 'lateblight'.

\* \*\*GNN Reasoning:\*\* A Temporal Graph Convolutional Network (T-GCN) analyzes 7-day snapshots of the knowledge graph to predict blight risk.

\* \*\*LLM Advisor:\*\* A RAG pipeline using GPT-4o interprets the risk prediction from the GNN and generates clear, human-readable advice for farmers.



\## Project Structure



\-   `/data/`: Holds raw data (like the outbreak CSV) and the image dataset structure (`inbox`, `train`, `val`, `test`). Ignored by Git.

\-   `/docker/`: Contains the `docker-compose.yml` to run the Neo4j database.

\-   `/etl/`: Contains all ETL (Extract, Transform, Load) scripts for populating the knowledge graph.

\-   `/gnn/`: Contains scripts for training, exporting, and benchmarking the GNN model.

\-   `/llm/`: Contains the final script for generating advice using the LLM.

\-   `/vision/`: Contains scripts for curating the image dataset and training/exporting/validating the vision model.

\-   `/utils/`: Utility helper scripts (e.g., for ID generation).



\## Setup and Installation



1\.  \*\*Environment:\*\* Set up WSL2 and Docker Desktop on Windows 11.

2\.  \*\*Database:\*\* Navigate to the `/docker` directory and run `docker compose up -d` to start the Neo4j database.

3\.  \*\*Python Environment:\*\*

&nbsp;   ```bash

&nbsp;   # Create and activate the virtual environment

&nbsp;   python -m venv .venv

&nbsp;   .\\.venv\\Scripts\\activate

&nbsp;   # Install required packages

&nbsp;   pip install -r requirements.txt

&nbsp;   ```

4\.  \*\*API Keys:\*\* Create a `.env` file in the project root and add your `OPENAI\_API\_KEY`. Use `.env.example` as a template.



\## How to Run



The project is run in sequential phases:



1\.  \*\*Load Data:\*\* Run the ETL scripts in `etl/` to populate the Neo4j database.

2\.  \*\*Train Models:\*\* Run the training scripts in `vision/` and `gnn/` to produce the `.pt` model files.

3\.  \*\*Optimize Models:\*\* Run the export and quantization scripts to create the final `.onnx` and `.ts` models.

4\.  \*\*Generate Advice:\*\* Place an image in `data/images/inbox` and run the `llm/generate\_advice.py` script.

