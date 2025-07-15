import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import torch
from neo4j import GraphDatabase
from torch_geometric.data import Data
from datetime import datetime, timedelta, timezone
import collections # <-- THIS IS THE MISSING LINE

# --- CONFIGURATION ---
CONFIG = {
    "torchscript_model_path": "gnn/gsage_v1.ts",
    "num_node_features": 2,
    "class_map": {0: 'Low', 1: 'High'}
}

class RiskPredictor:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"Loading TorchScript model from {CONFIG['torchscript_model_path']}...")
        self.model = torch.jit.load(CONFIG['torchscript_model_path'])
        self.model.eval()
        print("Model loaded successfully.")

    def close(self):
        self.driver.close()

    def get_all_field_ids(self):
        with self.driver.session(database="neo4j") as session:
            result = session.run("MATCH (f:Field) RETURN f.id AS fieldId")
            return [record["fieldId"] for record in result]

    def get_latest_snapshot_for_field(self, session, field_id):
        end_date = datetime.strptime("2024-05-30", "%Y-%m-%d")
        end_date_str = end_date.strftime('%Y-%m-%d')

        query = """
        MATCH (f:Field {id: $field_id})-[:OBSERVED_AT]->(w:WeatherState)
        WHERE date($end_date) - duration({days: 7}) < date(datetime({epochMillis: w.datetime})) <= date($end_date)
        RETURN w ORDER BY w.datetime
        """
        result = session.run(query, field_id=field_id, end_date=end_date_str)

        nodes_by_day = collections.defaultdict(list)
        for record in result:
            node = record["w"]
            dt_object = datetime.fromtimestamp(node["datetime"] / 1000, tz=timezone.utc)
            day_str = dt_object.strftime("%Y-%m-%d")
            nodes_by_day[day_str].append(node)

        if len(nodes_by_day) < 7:
            return None

        snapshot_sequence = []
        for day_str in sorted(nodes_by_day.keys()):
            daily_nodes = nodes_by_day[day_str]
            node_features = [[node['T'], node['RH']] for node in daily_nodes]
            x = torch.tensor(node_features, dtype=torch.float)

            edge_list = [[i, i + 1] for i in range(len(daily_nodes) - 1)]
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

            snapshot_sequence.append(Data(x=x, edge_index=edge_index))

        return snapshot_sequence

    def predict_and_write_risk(self, field_id):
        with self.driver.session(database="neo4j") as session:
            print(f"  - Fetching latest data for {field_id}...")
            snapshot_sequence = self.get_latest_snapshot_for_field(session, field_id)

            if snapshot_sequence is None or len(snapshot_sequence) != 7:
                print(f"  - Not enough recent data to make a prediction for {field_id}.")
                return

            x_list = [s.x for s in snapshot_sequence]
            edge_index_list = [s.edge_index for s in snapshot_sequence]
            batch_list = [torch.zeros(s.x.shape[0], dtype=torch.long) for s in snapshot_sequence]

            with torch.no_grad():
                logits = self.model(x_list, edge_index_list, batch_list)

            probabilities = torch.softmax(logits, dim=1)
            prediction_idx = logits.argmax(dim=1).item()
            confidence = probabilities[0][prediction_idx].item()
            risk_label = CONFIG["class_map"][prediction_idx]

            print(f"    -> Predicted Risk: '{risk_label}' with {confidence:.2f} confidence.")

            write_query = """
            MATCH (f:Field {id: $field_id})
            MERGE (f)-[:HAS_RISK_PREDICTION]->(r:RiskPrediction {date: date() - duration({days:1})})
            ON CREATE SET
                r.risk = $risk,
                r.probability = $probability
            """
            session.run(write_query, {
                "field_id": field_id,
                "risk": risk_label,
                "probability": confidence
            })
            print(f"    -> Wrote prediction to knowledge graph.")

def run_prediction_job():
    print("Starting daily risk prediction job...")
    predictor = RiskPredictor("bolt://localhost:7687", "neo4j", "test1234")

    field_ids = predictor.get_all_field_ids()
    for field_id in field_ids:
        print(f"Processing field: {field_id}")
        predictor.predict_and_write_risk(field_id)

    predictor.close()
    print("\nPrediction job complete.")

if __name__ == "__main__":
    run_prediction_job()