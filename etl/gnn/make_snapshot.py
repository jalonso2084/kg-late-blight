import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from neo4j import GraphDatabase
from torch_geometric.data import Data
import torch
from datetime import datetime, timedelta

# --- CONFIGURATION ---
CONFIG = {
    "output_dir": "data/snapshots",
    "days_in_snapshot": 7
}

class Neo4jExporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_all_field_ids(self):
        with self.driver.session(database="neo4j") as session:
            result = session.run("MATCH (f:Field) RETURN f.id AS fieldId")
            return [record["fieldId"] for record in result]

    def get_snapshot_dates(self, field_id):
        """
        NEW: This function now queries the database to find all possible
        end-dates for which a 7-day snapshot can be created.
        """
        query = """
        MATCH (f:Field {id: $field_id})-[:OBSERVED_AT]->(w:WeatherState)
        WITH min(date(datetime({epochMillis: w.datetime}))) AS minDate, 
             max(date(datetime({epochMillis: w.datetime}))) AS maxDate
        // We need at least 8 days of data to create one 7-day snapshot and have a label for the next day
        WHERE duration.between(minDate, maxDate).days >= 8
        RETURN minDate, maxDate
        """
        with self.driver.session(database="neo4j") as session:
            result = session.run(query, field_id=field_id).single()

        if not result:
            return []

        # Generate all possible end-dates for snapshots
        dates = []
        current_date = result["maxDate"].to_native()
        min_date = result["minDate"].to_native() + timedelta(days=CONFIG["days_in_snapshot"])

        while current_date >= min_date:
            dates.append(current_date)
            current_date -= timedelta(days=1)

        return dates

    def export_snapshot(self, field_id, end_date):
        query = """
        MATCH (f:Field {id: $field_id})
        CALL {
            WITH f
            MATCH (f)-[:OBSERVED_AT]->(w:WeatherState)
            WHERE date($end_date) - duration({days: 7}) < date(datetime({epochMillis: w.datetime})) <= date($end_date)
            RETURN collect(w) as weather_nodes
        }
        CALL {
            WITH f
            MATCH (f)<-[:OCCURRED_IN]-(o:Outbreak)
            WHERE date($end_date) - duration({days: 7}) < o.date <= date($end_date)
            RETURN collect(o) as outbreak_nodes
        }
        CALL {
            MATCH (f)<-[:OCCURRED_IN]-(label_outbreak:Outbreak)
            WHERE label_outbreak.date = date($end_date) + duration({days: 1})
            RETURN count(label_outbreak) > 0 AS label
        }
        RETURN weather_nodes, outbreak_nodes, label
        """
        end_date_str = end_date.strftime('%Y-%m-%d')

        with self.driver.session(database="neo4j") as session:
            result = session.run(query, {"field_id": field_id, "end_date": end_date_str})
            data = result.single()

        if not data or not data["weather_nodes"] or len(data["weather_nodes"]) < (24 * CONFIG["days_in_snapshot"]):
            print(f"  - Incomplete data for snapshot ending {end_date_str}. Skipping.")
            return

        nodes = data["weather_nodes"]
        nodes.sort(key=lambda n: n['datetime']) # Sort by time
        node_features = [ [node['T'], node['RH']] for node in nodes ]
        x = torch.tensor(node_features, dtype=torch.float)

        edge_list = []
        for i in range(len(nodes) - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        y = torch.tensor([1 if data["label"] else 0], dtype=torch.long)
        pyg_data = Data(x=x, edge_index=edge_index, y=y)

        output_filename = f"{field_id}_{end_date_str}.pt"
        output_path = os.path.join(CONFIG["output_dir"], output_filename)
        torch.save(pyg_data, output_path)
        print(f"  - Saved snapshot to {output_path}")

def run_exporter():
    print("Starting snapshot exporter...")
    if not os.path.exists(CONFIG["output_dir"]):
        os.makedirs(CONFIG["output_dir"])

    exporter = Neo4jExporter("bolt://localhost:7687", "neo4j", "test1234")

    field_ids = exporter.get_all_field_ids()
    if not field_ids:
        print("No fields found in the database.")
        return

    for field_id in field_ids:
        print(f"Processing field: {field_id}")
        snapshot_dates = exporter.get_snapshot_dates(field_id)
        if not snapshot_dates:
            print(f"  - Not enough consecutive data found for field {field_id} to create snapshots.")
            continue
        for date in snapshot_dates:
            exporter.export_snapshot(field_id, date)

    exporter.close()
    print("Snapshot export complete.")


if __name__ == "__main__":
    run_exporter()