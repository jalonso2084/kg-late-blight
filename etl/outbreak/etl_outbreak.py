# --- ADD THESE 4 LINES AT THE VERY TOP ---
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
# -----------------------------------------

import csv
import hashlib
from neo4j import GraphDatabase

CSV_FILE_PATH = "data/outbreaks.csv"

def generate_outbreak_id(field_id: str, outbreak_date: str) -> str:
    """Generates a deterministic SHA1 hash for an Outbreak node."""
    unique_string = f"{field_id}{outbreak_date}"
    return hashlib.sha1(unique_string.encode('utf-8')).hexdigest()

def run_etl():
    # --- This function is now updated to process one record at a time ---
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","test1234"))
    with driver.session(database="neo4j") as sess:
        # This simpler query processes one outbreak at a time
        query = """
        MERGE (f:Field {id: $field_id})
        MERGE (o:Outbreak {id: $outbreak_id})
          ON CREATE SET
            o.date = date($outbreak_date),
            o.severity = toFloat($severity)
        MERGE (f)<-[:OCCURRED_IN]-(o)
        """
        try:
            with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                print(f"Processing records from {CSV_FILE_PATH}...")
                for row in reader:
                    # For each row in the CSV, run the query
                    outbreak_id = generate_outbreak_id(row["fieldId"], row["date"])
                    sess.run(query, {
                        "field_id": row["fieldId"],
                        "outbreak_id": outbreak_id,
                        "outbreak_date": row["date"],
                        "severity": row["severity"]
                    })
                print("Push complete.")
        except FileNotFoundError:
            print(f"ERROR: Cannot find the data file at {CSV_FILE_PATH}")
    driver.close()

if __name__ == "__main__":
    run_etl()