import os, requests, datetime as dt, pandas as pd
from neo4j import GraphDatabase
# Use the correct function name from your helper script
from utils.id_helpers import generate_weather_state_id 

POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# --- NEW: Define all fields in one place ---
FIELDS = [
    {"id": "FIELD_42", "lat": -12.08, "lon": -75.21},
    {"id": "FIELD_7",  "lat": -12.15, "lon": -75.30},
    {"id": "FIELD_88", "lat": -12.00, "lon": -75.15}
]

# Fetch a full month of data
START      = "20240501"
END        = "20240531"

def fetch_power(lat, lon, start, end):
    params = {
        "parameters": "T2M,RH2M",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON"
    }
    print(f"Fetching data for {lat},{lon} from {start} to {end}")
    r = requests.get(POWER_URL, params=params, timeout=30)
    r.raise_for_status() 
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame.from_dict(data['T2M'], orient='index', columns=['T'])
    df['RH'] = data['RH2M']
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H')
    df = df[df['T'] != -999.0]
    return df.reset_index(names="datetime")

def push(driver, df, field_id):
    with driver.session(database="neo4j") as sess:
        query = """
        MERGE (f:Field {id:$field})
        WITH f
        UNWIND $rows AS row
          MERGE (w:WeatherState {id:row.id})
            ON CREATE SET
              w.datetime = apoc.date.fromISO8601(row.datetime),
              w.T = row.T,
              w.RH = row.RH
          MERGE (f)-[:OBSERVED_AT]->(w);
        """
        df_records = df.to_dict("records")
        for record in df_records:
            record['datetime'] = record['datetime'].isoformat() + "Z"

        print(f"Pushing {len(df_records)} records to Neo4j for {field_id}...")
        sess.run(query, field=field_id, rows=df_records)
        print("Push complete.")

if __name__ == "__main__":
    db_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j","test1234"))

    # --- NEW: Loop through each field and process it ---
    for field in FIELDS:
        print(f"\n--- Processing Field: {field['id']} ---")
        df = fetch_power(field["lat"], field["lon"], START, END)

        if not df.empty:
            df["id"] = df.apply(
                lambda r: generate_weather_state_id(field["id"], r["datetime"].isoformat() + "Z"), axis=1
            )
            push(db_driver, df, field["id"])
        else:
            print(f"DataFrame is empty for {field['id']}. No data will be pushed.")

    db_driver.close()
    print("\nAll fields processed.")