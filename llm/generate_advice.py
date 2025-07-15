import os
from openai import OpenAI
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "test1234"
# This now safely loads the key from your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_FIELD_ID = "FIELD_42"


def get_latest_risk_from_kg(tx, field_id):
    """Query the knowledge graph for the latest risk prediction."""
    query = """
    MATCH (f:Field {id: $field_id})-[:HAS_RISK_PREDICTION]->(r:RiskPrediction)
    RETURN r.risk AS risk, r.probability AS probability, r.date AS date
    ORDER BY r.date DESC
    LIMIT 1
    """
    result = tx.run(query, field_id=field_id)
    return result.single()

def generate_advice(risk_data):
    """Uses an LLM to generate advice based on the risk data."""
    if not risk_data:
        return "No risk data found for the specified field."

    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""
    You are an expert agronomist providing clear advice to a potato farmer.
    Based on the following data, write a one-paragraph summary.

    Data:
    - Field ID: {TARGET_FIELD_ID}
    - Predicted Risk of Late Blight: {risk_data['risk']}
    - Confidence Score: {risk_data['probability']:.2f}
    - Prediction Date: {risk_data['date']}

    Explain what the risk level means and suggest a course of action.
    """

    print("\nSending prompt to LLM...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert agronomist."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found. Please add it to your .env file.")
    else:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session(database="neo4j") as session:
            latest_risk = session.execute_read(get_latest_risk_from_kg, TARGET_FIELD_ID)

        advice = generate_advice(latest_risk)

        print("\n--- SmartField Blight Advisor ---")
        print(advice)
        print("---------------------------------")

        driver.close()