from kafka import KafkaProducer
import pandas as pd
import json
import time
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "happiness_data")

# Cargar dataset limpio
df = pd.read_csv("data/happiness_2015to2019_cleaned.csv")

# Crear productor Kafka
producer = KafkaProducer(
    bootstrap_servers=[BOOTSTRAP],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    key_serializer=lambda k: str(k).encode("utf-8")
)

print("Enviando datos a Kafka...")
for _, row in df.iterrows():
    record = {
        "Country": row["Country"],
        "Region": row["Region"],
        "Year": int(row["Year"]),
        "GDP_per_Capita": float(row["GDP_per_Capita"]),
        "Social_Support": float(row["Social_Support"]),
        "Healthy_Life_Expectancy": float(row["Healthy_Life_Expectancy"]),
        "Freedom": float(row["Freedom"]),
        "Generosity": float(row["Generosity"]),
        "Perceptions_of_Corruption": float(row["Perceptions_of_Corruption"]),
        "Happiness_Score": float(row["Happiness_Score"])
    }
    key = f"{row['Country']}-{int(row['Year'])}"
    producer.send(TOPIC, key=key, value=record)
    time.sleep(0.2)  # simula streaming

producer.flush()
producer.close()
print("Todos los registros enviados.")
