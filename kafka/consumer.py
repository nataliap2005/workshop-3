from kafka import KafkaConsumer
import json
import joblib
import pandas as pd
import os
import mysql.connector
from mysql.connector import errorcode
from dotenv import load_dotenv

# ====== Config ======
FEATURE_ORDER = [
    "GDP_per_Capita", "Social_Support", "Healthy_Life_Expectancy",
    "Freedom", "Generosity", "Perceptions_of_Corruption"
]

# Cargar variables de entorno
load_dotenv()
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "happiness_data")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "happiness")

# ====== Modelo ======
model = joblib.load("model/happiness_regression.pkl")

# ====== Conexión y preparación de BD MySQL ======
def ensure_mysql():
    # Conexión al servidor (sin DB) para crear la BD si no existe
    conn_server = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD
    )
    cur = conn_server.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` DEFAULT CHARACTER SET utf8mb4;")
    cur.close()
    conn_server.close()

    # Conexión ya a la BD
    conn = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DB
    )
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        Country VARCHAR(100),
        Region VARCHAR(100),
        Year INT,
        GDP_per_Capita DOUBLE,
        Social_Support DOUBLE,
        Healthy_Life_Expectancy DOUBLE,
        Freedom DOUBLE,
        Generosity DOUBLE,
        Perceptions_of_Corruption DOUBLE,
        Predicted_Score DOUBLE,
        Actual_Score DOUBLE,
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    return conn

conn = ensure_mysql()
cursor = conn.cursor()

# ====== Kafka Consumer ======
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=[BOOTSTRAP],
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    group_id="happiness-consumer",
    auto_offset_reset="earliest",  # para leer desde el inicio en pruebas
    enable_auto_commit=True
)

print(f"Esperando datos en '{TOPIC}'... (MySQL DB: {MYSQL_DB})")

for msg in consumer:
    record = msg.value

    # DataFrame con features en el orden esperado por el modelo
    features = pd.DataFrame([[record[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)

    # Predicción
    pred = float(model.predict(features)[0])

    # Insertar en MySQL: features + real + pred
    cursor.execute("""
        INSERT INTO predictions (
            Country, Region, Year,
            GDP_per_Capita, Social_Support, Healthy_Life_Expectancy,
            Freedom, Generosity, Perceptions_of_Corruption,
            Predicted_Score, Actual_Score
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        record["Country"],
        record["Region"],
        int(record["Year"]),
        float(record["GDP_per_Capita"]),
        float(record["Social_Support"]),
        float(record["Healthy_Life_Expectancy"]),
        float(record["Freedom"]),
        float(record["Generosity"]),
        float(record["Perceptions_of_Corruption"]),
        pred,
        float(record["Happiness_Score"])
    ))
    conn.commit()

    print(f"{record['Country']} ({record['Year']}): Pred {pred:.3f} | Real {record['Happiness_Score']:.3f}")
