from kafka import KafkaConsumer
import json
import joblib
import pandas as pd
import os
import sys
import mysql.connector
from dotenv import load_dotenv

# ========= Config =========
FEATURE_ORDER = [
    "GDP_per_Capita", "Social_Support", "Healthy_Life_Expectancy",
    "Freedom", "Generosity", "Perceptions_of_Corruption"
]

# Variables de entorno
load_dotenv()
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "happiness_data")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "happiness")

MODEL_PATH = os.getenv("MODEL_PATH", "../model/happiness_regression.pkl")
SPLIT_MAP_PATH = os.getenv("SPLIT_MAP_PATH", "../model/train_test_split.csv")

GROUP_ID = os.getenv("KAFKA_GROUP_ID", "happiness-consumer")
AUTO_OFFSET_RESET = os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")  # earliest|latest
ENABLE_AUTO_COMMIT = os.getenv("KAFKA_ENABLE_AUTO_COMMIT", "true").lower() == "true"
BATCH_COMMIT = int(os.getenv("BATCH_COMMIT", "1"))  # commit cada N inserciones

# ========= Cargar modelo =========
try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Modelo cargado: {MODEL_PATH}")
except Exception as e:
    print(f"[FATAL] No se pudo cargar el modelo en {MODEL_PATH}: {e}", file=sys.stderr)
    sys.exit(1)

# ========= Cargar mapa train/test =========
if os.path.exists(SPLIT_MAP_PATH):
    split_df = pd.read_csv(SPLIT_MAP_PATH)
    split_map = dict(zip(split_df["Key"], split_df["Data_Set"]))
    print(f"[OK] Split map cargado: {SPLIT_MAP_PATH} (registros: {len(split_map)})")
else:
    split_map = {}
    print(f"[WARN] No existe {SPLIT_MAP_PATH}; Data_Set se marcará 'unknown'.")

def get_data_set(country: str, year: int) -> str:
    return split_map.get(f"{country}|{year}", "unknown")

# ========= MySQL =========
def ensure_mysql():
    # Crear DB si no existe
    conn_server = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD
    )
    cur = conn_server.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}` DEFAULT CHARACTER SET utf8mb4;")
    cur.close()
    conn_server.close()

    # Conectar a la DB concreta
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
        Data_Set VARCHAR(10),
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_country_year (Country, Year)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    conn.commit()
    cur.close()
    return conn

conn = ensure_mysql()
cursor = conn.cursor()  # usamos placeholders %s

# Query con UPSERT e inclusión de Data_Set
insert_sql = """
INSERT INTO predictions (
    Country, Region, Year,
    GDP_per_Capita, Social_Support, Healthy_Life_Expectancy,
    Freedom, Generosity, Perceptions_of_Corruption,
    Predicted_Score, Actual_Score, Data_Set
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    GDP_per_Capita = VALUES(GDP_per_Capita),
    Social_Support = VALUES(Social_Support),
    Healthy_Life_Expectancy = VALUES(Healthy_Life_Expectancy),
    Freedom = VALUES(Freedom),
    Generosity = VALUES(Generosity),
    Perceptions_of_Corruption = VALUES(Perceptions_of_Corruption),
    Predicted_Score = VALUES(Predicted_Score),
    Actual_Score = VALUES(Actual_Score),
    Data_Set = VALUES(Data_Set),
    ts = CURRENT_TIMESTAMP
"""

# ========= Kafka Consumer =========
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=[BOOTSTRAP],
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    group_id=GROUP_ID,
    auto_offset_reset=AUTO_OFFSET_RESET,
    enable_auto_commit=ENABLE_AUTO_COMMIT
)

print(f"Escuchando '{TOPIC}' @ {BOOTSTRAP} | DB: {MYSQL_DB}")

processed = 0
for msg in consumer:
    try:
        record = msg.value

        # Validación mínima
        required = set(FEATURE_ORDER + ["Country", "Region", "Year", "Happiness_Score"])
        if not required.issubset(record):
            missing = list(required - set(record))
            print(f"[WARN] Faltan campos {missing} (offset {msg.offset}). Omitido.")
            continue

        # Vector de features
        features = pd.DataFrame([[record[k] for k in FEATURE_ORDER]], columns=FEATURE_ORDER)

        # Predicción
        pred = float(model.predict(features)[0])

        # Determinar train/test
        data_set = get_data_set(str(record["Country"]), int(record["Year"]))

        params = (
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
            float(record["Happiness_Score"]),
            data_set
        )
        cursor.execute(insert_sql, params)
        processed += 1

        if processed % BATCH_COMMIT == 0:
            conn.commit()

        print(f"{record['Country']} ({record['Year']}): "
              f"Set={data_set} | Pred {pred:.3f} | Real {record['Happiness_Score']:.3f}")

    except mysql.connector.Error as db_err:
        print(f"[ERROR][MySQL] {db_err}", file=sys.stderr)
        conn.rollback()
    except Exception as e:
        print(f"[ERROR] Offset={msg.offset}: {e}", file=sys.stderr)

# Commit final por si queda algo pendiente (si el loop se interrumpe)
try:
    conn.commit()
except:
    pass
