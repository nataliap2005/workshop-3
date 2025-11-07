# The Happy Pipeline · World Happiness (2015–2019)

Dashboard interactivo + ETL + ML + Streaming (Kafka → MySQL) para predecir el **Happiness Score** y evaluar el desempeño del modelo con datos en tiempo (casi) real.

##  Componentes

- **/data/**: CSV limpio 2015–2019 (`happiness_2015to2019_cleaned.csv`).
- **/notebooks/** o **/eda/**: EDA univariado/multivariado.
- **/model/**: entrenamiento y artefactos (`happiness_regression.pkl`).
- **/streaming/**: Producer (transforma y publica) y Consumer (predice y persiste).
- **/dashboard/**: app Dash (`the_happy_dashboard.py`).
- **/sql/**: DDL/consultas útiles (creación de tabla `predictions`).

```
repo/
├─ data/
├─ dashboard/
│  └─ the_happy_dashboard.py
├─ model/
│  ├─ train_model.py
│  └─ happiness_regression.pkl
├─ streaming/
│  ├─ producer.py
│  └─ consumer.py
├─ sql/
│  └─ create_predictions.sql
├─ .env
├─ requirements.txt
└─ README.md
```

##  Objetivo del proyecto

1) **Preparar** y entender los datos (EDA).  
2) **Entrenar** un modelo de regresión lineal múltiple.  
3) **Servir predicciones por streaming** (Kafka → Consumer → MySQL).  
4) **Evaluar** el modelo y **reportar** resultados en un **Dashboard**.

---

##  Quickstart

### 1) Requisitos

- Python 3.10+  
- MySQL 8.x (local)  
- Kafka 3.x (local; Zookeeper/Bootstrap)  
- (Opcional) WSL2 en Windows

### 2) Clonar e instalar

```bash
git clone <tu-repo>
cd <tu-repo>
python -m venv .venv
source .venv/bin/activate     # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Variables de entorno

Crea **.env** en la raíz:

```env
# Datos / Modelo
DATA_CSV=./data/happiness_2015to2019_cleaned.csv
MODEL_PKL=./model/happiness_regression.pkl

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=tu_password
MYSQL_DB=happiness

# Kafka
KAFKA_BOOTSTRAP=localhost:9092
KAFKA_TOPIC=world_happiness
```

### 4) Base de datos

Crea la base y la tabla:

```sql
-- sql/create_predictions.sql
CREATE DATABASE IF NOT EXISTS happiness;
USE happiness;

CREATE TABLE IF NOT EXISTS predictions (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  Country VARCHAR(100),
  Region  VARCHAR(100),
  Year INT,

  GDP_per_Capita DOUBLE,
  Social_Support DOUBLE,
  Healthy_Life_Expectancy DOUBLE,
  Freedom DOUBLE,
  Generosity DOUBLE,
  Perceptions_of_Corruption DOUBLE,

  Predicted_Score DOUBLE,
  Actual_Score DOUBLE,
  Data_Set ENUM('train','test') NULL,
  ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 5) Entrenar el modelo

Si no tienes el `.pkl`, ejecuta:

```bash
python model/train_model.py
# guarda ./model/happiness_regression.pkl
```

> El script divide 70/30 (train/test), entrena **Regresión Lineal**, calcula R²/MAE/RMSE y guarda el modelo.

### 6) Iniciar Kafka y topics

```bash
# Ejemplos (ajusta a tu instalación)
bin/zookeeper-server-start.sh config/zookeeper.properties
bin/kafka-server-start.sh     config/server.properties
bin/kafka-topics.sh --bootstrap-server localhost:9092 \
  --create --topic world_happiness --partitions 1 --replication-factor 1
```

### 7) Streaming

- **Producer**: lee fuente (o batches del CSV), aplica **transformaciones** (tu bloque de limpieza) y publica al topic.
- **Consumer**: recibe, **predice** con el `.pkl` y **escribe** en MySQL (`predictions`), incluyendo `Data_Set` (train/test).

```bash
# En terminal A
python streaming/producer.py

# En terminal B
python streaming/consumer.py
```

### 8) Dashboard

```bash
python dashboard/the_happy_dashboard.py
# http://localhost:8050
```

En la pestaña **Overview**:
- Selector **Train/Test/All** (dropdown) — filtra por la columna `Data_Set`.
- KPIs globales: **R², MAE, RMSE, N**.
- Gráficas de desempeño y diagnóstico (ver “Visualizaciones”).

---

## Visualizaciones y qué significan

- **Predicted vs Actual**: puntos por país-año; línea punteada = ideal (y=x). Cuanto más cerca a la línea, mejor.  
- **RMSE por Año**: barras del error medio cuadrático por año del subset seleccionado.  
- **Residuos vs Predicción**: nube alrededor de y=0; patrón aleatorio sugiere buena especificación.  
- **Histograma de errores**: distribución de `Actual − Pred`; centrada en 0 y simétrica ≈ sin sesgo.  
- **Calibración por deciles**: promedios reales vs promedios predichos por bins; cerca de la diagonal = buena calibración.  
- **Top-15 errores absolutos**: países donde el modelo falla más (para análisis).  
- **RMSE Región×Año (heatmap)**: comparación espacial/temporal del error.  
- **Q–Q plot de residuos**: cercanía a la 45° indica normalidad aproximada de errores.

---

##  Métricas

- **MAE** = mean(|y − ŷ|)  
- **RMSE** = sqrt(mean((y − ŷ)²))  
- **R²** = 1 − SS_res / SS_tot  

> En el Dashboard se calculan **global** y por **Año** sobre el conjunto seleccionado (Train/Test/All).

---

## Esquemas y llaves

**Features usadas (orden):**

1. `GDP_per_Capita`  
2. `Social_Support`  
3. `Healthy_Life_Expectancy`  
4. `Freedom`  
5. `Generosity`  
6. `Perceptions_of_Corruption`

**Tabla `predictions` (streaming → evaluación):**

| Campo | Tipo | Descripción |
|---|---|---|
| Country, Region | texto | Identificación |
| Year | int | 2015–2019 |
| 6 features | double | predictoras |
| Predicted_Score | double | ŷ |
| Actual_Score | double | y (si se conoce) |
| Data_Set | enum('train','test') | origen del registro |
| ts | timestamp | momento de inserción |

---

## Scripts típicos

```bash
# contar registros por subset
mysql -u root -p -e "SELECT Data_Set, COUNT(*) AS n FROM happiness.predictions GROUP BY Data_Set;"

# solo test
mysql -u root -p -e "SELECT COUNT(*) FROM happiness.predictions WHERE Data_Set='test';"

# limpiar tabla (con cuidado)
mysql -u root -p -e "TRUNCATE TABLE happiness.predictions;"
```

---

## `the_happy_dashboard.py` (resumen)

- Lee `.env` para rutas de CSV/MODEL y credenciales MySQL.
- `load_predictions(dataset)` filtra por `Data_Set ∈ {train, test, all}` (con fallback si la columna no existe).
- Dropdown **dataset_dd** controla todo el Overview + tabla **Streaming**.
- Pestañas: **Overview**, **EDA**, **Model**, **Streaming**.

---

## Troubleshooting

- **No aparecen datos en el Dashboard**  
  - Verifica que el **consumer** esté insertando en MySQL sin errores.  
  - Revisa conexión/credenciales de MySQL en `.env`.  
  - Asegúrate de que la tabla `predictions` existe y tiene columnas correctas.

- **`ModuleNotFoundError`**  
  - Ejecuta `pip install -r requirements.txt`.  
  - Para VIF, instala `statsmodels`.

- **Kafka 404/puertos**  
  - Revisa `KAFKA_BOOTSTRAP` y que el topic exista.  
  - En Windows+WSL, expón puertos correctamente.

- **`MODEL_PKL` no encontrado**  
  - Corre `python model/train_model.py` o ajusta la ruta en `.env`.

---

## Criterios de evaluación (checklist)

- [x] **Compare predicted vs actual** en **TEST** (Dashboard → Overview).  
- [x] **Compute metrics** MAE/RMSE/R² (global y por año).  
- [x] **Summarize EDA + Model + Streaming** (gráficas y KPIs integradas).  
- [x] **Filtro Train/Test/All** para análisis diferenciado.

---

## requirements.txt (sugerido)

```
python-dotenv
pandas
numpy
scikit-learn
joblib
plotly
dash
dash-bootstrap-components
mysql-connector-python
statsmodels
```

---




