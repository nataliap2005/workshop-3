from kafka import KafkaProducer
import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
import re
import pandas as pd
import country_converter as coco

# Cargar variables de entorno
load_dotenv()
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "happiness_data")

y_2015 = pd.read_csv('../data/2015.csv')
y_2016 = pd.read_csv('../data/2016.csv')
y_2017 = pd.read_csv('../data/2017.csv')
y_2018 = pd.read_csv('../data/2018.csv')
y_2019 = pd.read_csv('../data/2019.csv')

# --- 2015 ---
rename_2015 = {
    "Happiness Score": "Happiness_Score",
    "Happiness Rank": "Happiness_Rank",
    "Economy (GDP per Capita)": "GDP_per_Capita",
    "Health (Life Expectancy)": "Healthy_Life_Expectancy",
    "Trust (Government Corruption)": "Perceptions_of_Corruption",
    "Family": "Social_Support",
}
y_2015 = y_2015.rename(columns=rename_2015)
y_2015["Year"] = 2015

# --- 2016 ---
rename_2016 = {
    "Happiness Score": "Happiness_Score",
    "Happiness Rank": "Happiness_Rank",
    "Economy (GDP per Capita)": "GDP_per_Capita",
    "Health (Life Expectancy)": "Healthy_Life_Expectancy",
    "Trust (Government Corruption)": "Perceptions_of_Corruption",
    "Family": "Social_Support",
}
y_2016 = y_2016.rename(columns=rename_2016)
y_2016["Year"] = 2016

# --- 2017 ---
rename_2017 = {
    "Happiness.Rank": "Happiness_Rank",
    "Happiness.Score": "Happiness_Score",
    "Economy..GDP.per.Capita.": "GDP_per_Capita",
    "Health..Life.Expectancy.": "Healthy_Life_Expectancy",
    "Trust..Government.Corruption.": "Perceptions_of_Corruption",
    "Family": "Social_Support",
    "Whisker.high": "Upper_Confidence_Interval",
    "Whisker.low": "Lower_Confidence_Interval",
    "Dystopia.Residual": "Dystopia_Residual",
}
y_2017 = y_2017.rename(columns=rename_2017)
y_2017["Year"] = 2017
y_2017["Region"] = pd.NA  # no existe en 2017

# --- 2018 ---
rename_2018 = {
    "Overall rank": "Happiness_Rank",
    "Country or region": "Country",
    "Score": "Happiness_Score",
    "GDP per capita": "GDP_per_Capita",
    "Social support": "Social_Support",
    "Healthy life expectancy": "Healthy_Life_Expectancy",
    "Freedom to make life choices": "Freedom",
    "Generosity": "Generosity",
    "Perceptions of corruption": "Perceptions_of_Corruption",
}
y_2018 = y_2018.rename(columns=rename_2018)
y_2018["Year"] = 2018
y_2018["Region"] = pd.NA  # no existe en 2018

# --- 2019 ---
rename_2019 = {
    "Overall rank": "Happiness_Rank",
    "Country or region": "Country",
    "Score": "Happiness_Score",
    "GDP per capita": "GDP_per_Capita",
    "Social support": "Social_Support",
    "Healthy life expectancy": "Healthy_Life_Expectancy",
    "Freedom to make life choices": "Freedom",
    "Generosity": "Generosity",
    "Perceptions of corruption": "Perceptions_of_Corruption",
}
y_2019 = y_2019.rename(columns=rename_2019)
y_2019["Year"] = 2019
y_2019["Region"] = pd.NA  # no existe en 2019

cc = coco.CountryConverter()

# --- Normalizaciones de nombres para que coco los reconozca ---
FIX_NAMES = {
    "Hong Kong S.A.R., China": "Hong Kong",
    "Taiwan Province of China": "Taiwan",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "Congo (Brazzaville)": "Republic of the Congo",
    "Congo (Kinshasa)": "Democratic Republic of the Congo",
    "Ivory Coast": "Côte d'Ivoire",
    "Macedonia": "North Macedonia",
    "Swaziland": "Eswatini",
    "North Cyprus": "Cyprus",
    "Northern Cyprus": "Cyprus",
    "Palestinian Territories": "State of Palestine",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Laos": "Lao People's Democratic Republic",
    "Russia": "Russian Federation",
    "South Korea": "Korea, Republic of",
}

def add_region(df, country_col="Country", region_col="Region", to="UNregion"):
    """
    Asigna la región ONU a cada país del df[country_col] y crea df[region_col].
    'to' puede ser UNregion' (continente).
    """
    # 1) limpiar strings
    df[country_col] = df[country_col].astype(str).str.strip()

    # 2) normalizar nombres difíciles
    df[country_col] = df[country_col].replace(FIX_NAMES)

    # 3) convertir con coco
    conv = cc.convert(names=df[country_col].tolist(), to=to, not_found=None)
    df[region_col] = pd.Series(conv, index=df.index)

    return df

# === Aplica a tus dataframes de cada año ===
y_2017 = add_region(y_2017, country_col="Country", region_col="Region", to="UNregion")
y_2018 = add_region(y_2018, country_col="Country", region_col="Region", to="UNregion")
y_2019 = add_region(y_2019, country_col="Country", region_col="Region", to="UNregion")

def check_missing(df, year_label):
    miss = df[df["Region"].isna()]["Country"].unique().tolist()
    print(f"{year_label} - países sin región asignada:", miss if miss else "Ninguno")

check_missing(y_2017, "2017")
check_missing(y_2018, "2018")
check_missing(y_2019, "2019")

COUNTRY_MAPPING = {
    'Korea, Republic of': 'South Korea',
    'Republic of Korea': 'South Korea',
    'Korea, Rep.': 'South Korea',
    'Russian Federation': 'Russia',
    'Kyrgyz Republic': 'Kyrgyzstan',
    "Lao People's Democratic Republic": 'Laos',
    'Lao PDR': 'Laos',
    'State of Palestine': 'Palestinian Territories',
    'Palestinian Territory': 'Palestinian Territories',
    'Palestinian Territories': 'Palestinian Territories',
    'Democratic Republic of the Congo': 'Congo (Kinshasa)',
    'Republic of the Congo': 'Congo (Brazzaville)',
    "Côte d'Ivoire": 'Ivory Coast',
    'Cote dIvoire': 'Ivory Coast',
    'Trinidad & Tobago': 'Trinidad and Tobago',
    'Trinidad and Tobago': 'Trinidad and Tobago',
    'Swaziland': 'Eswatini',
    'Eswatini': 'Eswatini',
    'Macedonia': 'North Macedonia',
    'North Macedonia': 'North Macedonia',
    'Hong Kong S.A.R., China': 'Hong Kong',
    'Taiwan Province of China': 'Taiwan',
    'United States of America': 'United States',
    'USA': 'United States',
    'U.S.A.': 'United States',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'UK': 'United Kingdom',
    'Somaliland Region': 'Somalia',
    'Somaliland region': 'Somalia',
}

REGION_MAPPING = {
    'Northern America': 'North America',
    'North America': 'North America',
    'South America': 'Latin America and Caribbean',
    'Central America': 'Latin America and Caribbean',
    'Caribbean': 'Latin America and Caribbean',
    'South-eastern Asia': 'Southeastern Asia',
    'Southeast Asia': 'Southeastern Asia',
    'South Eastern Asia': 'Southeastern Asia',
    'Western Asia': 'Middle East and Northern Africa',
    'Northern Africa': 'Middle East and Northern Africa',
    'Middle East and Northern Africa': 'Middle East and Northern Africa',
    'Eastern Europe': 'Central and Eastern Europe',
    'Central Asia': 'Central and Eastern Europe',
    'Southern Europe': 'Western Europe',
    'Northern Europe': 'Western Europe',
    'Western Europe': 'Western Europe',
    'Eastern Africa': 'Sub-Saharan Africa',
    'Western Africa': 'Sub-Saharan Africa',
    'Southern Africa': 'Sub-Saharan Africa',
    'Middle Africa': 'Sub-Saharan Africa',
    'Australia and New Zealand': 'Australia and New Zealand',
    'Eastern Asia': 'Eastern Asia',
    'Southern Asia': 'Southern Asia',
}

def normalize_data(df):
    df_norm = df.copy()
    country_col = None
    region_col = None
    possible_country_names = ['country', 'country name', 'country or region', 'nation']
    possible_region_names = ['region', 'area', 'continent']
    for col in df_norm.columns:
        col_lower = col.lower()
        if any(name in col_lower for name in possible_country_names):
            country_col = col
        if any(name in col_lower for name in possible_region_names):
            region_col = col
    if country_col:
        try:
            if df_norm[country_col].dtype != 'object':
                df_norm[country_col] = df_norm[country_col].astype(str)
            mask = df_norm[country_col].notna()
            df_norm.loc[mask, country_col] = (
                df_norm.loc[mask, country_col]
                .str.strip()
                .str.replace(r'\s+&\s+', ' and ', regex=True)
                .str.replace(r'\s{2,}', ' ', regex=True)
            )
            df_norm[country_col] = df_norm[country_col].replace(COUNTRY_MAPPING)
            if country_col != 'Country':
                df_norm = df_norm.rename(columns={country_col: 'Country'})
        except Exception as e:
            print(f"Advertencia: Error normalizando columna de país '{country_col}': {e}")
            if country_col != 'Country':
                df_norm = df_norm.rename(columns={country_col: 'Country'})
    if region_col:
        try:
            if df_norm[region_col].dtype != 'object':
                df_norm[region_col] = df_norm[region_col].astype(str)
            mask = df_norm[region_col].notna()
            df_norm.loc[mask, region_col] = (
                df_norm.loc[mask, region_col]
                .str.strip()
                .str.replace('South-eastern', 'Southeastern', regex=False)
                .str.replace(r'\s{2,}', ' ', regex=True)
            )
            df_norm[region_col] = df_norm[region_col].replace(REGION_MAPPING)
            if region_col != 'Region':
                df_norm = df_norm.rename(columns={region_col: 'Region'})
        except Exception as e:
            print(f"Advertencia: Error normalizando columna de región '{region_col}': {e}")
            if region_col != 'Region':
                df_norm = df_norm.rename(columns={region_col: 'Region'})
    return df_norm

def check_data_quality(df, year):
    print(f"\n--- Calidad de datos {year} ---")
    print(f"Total de filas: {len(df)}")
    if 'Country' in df.columns:
        null_countries = df['Country'].isna().sum()
        unique_countries = df['Country'].nunique()
        print(f"Países únicos: {unique_countries}")
        print(f"Países nulos: {null_countries}")
        if null_countries > 0:
            print("Advertencia: Hay países nulos")
    if 'Region' in df.columns:
        null_regions = df['Region'].isna().sum()
        unique_regions = df['Region'].nunique()
        print(f"Regiones únicas: {unique_regions}")
        print(f"Regiones nulas: {null_regions}")

for year, df in [('2015', y_2015), ('2016', y_2016), ('2017', y_2017), ('2018', y_2018), ('2019', y_2019)]:
    print(f"Columnas en {year}: {df.columns.tolist()}")

y_2015 = normalize_data(y_2015)
y_2016 = normalize_data(y_2016)
y_2017 = normalize_data(y_2017)
y_2018 = normalize_data(y_2018)
y_2019 = normalize_data(y_2019)

check_data_quality(y_2015, '2015')
check_data_quality(y_2016, '2016')
check_data_quality(y_2017, '2017')
check_data_quality(y_2018, '2018')
check_data_quality(y_2019, '2019')

KEEP_FOR_MODEL = [
    "Country",
    "Region",
    "Year",
    "Happiness_Score",
    "GDP_per_Capita",
    "Social_Support",
    "Healthy_Life_Expectancy",
    "Freedom",
    "Generosity",
    "Perceptions_of_Corruption",
]

df_2015_clean = y_2015[KEEP_FOR_MODEL]
df_2016_clean = y_2016[KEEP_FOR_MODEL]
df_2017_clean = y_2017[KEEP_FOR_MODEL]
df_2018_clean = y_2018[KEEP_FOR_MODEL]
df_2019_clean = y_2019[KEEP_FOR_MODEL]

df_complete = pd.concat([
    df_2015_clean,
    df_2016_clean,
    df_2017_clean,
    df_2018_clean,
    df_2019_clean
], ignore_index=True)

print(f"DataFrame completo creado: {len(df_complete)} registros")
print(f"Estructura: {df_complete.shape}")
print(f"Años incluidos: {sorted(df_complete['Year'].unique())}")

# Caps y redondeo
df_complete.loc[df_complete["GDP_per_Capita"] > 2.0, "GDP_per_Capita"] = 2.0
df_complete.loc[df_complete["Social_Support"] > 1.0, "Social_Support"] = 1.0
df_complete.loc[df_complete["Healthy_Life_Expectancy"] > 1.0, "Healthy_Life_Expectancy"] = 1.0
df_complete.loc[df_complete["Freedom"] > 1.0, "Freedom"] = 1.0
df_complete.loc[df_complete["Generosity"] > 0.8, "Generosity"] = 0.8
df_complete.loc[df_complete["Perceptions_of_Corruption"] > 1.0, "Perceptions_of_Corruption"] = 1.0
df_complete = df_complete.round(5)

# Fix UAE 2018
_ = df_complete.loc[
    df_complete["Country"] == "United Arab Emirates",
    ["Year", "Country", "Perceptions_of_Corruption"]
]
mean_uae = df_complete.loc[
    (df_complete["Country"] == "United Arab Emirates") &
    (df_complete["Perceptions_of_Corruption"].notna()),
    "Perceptions_of_Corruption"
].mean()
print("Promedio UAE:", round(mean_uae, 4))
df_complete.loc[
    (df_complete["Country"] == "United Arab Emirates") &
    (df_complete["Year"] == 2018),
    "Perceptions_of_Corruption"
] = mean_uae

# Guarda el CSV limpio (misma ruta que usabas)
# df_complete.to_csv('../data/happiness_2015to2019_cleaned.csv', index=False)

# ============================
# ENVÍO A KAFKA (sin tocar tu limpieza)
# ============================

# Crea el productor Kafka
producer = KafkaProducer(
    bootstrap_servers=[BOOTSTRAP],
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    key_serializer=lambda k: str(k).encode("utf-8")
)

print(f"Enviando datos a Kafka (topic='{TOPIC}') ...")
sent = 0
for _, row in df_complete.iterrows():
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
    sent += 1
    time.sleep(0.2)  # simula streaming

producer.flush()
producer.close()
print(f"Todos los registros enviados. Total: {sent}")
