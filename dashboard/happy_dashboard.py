# dashboard/the_happy_dashboard.py
import os
import numpy as np
import pandas as pd
import joblib

from dotenv import load_dotenv
import mysql.connector

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc

# =========================
# Config & Data Loading
# =========================
load_dotenv()

# Paths relativos asumiendo estructura:
# WORKSHOP-3/
#   data/happiness_2015to2019_cleaned.csv
#   model/happiness_regression.pkl
#   dashboard/the_happy_dashboard.py
DATA_CSV = os.getenv("DATA_CSV", "../data/happiness_2015to2019_cleaned.csv")
MODEL_PKL = os.getenv("MODEL_PKL", "../model/happiness_regression.pkl")

# MySQL (Workbench)
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "happiness")

FEATURE_ORDER = [
    "GDP_per_Capita",
    "Social_Support",
    "Healthy_Life_Expectancy",
    "Freedom",
    "Generosity",
    "Perceptions_of_Corruption",
]

# --- Cargar CSV limpio (para EDA y baseline) ---
df_clean = pd.read_csv(DATA_CSV)
df_clean["Year"] = df_clean["Year"].astype(int)

# --- Cargar modelo (para coeficientes) ---
model = joblib.load(MODEL_PKL)

def get_mysql_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
    )

def load_predictions():
    """
    Lee la tabla predictions del Consumer (MySQL).
    """
    try:
        conn = get_mysql_connection()
        q = """
        SELECT
            Country, Region, Year,
            GDP_per_Capita, Social_Support, Healthy_Life_Expectancy,
            Freedom, Generosity, Perceptions_of_Corruption,
            Predicted_Score, Actual_Score, ts
        FROM predictions
        ORDER BY ts ASC
        """
        dfp = pd.read_sql(q, conn)
        conn.close()
        if not dfp.empty:
            dfp["Year"] = dfp["Year"].astype(int)
            dfp["ts"] = pd.to_datetime(dfp["ts"])
        return dfp
    except Exception as e:
        print("WARN: no se pudo leer MySQL predictions:", e)
        return pd.DataFrame(
            columns=[
                "Country","Region","Year",
                *FEATURE_ORDER,"Predicted_Score","Actual_Score","ts"
            ]
        )

def compute_metrics(dfp: pd.DataFrame):
    """
    Calcula métricas globales y por año a partir de predictions.
    """
    if dfp.empty:
        return {
            "global": {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "n": 0},
            "by_year": pd.DataFrame(columns=["Year","R2","MAE","RMSE","N"])
        }

    y_true = dfp["Actual_Score"].values
    y_pred = dfp["Predicted_Score"].values

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    rows = []
    for y, g in dfp.groupby("Year"):
        yt = g["Actual_Score"].values
        yp = g["Predicted_Score"].values
        mae_y = np.mean(np.abs(yt - yp))
        rmse_y = np.sqrt(np.mean((yt - yp) ** 2))
        ss_res_y = np.sum((yt - yp) ** 2)
        ss_tot_y = np.sum((yt - np.mean(yt)) ** 2)
        r2_y = 1 - ss_res_y / ss_tot_y if ss_tot_y > 0 else np.nan
        rows.append({"Year": int(y), "R2": r2_y, "MAE": mae_y, "RMSE": rmse_y, "N": len(g)})
    by_year = pd.DataFrame(rows).sort_values("Year")

    return {"global": {"r2": r2, "mae": mae, "rmse": rmse, "n": len(dfp)}, "by_year": by_year}

def linreg_coefficients():
    """
    Extrae coeficientes si es LinearRegression; si no, los deja vacíos.
    """
    coefs = {}
    try:
        if hasattr(model, "coef_"):
            beta = model.coef_
            coefs = dict(zip(FEATURE_ORDER, beta))
    except Exception as e:
        print("WARN: no se pudo leer coeficientes:", e)
    return coefs

def fmt3(x):
    """Formatea números a 3 decimales, o '—' si es NaN."""
    return f"{x:.3f}" if pd.notna(x) else "—"

# =========================
# Dash App
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "The Happy Dashboard"

# Dropdown options
years = sorted(df_clean["Year"].unique().tolist())
regions = ["All"] + sorted(df_clean["Region"].dropna().unique().tolist())
countries = ["All"] + sorted(df_clean["Country"].dropna().unique().tolist())

# Layout Components
def kpi_card(title, value, color="primary", subtitle=None):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-muted"),
            html.H3(value, className=f"text-{color}"),
            html.Div(subtitle or "", className="small text-muted")
        ]),
        className="mb-3 shadow-sm rounded-3"
    )

header = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("😊 The Happy Dashboard", className="ms-2"),
        html.Span("World Happiness Report — EDA · Model · Streaming · Report", className="text-muted")
    ]),
    class_name="mb-3 shadow-sm bg-light border-bottom",
    sticky="top"
)

controls = dbc.Card(
    dbc.CardBody([
        html.Div([
            html.Label("Year"),
            dcc.Dropdown(options=[{"label": y, "value": y} for y in years],
                         value=years[-1], id="year_dd", clearable=False)
        ], className="mb-2"),
        html.Div([
            html.Label("Region"),
            dcc.Dropdown(options=[{"label": r, "value": r} for r in regions],
                         value="All", id="region_dd", clearable=False)
        ], className="mb-2"),
        html.Div([
            html.Label("Country"),
            dcc.Dropdown(options=[{"label": c, "value": c} for c in countries],
                         value="All", id="country_dd", clearable=False)
        ], className="mb-2"),
        html.Hr(),
        html.Div([
            html.Label("Auto-refresh streaming (s)"),
            dcc.Input(id="refresh_secs", type="number", min=0, max=30, step=1, value=5),
            html.Div(className="small text-muted", children="0 = desactivado")
        ])
    ]),
    className="mb-3"
)

app.layout = dbc.Container([
    header,
    dbc.Row([
        dbc.Col(controls, md=3),
        dbc.Col([
            dbc.Row(id="kpi_row"),
            dbc.Tabs([
                dbc.Tab(label="Overview", tab_id="tab-overview", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="pred_vs_real_scatter"), md=7),
                        dbc.Col(dcc.Graph(id="error_by_year_bar"), md=5),
                    ])
                ]),
                dbc.Tab(label="EDA", tab_id="tab-eda", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="scatter_gdp_happy"), md=7),
                        dbc.Col(dcc.Graph(id="corr_heatmap"), md=5),
                    ])
                ]),
                dbc.Tab(label="Model", tab_id="tab-model", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="coef_bar"), md=6),
                        dbc.Col(dcc.Graph(id="feature_importance_note"), md=6),
                    ])
                ]),
                dbc.Tab(label="Streaming", tab_id="tab-stream", children=[
                    dbc.Row([
                        dbc.Col(dash_table.DataTable(
                            id="pred_table",
                            columns=[{"name": c, "id": c} for c in
                                     ["ts", "Country", "Year", "Predicted_Score", "Actual_Score"]],
                            page_size=10, style_table={"overflowX": "auto"},
                            sort_action="native"
                        ), width=12)
                    ])
                ]),
                dbc.Tab(label="Report", tab_id="tab-report", children=[
                    dbc.Row(dbc.Col(dcc.Markdown(id="report_md"), md=12))
                ]),
            ], id="tabs", active_tab="tab-overview", className="mb-4")
        ], md=9)
    ]),
    dcc.Interval(id="stream_interval", interval=5_000, n_intervals=0)  # default 5s
], fluid=True)

# =========================
# Callbacks
# =========================
@app.callback(
    Output("stream_interval", "interval"),
    Input("refresh_secs", "value"),
)
def update_interval(secs):
    try:
        secs = int(secs or 0)
    except Exception:
        secs = 0
    if secs <= 0:
        return 24*60*60*1000  # desactivado (1 día)
    return secs * 1000

@app.callback(
    Output("kpi_row", "children"),
    Output("pred_vs_real_scatter", "figure"),
    Output("error_by_year_bar", "figure"),
    Output("scatter_gdp_happy", "figure"),
    Output("corr_heatmap", "figure"),
    Output("coef_bar", "figure"),
    Output("feature_importance_note", "figure"),
    Output("pred_table", "data"),
    Output("report_md", "children"),
    Input("year_dd", "value"),
    Input("region_dd", "value"),
    Input("country_dd", "value"),
    Input("stream_interval", "n_intervals"),
)
def refresh_dashboard(year_val, region_val, country_val, _n):
    # --- cargar predicciones (streaming) ---
    dfp = load_predictions()

    # --- filtros base para EDA (df_clean) ---
    df_eda = df_clean.copy()
    if year_val:
        df_eda = df_eda[df_eda["Year"] == year_val]
    if region_val and region_val != "All":
        df_eda = df_eda[df_eda["Region"] == region_val]
    if country_val and country_val != "All":
        df_eda = df_eda[df_eda["Country"] == country_val]

    # KPIs (usar predictions si hay; si no, basarse en df_clean)
    metrics = compute_metrics(dfp)
    kpi_cards = dbc.Row([
        dbc.Col(kpi_card("Registros en streaming", f"{metrics['global']['n']:,}",
                         "primary", "Tabla: predictions (MySQL)"), md=3),
        dbc.Col(kpi_card("R² (global)", fmt3(metrics["global"]["r2"]), "success"), md=3),
        dbc.Col(kpi_card("MAE (global)", fmt3(metrics["global"]["mae"]), "warning"), md=3),
        dbc.Col(kpi_card("RMSE (global)", fmt3(metrics["global"]["rmse"]), "danger"), md=3),
    ])

    # --- Overview: Pred vs Real ---
    if not dfp.empty:
        fig_pvr = px.scatter(
            dfp, x="Actual_Score", y="Predicted_Score",
            color="Year", hover_data=["Country","Region","ts"],
            trendline="ols", title="Predicted vs Actual (streaming)"
        )
        fig_pvr.add_trace(go.Scatter(
            x=[dfp["Actual_Score"].min(), dfp["Actual_Score"].max()],
            y=[dfp["Actual_Score"].min(), dfp["Actual_Score"].max()],
            mode="lines", name="Ideal", line=dict(dash="dash")
        ))
    else:
        fig_pvr = go.Figure().update_layout(title="Predicted vs Actual — esperando datos de streaming")

    # --- Overview: Error por año ---
    if not metrics["by_year"].empty:
        fig_err_year = px.bar(
            metrics["by_year"], x="Year", y="RMSE", text="N",
            title="RMSE por Año (streaming)"
        )
    else:
        fig_err_year = go.Figure().update_layout(title="RMSE por Año — sin datos aún")

    # --- EDA: Scatter GDP vs Happiness ---
    if not df_eda.empty:
        fig_scatter = px.scatter(
            df_eda, x="GDP_per_Capita", y="Happiness_Score",
            color="Region", hover_data=["Country","Year"],
            trendline="ols",
            title=f"GDP vs Happiness — Filtro: Year={year_val}, Region={region_val}, Country={country_val}"
        )
    else:
        fig_scatter = go.Figure().update_layout(title="GDP vs Happiness — sin datos para el filtro actual")

    # --- EDA: Correlaciones (features + target) ---
    corr_cols = FEATURE_ORDER + ["Happiness_Score"]
    corr_df = df_clean[corr_cols].corr(numeric_only=True)
    fig_corr = px.imshow(
        corr_df, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
        title="Matriz de Correlación (dataset limpio)"
    )

    # --- Model: Coeficientes ---
    coefs = linreg_coefficients()
    if coefs:
        coef_ser = pd.Series(coefs).sort_values()
        fig_coef = px.bar(
            coef_ser, orientation="h",
            title="Coeficientes del Modelo (Linear Regression)",
            labels={"value":"Coeficiente","index":"Feature"}
        )
    else:
        fig_coef = go.Figure().update_layout(title="Coeficientes no disponibles (modelo no lineal o no compatible)")

    # Nota / gráfico vacío con texto
    fig_note = go.Figure()
    fig_note.add_annotation(
        text=("Nota: Los coeficientes muestran la influencia lineal de cada feature\n"
              "sobre el Happiness_Score (mayor coeficiente ⇒ mayor contribución positiva,\n"
              "manteniendo constantes las otras variables)."),
        xref="paper", yref="paper", x=0, y=0.5, showarrow=False
    )
    fig_note.update_layout(title="Interpretación de coeficientes", xaxis_visible=False, yaxis_visible=False)

    # --- Tabla streaming (últimos) ---
    if not dfp.empty:
        dfp_tbl = dfp[["ts","Country","Year","Predicted_Score","Actual_Score"]].sort_values("ts", ascending=False)
        data_tbl = dfp_tbl.head(200).to_dict("records")
    else:
        data_tbl = []

    # --- Report (Markdown) ---
    n_countries = df_clean["Country"].nunique()
    n_years = df_clean["Year"].nunique()

    r2g = fmt3(metrics["global"]["r2"])
    maeg = fmt3(metrics["global"]["mae"])
    rmseg = fmt3(metrics["global"]["rmse"])

    report = f"""
### Report — World Happiness (2015–2019)

**Datasets & Scope**  
Se integraron 5 archivos (2015–2019), con **{n_countries} países** a lo largo de **{n_years} años**.  
Se homogenizaron columnas (nombres, rangos oficiales, decimales) y se normalizaron `Country/Region`.  
Se corrigieron outliers por truncado a límites oficiales (p. ej., `Social_Support ≤ 1.0`).

**EDA — Key Findings**  
- Tendencia positiva clara entre **GDP_per_Capita** y **Happiness_Score**.  
- Alta correlación adicional con **Social_Support** y **Healthy_Life_Expectancy**.  
- Variables como **Generosity** y **Perceptions_of_Corruption** aportan señal pero menor.

**Model — Selection & Training**  
Se usó **Linear Regression** con split **70/30** y features: {", ".join(FEATURE_ORDER)}.  
El modelo se entrenó sobre el dataset limpio y se serializó en `happiness_regression.pkl`.

**Evaluation (Streaming)**  
Con base en la tabla `predictions` (MySQL), se obtuvieron métricas globales:
- **R²**: {r2g}
- **MAE**: {maeg}
- **RMSE**: {rmseg}

Por año (ver barra): diferencias reflejan variación en cobertura/países y ruido en el stream.

**Streaming Process**  
- **Producer** streamea registros del CSV limpio (uno a uno) al tópico Kafka.  
- **Consumer** recibe, **carga el modelo**, **predice** y **persiste** en MySQL (`predictions`).  
- Dashboard refresca cada *X* segundos y muestra comparación **Pred vs Real**, métricas y log.

**Conclusion**  
El pipeline **ETL → Model → Kafka → DB → Dashboard** funciona de extremo a extremo.  
Próximos pasos: probar modelos no lineales (Random Forest/GB), validación cruzada y features adicionales (codificar `Region`).
"""

    return (
        kpi_cards,
        fig_pvr,
        fig_err_year,
        fig_scatter,
        fig_corr,
        fig_coef,
        fig_note,
        data_tbl,
        report
    )

# =========================
# Main
# =========================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
