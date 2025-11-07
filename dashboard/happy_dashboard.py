# import os
# import numpy as np
# import pandas as pd
# import joblib
# from datetime import datetime

# from dotenv import load_dotenv
# import mysql.connector

# import plotly.express as px
# import plotly.graph_objects as go

# from dash import Dash, dcc, html, dash_table, Input, Output
# import dash_bootstrap_components as dbc

# # ====== Opcional (para algunos gráficos de Model) ======
# from sklearn.inspection import permutation_importance
# from sklearn.model_selection import train_test_split
# from sklearn.utils import resample
# try:
#     import statsmodels.api as sm
#     from statsmodels.stats.outliers_influence import variance_inflation_factor
#     HAS_STATSMODELS = True
# except Exception:
#     HAS_STATSMODELS = False
# # =======================================================

# # =========================
# # Config & Data Loading
# # =========================
# load_dotenv()

# DATA_CSV  = os.getenv("DATA_CSV",  "../data/happiness_2015to2019_cleaned.csv")
# MODEL_PKL = os.getenv("MODEL_PKL", "../model/happiness_regression.pkl")

# MYSQL_HOST     = os.getenv("MYSQL_HOST", "localhost")
# MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
# MYSQL_USER     = os.getenv("MYSQL_USER", "root")
# MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
# MYSQL_DB       = os.getenv("MYSQL_DB", "happiness")

# FEATURE_ORDER = [
#     "GDP_per_Capita",
#     "Social_Support",
#     "Healthy_Life_Expectancy",
#     "Freedom",
#     "Generosity",
#     "Perceptions_of_Corruption",
# ]

# df_clean = pd.read_csv(DATA_CSV)
# df_clean["Year"] = df_clean["Year"].astype(int)

# # Matrices para Model
# X_ALL = df_clean[FEATURE_ORDER].copy()
# y_ALL = df_clean["Happiness_Score"].copy()

# # Modelo entrenado
# model = joblib.load(MODEL_PKL)

# def get_mysql_connection():
#     return mysql.connector.connect(
#         host=MYSQL_HOST, port=MYSQL_PORT,
#         user=MYSQL_USER, password=MYSQL_PASSWORD,
#         database=MYSQL_DB,
#     )

# def load_predictions():
#     """
#     Lee la tabla predictions y devuelve SOLO los registros con Data_Set='test'.
#     Si la columna no existe (DB antigua), hace fallback sin filtro y luego
#     filtra en pandas si está disponible.
#     """
#     try:
#         conn = get_mysql_connection()

#         # Intento principal: filtrar en SQL
#         q = """
#         SELECT
#             Country, Region, Year,
#             GDP_per_Capita, Social_Support, Healthy_Life_Expectancy,
#             Freedom, Generosity, Perceptions_of_Corruption,
#             Predicted_Score, Actual_Score, ts, Data_Set
#         FROM predictions
#         WHERE Data_Set = 'test'
#         ORDER BY ts ASC
#         """
#         try:
#             dfp = pd.read_sql(q, conn)
#         except Exception:
#             # Fallback si aún no existe Data_Set en tu tabla
#             q_legacy = """
#             SELECT
#                 Country, Region, Year,
#                 GDP_per_Capita, Social_Support, Healthy_Life_Expectancy,
#                 Freedom, Generosity, Perceptions_of_Corruption,
#                 Predicted_Score, Actual_Score, ts
#             FROM predictions
#             ORDER BY ts ASC
#             """
#             dfp = pd.read_sql(q_legacy, conn)

#         conn.close()

#         if dfp.empty:
#             return pd.DataFrame(
#                 columns=[
#                     "Country","Region","Year",
#                     *FEATURE_ORDER,"Predicted_Score","Actual_Score","ts","Data_Set"
#                 ]
#             )

#         # Tipos
#         if "Year" in dfp.columns:
#             dfp["Year"] = dfp["Year"].astype(int)
#         if "ts" in dfp.columns:
#             dfp["ts"] = pd.to_datetime(dfp["ts"])

#         # Si no se pudo filtrar en SQL pero la columna existe, filtramos aquí
#         if "Data_Set" in dfp.columns:
#             dfp = dfp[dfp["Data_Set"].astype(str).str.lower() == "test"]

#         return dfp

#     except Exception as e:
#         print("WARN: no se pudo leer MySQL predictions:", e)
#         return pd.DataFrame(
#             columns=[
#                 "Country","Region","Year",
#                 *FEATURE_ORDER,"Predicted_Score","Actual_Score","ts","Data_Set"
#             ]
#         )

# def compute_metrics(dfp: pd.DataFrame):
#     if dfp.empty:
#         return {"global":{"r2":np.nan,"mae":np.nan,"rmse":np.nan,"n":0},
#                 "by_year": pd.DataFrame(columns=["Year","R2","MAE","RMSE","N"])}

#     y_true = dfp["Actual_Score"].values
#     y_pred = dfp["Predicted_Score"].values

#     mae  = np.mean(np.abs(y_true - y_pred))
#     rmse = np.sqrt(np.mean((y_true - y_pred)**2))
#     ss_res, ss_tot = np.sum((y_true-y_pred)**2), np.sum((y_true-np.mean(y_true))**2)
#     r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan

#     rows = []
#     for y, g in dfp.groupby("Year"):
#         yt, yp = g["Actual_Score"].values, g["Predicted_Score"].values
#         mae_y  = np.mean(np.abs(yt-yp))
#         rmse_y = np.sqrt(np.mean((yt-yp)**2))
#         ss_res_y, ss_tot_y = np.sum((yt-yp)**2), np.sum((yt-np.mean(yt))**2)
#         r2_y = 1 - ss_res_y/ss_tot_y if ss_tot_y>0 else np.nan
#         rows.append({"Year":int(y),"R2":r2_y,"MAE":mae_y,"RMSE":rmse_y,"N":len(g)})
#     by_year = pd.DataFrame(rows).sort_values("Year")

#     return {"global":{"r2":r2,"mae":mae,"rmse":rmse,"n":len(dfp)}, "by_year":by_year}

# def linreg_coefficients():
#     coefs = {}
#     try:
#         if hasattr(model, "coef_"):
#             coefs = dict(zip(FEATURE_ORDER, model.coef_))
#     except Exception as e:
#         print("WARN: no se pudo leer coeficientes:", e)
#     return coefs

# def fmt3(x):  # 3 decimales o raya
#     return f"{x:.3f}" if pd.notna(x) else "—"

# # =========================
# # Dash App
# # =========================
# app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
# app.title = "The Happy Dashboard"

# years     = sorted(df_clean["Year"].unique().tolist())
# regions   = ["All"] + sorted(df_clean["Region"].dropna().unique().tolist())
# countries = ["All"] + sorted(df_clean["Country"].dropna().unique().tolist())

# def kpi_card(title, value, color="primary", subtitle=None):
#     return dbc.Card(
#         dbc.CardBody([
#             html.H6(title, className="text-muted"),
#             html.H3(value, className=f"text-{color}"),
#             html.Div(subtitle or "", className="small text-muted")
#         ]),
#         className="mb-3 shadow-sm rounded-3"
#     )

# header = dbc.Navbar(
#     dbc.Container([
#         dbc.NavbarBrand("😊 The Happy Dashboard", className="ms-2"),
#         html.Span("World Happiness", className="text-muted")
#     ]),
#     class_name="mb-3 shadow-sm bg-light border-bottom",
#     sticky="top"
# )

# # ---- CONTROLES por pestaña ----
# controls_eda = dbc.Card(
#     dbc.CardBody([
#         html.Label("Year"),
#         dcc.Dropdown([{"label":y,"value":y} for y in years], value=years[-1], id="year_dd", clearable=False),
#         html.Br(),
#         html.Label("Region"),
#         dcc.Dropdown([{"label":r,"value":r} for r in regions], value="All", id="region_dd", clearable=False),
#         html.Br(),
#         html.Label("Country"),
#         dcc.Dropdown([{"label":c,"value":c} for c in countries], value="All", id="country_dd", clearable=False),
#     ])
# )

# controls_stream = dbc.Card(
#     dbc.CardBody([
#         html.Label("Auto-refresh streaming (s)"),
#         dcc.Input(id="refresh_secs", type="number", min=0, max=30, step=1, value=5),
#         html.Div("0 = desactivado", className="small text-muted mt-1"),
#     ])
# )

# app.layout = dbc.Container([
#     header,

#     # KPIs
#     dbc.Row(id="kpi_row"),

#     dbc.Tabs([
#         # ================= Overview =================
#         dbc.Tab(label="Overview", tab_id="tab-overview", children=[
#             dbc.Row([
#                 dbc.Col(dcc.Graph(id="pred_vs_real_scatter"), md=6),
#                 dbc.Col(dcc.Graph(id="error_by_year_bar"),   md=6),
#             ]),
#             dbc.Row([
#                 dbc.Col(dcc.Graph(id="resid_vs_pred"), md=6),
#                 dbc.Col(dcc.Graph(id="resid_hist"),    md=6),
#             ]),
#             dbc.Row([
#                 dbc.Col(dcc.Graph(id="calib_scatter"), md=6),
#                 dbc.Col(dcc.Graph(id="top_err_bar"),   md=6),
#             ]),
#             dbc.Row([
#                 dbc.Col(dcc.Graph(id="region_year_heat"), md=6),
#                 dbc.Col(dcc.Graph(id="qq_plot"),         md=6),
#             ]),
#         ]),

#         # ================= EDA =================
#         dbc.Tab(label="EDA", tab_id="tab-eda", children=[
#             dbc.Row([
#                 dbc.Col(controls_eda, md=3),
#                 dbc.Col([
#                     dbc.Row([ dbc.Col(dcc.Graph(id="eda_box_by_region"), md=12) ]),
#                     dbc.Row([ dbc.Col(dcc.Graph(id="eda_bubble"),       md=12) ]),
#                     dbc.Row([
#                         dbc.Col(dcc.Graph(id="eda_region_trend"), md=7),
#                         dbc.Col(dcc.Graph(id="eda_top10"),       md=5),
#                     ]),
#                 ], md=9),
#             ])
#         ]),

#         # ================= Model =================
#         dbc.Tab(label="Model", tab_id="tab-model", children=[
#             dbc.Row([
#                 dbc.Col(dcc.Graph(id="coef_bar"),                 md=6),
#                 dbc.Col(dcc.Graph(id="feature_importance_note"),  md=6),
#             ]),
#             dbc.Row([
#                 dbc.Col(dcc.Graph(id="perm_importance"), md=6),
#                 dbc.Col(dcc.Graph(id="pdp_gdp"),         md=6),
#             ]),
#             dbc.Row([
#                 dbc.Col(dcc.Graph(id="coef_ci"),   md=6),
#                 dbc.Col(dcc.Graph(id="vif_bar"),   md=6),
#             ]),
#             dbc.Row([
#                 dbc.Col(dcc.Graph(id="res_vs_feat"), md=12),
#             ]),
#         ]),

#         # ================= Streaming =================
#         dbc.Tab(label="Streaming", tab_id="tab-stream", children=[
#             dbc.Row([
#                 dbc.Col(controls_stream, md=3),
#                 dbc.Col(dash_table.DataTable(
#                     id="pred_table",
#                     columns=[{"name": c, "id": c}
#                              for c in ["ts","Country","Year","Predicted_Score","Actual_Score"]],
#                     page_size=10, sort_action="native",
#                     style_table={"overflowX":"auto"}
#                 ), md=9),
#             ])
#         ]),
#     ], id="tabs", active_tab="tab-overview", className="mb-4"),

#     dcc.Interval(id="stream_interval", interval=5_000, n_intervals=0)
# ], fluid=True)

# # =========================
# # Callbacks
# # =========================
# # 0) Interval — SOLO Streaming
# @app.callback(Output("stream_interval","interval"), Input("refresh_secs","value"))
# def update_interval(secs):
#     try: secs = int(secs or 0)
#     except Exception: secs = 0
#     return (24*60*60*1000) if secs<=0 else secs*1000

# # 1) Overview (KPIs + TODAS las figuras de streaming)
# @app.callback(
#     Output("kpi_row","children"),
#     Output("pred_vs_real_scatter","figure"),
#     Output("error_by_year_bar","figure"),
#     Output("resid_vs_pred","figure"),
#     Output("resid_hist","figure"),
#     Output("calib_scatter","figure"),
#     Output("top_err_bar","figure"),
#     Output("region_year_heat","figure"),
#     Output("qq_plot","figure"),
#     Input("stream_interval","n_intervals"),
# )
# def refresh_overview(_n):
#     dfp = load_predictions()
#     metrics = compute_metrics(dfp)

#     kpis = dbc.Row([
#         dbc.Col(kpi_card("Registros en streaming",
#                          f"{metrics['global']['n']:,}", "primary",
#                          "Tabla: predictions (MySQL)"), md=3),
#         dbc.Col(kpi_card("R² (global)",  fmt3(metrics["global"]["r2"]),  "success"), md=3),
#         dbc.Col(kpi_card("MAE (global)", fmt3(metrics["global"]["mae"]), "warning"), md=3),
#         dbc.Col(kpi_card("RMSE (global)",fmt3(metrics["global"]["rmse"]), "danger"), md=3),
#     ])

#     # ---- Pred vs Real
#     if not dfp.empty:
#         fig_pvr = px.scatter(
#             dfp, x="Actual_Score", y="Predicted_Score",
#             color="Year", hover_data=["Country","Region","ts"],
#             trendline="ols", title="Predicted vs Actual (streaming · TEST)"
#         )
#         fig_pvr.add_trace(go.Scatter(
#             x=[dfp["Actual_Score"].min(), dfp["Actual_Score"].max()],
#             y=[dfp["Actual_Score"].min(), dfp["Actual_Score"].max()],
#             mode="lines", name="Ideal", line=dict(dash="dash")
#         ))
#     else:
#         fig_pvr = go.Figure().update_layout(title="Predicted vs Actual — esperando datos de streaming")

#     # ---- RMSE por año
#     if not metrics["by_year"].empty:
#         fig_err_year = px.bar(metrics["by_year"], x="Year", y="RMSE", text="N",
#                               title="RMSE por Año (streaming · TEST)")
#     else:
#         fig_err_year = go.Figure().update_layout(title="RMSE por Año — sin datos aún")

#     # Si no hay datos, devuelve placeholders
#     if dfp.empty:
#         empty = go.Figure().update_layout(title="Esperando datos de streaming…")
#         return (kpis, fig_pvr, fig_err_year, empty, empty, empty, empty, empty, empty)

#     # ---- Residuos
#     dfp = dfp.copy()
#     dfp["_resid"] = dfp["Actual_Score"] - dfp["Predicted_Score"]

#     fig_resid_fitted = px.scatter(
#         dfp, x="Predicted_Score", y="_resid", color="Year",
#         hover_data=["Country","Region","Actual_Score","ts"],
#         title="Residuos vs Predicción"
#     )
#     fig_resid_fitted.add_hline(y=0, line_dash="dash")

#     fig_resid_hist = px.histogram(
#         dfp, x="_resid", nbins=40, marginal="box",
#         title="Distribución de errores (Actual − Pred)"
#     )

#     # ---- Calibración (deciles)
#     try:
#         bins = pd.qcut(dfp["Predicted_Score"], q=10, duplicates="drop")
#         cal = dfp.groupby(bins).agg(
#             pred=("Predicted_Score","mean"),
#             real=("Actual_Score","mean"),
#             n=("Actual_Score","size")
#         ).reset_index(drop=True)
#         fig_calib = px.scatter(cal, x="pred", y="real", size="n",
#                                title="Calibración por deciles")
#         fig_calib.add_trace(go.Scatter(x=cal["pred"], y=cal["pred"],
#                                        mode="lines", name="Ideal",
#                                        line=dict(dash="dash")))
#     except Exception:
#         fig_calib = go.Figure().update_layout(title="Calibración — no suficiente variación")

#     # ---- Top-15 mayores errores absolutos
#     topn = dfp.assign(abs_err=(dfp["_resid"]).abs()).nlargest(15, "abs_err")
#     fig_top_err = px.bar(topn, x="abs_err", y="Country", color="Year",
#                          orientation="h", title="Top 15 | Error absoluto por país (streaming)")

#     # ---- Heatmap RMSE Región×Año
#     err_reg = (dfp.groupby(["Region","Year"])["_resid"]
#                  .apply(lambda s: np.sqrt(np.mean(s**2)))
#                  .reset_index(name="RMSE"))
#     pivot = err_reg.pivot(index="Region", columns="Year", values="RMSE")
#     fig_err_heat = px.imshow(pivot, color_continuous_scale="RdBu_r",
#                              aspect="auto", title="RMSE por Región y Año")

#     # ---- Q-Q plot de residuos
#     res = np.sort(dfp["_resid"].values)
#     q = np.sort(np.random.normal(0, res.std(ddof=1) if res.std(ddof=1)>0 else 1e-6, size=len(res)))
#     fig_qq = px.scatter(x=q, y=res, title="Q-Q plot de residuos")
#     fig_qq.add_trace(go.Scatter(x=[q.min(), q.max()], y=[q.min(), q.max()],
#                                 mode="lines", name="45°", line=dict(dash="dash")))

#     return (kpis, fig_pvr, fig_err_year, fig_resid_fitted, fig_resid_hist,
#             fig_calib, fig_top_err, fig_err_heat, fig_qq)

# # 2) EDA — 4 figuras
# @app.callback(
#     Output("eda_box_by_region","figure"),
#     Output("eda_bubble","figure"),
#     Output("eda_region_trend","figure"),
#     Output("eda_top10","figure"),
#     Input("year_dd","value"),
#     Input("region_dd","value"),
#     Input("country_dd","value"),
# )
# def refresh_eda(year_val, region_val, country_val):
#     df_eda = df_clean.copy()
#     if year_val:
#         df_eda = df_eda[df_eda["Year"] == year_val]
#     if region_val and region_val != "All":
#         df_eda = df_eda[df_eda["Region"] == region_val]
#     if country_val and country_val != "All":
#         df_eda = df_eda[df_eda["Country"] == country_val]

#     # Boxplot por región
#     if not df_eda.empty:
#         fig_box_region = px.box(
#             df_eda, x="Region", y="Happiness_Score", points="all",
#             title="Distribución del Happiness Score por Región"
#         ).update_layout(xaxis_title="", yaxis_title="Happiness_Score")
#     else:
#         fig_box_region = go.Figure().update_layout(title="Distribución por Región — sin datos para el filtro actual")

#     # Bubble GDP vs Happiness (size = Social_Support)
#     if not df_eda.empty:
#         fig_bubble = px.scatter(
#             df_eda, x="GDP_per_Capita", y="Happiness_Score",
#             size="Social_Support", color="Region",
#             hover_data=["Country","Year"],
#             title="GDP vs Happiness (tamaño = Social_Support)"
#         )
#     else:
#         fig_bubble = go.Figure().update_layout(title="GDP vs Happiness — sin datos para el filtro actual")

#     # Tendencia por región (promedio por año)
#     trend = (df_clean.groupby(["Year","Region"], as_index=False)["Happiness_Score"]
#              .mean())
#     if region_val and region_val != "All":
#         trend = trend[trend["Region"] == region_val]
#     fig_region_trend = px.line(
#         trend, x="Year", y="Happiness_Score", color="Region",
#         markers=True, title="Tendencia del Happiness (promedio) por Región"
#     )

#     # Top-10 países (promedio 2015–2019)
#     top10 = (df_clean.groupby("Country", as_index=False)["Happiness_Score"]
#              .mean()
#              .sort_values("Happiness_Score", ascending=False)
#              .head(10))
#     fig_top10 = px.bar(
#         top10.sort_values("Happiness_Score"),
#         x="Happiness_Score", y="Country", orientation="h",
#         title="Top 10 países más felices (promedio 2015–2019)",
#         labels={"Happiness_Score":"Promedio","Country":"País"}
#     )

#     return fig_box_region, fig_bubble, fig_region_trend, fig_top10

# # 3) Model — diagnósticos/explicabilidad
# @app.callback(
#     Output("coef_bar","figure"),
#     Output("feature_importance_note","figure"),
#     Output("perm_importance","figure"),
#     Output("pdp_gdp","figure"),
#     Output("coef_ci","figure"),
#     Output("vif_bar","figure"),
#     Output("res_vs_feat","figure"),
#     Input("tabs","active_tab"),  # simple trigger
# )
# def refresh_model(_tab):
#     # Coeficientes
#     coefs = linreg_coefficients()
#     if coefs:
#         coef_ser = pd.Series(coefs).sort_values()
#         fig_coef = px.bar(coef_ser, orientation="h",
#                           title="Coeficientes del Modelo (Linear Regression)",
#                           labels={"value":"Coeficiente","index":"Feature"})
#     else:
#         fig_coef = go.Figure().update_layout(title="Coeficientes no disponibles")

#     fig_note = go.Figure()
#     fig_note.add_annotation(
#         text=("Nota: Los coeficientes muestran la influencia lineal de cada feature\n"
#               "sobre el Happiness_Score (mayor coeficiente ⇒ mayor contribución positiva,\n"
#               "manteniendo constantes las otras variables)."),
#         xref="paper", yref="paper", x=0, y=0.5, showarrow=False
#     )
#     fig_note.update_layout(title="Interpretación de coeficientes",
#                            xaxis_visible=False, yaxis_visible=False)

#     # Split para métricas de modelo
#     Xtr, Xte, ytr, yte = train_test_split(X_ALL, y_ALL, test_size=0.3, random_state=42)

#     # Importancia por permutación
#     try:
#         pi = permutation_importance(model, Xte, yte, n_repeats=20,
#                                     random_state=0, scoring="neg_mean_squared_error")
#         imp = pd.Series(pi.importances_mean, index=X_ALL.columns).sort_values()
#         fig_pi = px.bar(imp, orientation="h", title="Importancia por permutación (↓RMSE)")
#     except Exception as e:
#         fig_pi = go.Figure().update_layout(title=f"Importancia por permutación — no disponible ({e})")

#     # PDP para GDP_per_Capita (promedio sobre otras variables)
#     def pdp_curve(model_, X, feature, grid=40):
#         xs = np.linspace(X[feature].min(), X[feature].max(), grid)
#         Xc = X.copy()
#         preds = []
#         for v in xs:
#             Xc[feature] = v
#             preds.append(model_.predict(Xc).mean())
#         return pd.DataFrame({feature: xs, "pred": preds})
#     try:
#         pdp1 = pdp_curve(model, Xtr, "GDP_per_Capita")
#         fig_pdp = px.line(pdp1, x="GDP_per_Capita", y="pred",
#                           title="PDP: GDP_per_Capita → Predicción")
#     except Exception as e:
#         fig_pdp = go.Figure().update_layout(title=f"PDP — no disponible ({e})")

#     # IC 95% de coeficientes (bootstrap)
#     try:
#         B = 200
#         coef_samples=[]
#         for _ in range(B):
#             Xb, yb = resample(Xtr, ytr, replace=True)
#             mb = model.__class__().fit(Xb, yb)
#             coef_samples.append(mb.coef_)
#         coef_df = pd.DataFrame(coef_samples, columns=X_ALL.columns)
#         ci = coef_df.quantile([0.025,0.975]).T.reset_index().rename(
#             columns={"index":"Feature",0.025:"low",0.975:"high"}
#         )
#         ci["coef"] = pd.Series(model.coef_, index=X_ALL.columns).values
#         fig_ci = px.scatter(ci, x="coef", y="Feature",
#                             error_x=ci["coef"]-ci["low"],
#                             error_x_minus=ci["high"]-ci["coef"],
#                             title="Coeficientes con IC 95% (bootstrap)")
#     except Exception as e:
#         fig_ci = go.Figure().update_layout(title=f"IC de coeficientes — no disponible ({e})")

#     # VIF
#     if HAS_STATSMODELS:
#         try:
#             Xc = sm.add_constant(X_ALL)
#             vif_vals = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
#             vif = pd.DataFrame({"Feature": Xc.columns, "VIF": vif_vals})
#             vif = vif[vif["Feature"] != "const"].sort_values("VIF", ascending=False)
#             fig_vif = px.bar(vif, x="VIF", y="Feature", orientation="h", title="VIF por feature")
#         except Exception as e:
#             fig_vif = go.Figure().update_layout(title=f"VIF — error ({e})")
#     else:
#         fig_vif = go.Figure()
#         fig_vif.add_annotation(text="Instala statsmodels para ver VIF: pip install statsmodels",
#                                xref="paper", yref="paper", x=0, y=0.5, showarrow=False)
#         fig_vif.update_layout(title="VIF — no disponible")

#     # Residuos vs una feature (Healthy_Life_Expectancy)
#     try:
#         pred_all = model.predict(X_ALL)
#         res_all = y_ALL - pred_all
#         fig_res_feat = px.scatter(x=X_ALL["Healthy_Life_Expectancy"], y=res_all,
#                                   labels={"x":"Healthy_Life_Expectancy","y":"Residuo"},
#                                   title="Residuos vs Healthy_Life_Expectancy")
#         fig_res_feat.add_hline(y=0, line_dash="dash")
#     except Exception as e:
#         fig_res_feat = go.Figure().update_layout(title=f"Residuos vs feature — no disponible ({e})")

#     return (fig_coef, fig_note, fig_pi, fig_pdp, fig_ci, fig_vif, fig_res_feat)

# # 4) Streaming — tabla en vivo
# @app.callback(
#     Output("pred_table","data"),
#     Input("stream_interval","n_intervals"),
# )
# def refresh_stream_table(_n):
#     dfp = load_predictions()
#     if dfp.empty:
#         return []
#     dfp_tbl = dfp[["ts","Country","Year","Predicted_Score","Actual_Score"]].sort_values("ts", ascending=False)
#     return dfp_tbl.head(200).to_dict("records")

# # =========================
# # Main
# # =========================
# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=8050)
# dashboard/the_happy_dashboard.py
import os
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from dotenv import load_dotenv
import mysql.connector

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc

# ====== Opcional (para algunos gráficos de Model) ======
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False
# =======================================================

# =========================
# Config & Data Loading
# =========================
load_dotenv()

DATA_CSV  = os.getenv("DATA_CSV",  "../data/happiness_2015to2019_cleaned.csv")
MODEL_PKL = os.getenv("MODEL_PKL", "../model/happiness_regression.pkl")

MYSQL_HOST     = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER     = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB       = os.getenv("MYSQL_DB", "happiness")

FEATURE_ORDER = [
    "GDP_per_Capita",
    "Social_Support",
    "Healthy_Life_Expectancy",
    "Freedom",
    "Generosity",
    "Perceptions_of_Corruption",
]

df_clean = pd.read_csv(DATA_CSV)
df_clean["Year"] = df_clean["Year"].astype(int)

# Matrices para Model
X_ALL = df_clean[FEATURE_ORDER].copy()
y_ALL = df_clean["Happiness_Score"].copy()

# Modelo entrenado
model = joblib.load(MODEL_PKL)

def get_mysql_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT,
        user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=MYSQL_DB,
    )

def load_predictions(dataset: str = "test"):
    """
    Lee la tabla predictions y aplica filtro por Data_Set:
    dataset ∈ {"test","train","all"}.
    Si la columna no existe, hace fallback y filtra en pandas si se puede.
    """
    dataset = (dataset or "test").lower()
    try:
        conn = get_mysql_connection()

        if dataset in {"test", "train"}:
            q = f"""
            SELECT Country, Region, Year,
                   GDP_per_Capita, Social_Support, Healthy_Life_Expectancy,
                   Freedom, Generosity, Perceptions_of_Corruption,
                   Predicted_Score, Actual_Score, ts, Data_Set
            FROM predictions
            WHERE Data_Set = '{dataset}'
            ORDER BY ts ASC
            """
        else:  # "all"
            q = """
            SELECT Country, Region, Year,
                   GDP_per_Capita, Social_Support, Healthy_Life_Expectancy,
                   Freedom, Generosity, Perceptions_of_Corruption,
                   Predicted_Score, Actual_Score, ts,
                   COALESCE(Data_Set,'unknown') AS Data_Set
            FROM predictions
            ORDER BY ts ASC
            """

        try:
            dfp = pd.read_sql(q, conn)
        except Exception:
            # Fallback si la columna Data_Set no existe
            q_legacy = """
            SELECT Country, Region, Year,
                   GDP_per_Capita, Social_Support, Healthy_Life_Expectancy,
                   Freedom, Generosity, Perceptions_of_Corruption,
                   Predicted_Score, Actual_Score, ts
            FROM predictions
            ORDER BY ts ASC
            """
            dfp = pd.read_sql(q_legacy, conn)

        conn.close()

        if dfp.empty:
            return pd.DataFrame(columns=[
                "Country","Region","Year",*FEATURE_ORDER,
                "Predicted_Score","Actual_Score","ts","Data_Set"
            ])

        if "Year" in dfp.columns:
            dfp["Year"] = dfp["Year"].astype(int)
        if "ts" in dfp.columns:
            dfp["ts"] = pd.to_datetime(dfp["ts"], errors="coerce")

        # Si vino sin filtrar y existe Data_Set, filtramos aquí
        if dataset in {"test","train"} and "Data_Set" in dfp.columns:
            dfp = dfp[dfp["Data_Set"].astype(str).str.lower() == dataset]

        # Normaliza valores inesperados
        if "Data_Set" in dfp.columns:
            dfp["Data_Set"] = dfp["Data_Set"].fillna("unknown").str.lower()

        return dfp

    except Exception as e:
        print("WARN: no se pudo leer MySQL predictions:", e)
        return pd.DataFrame(columns=[
            "Country","Region","Year",*FEATURE_ORDER,
            "Predicted_Score","Actual_Score","ts","Data_Set"
        ])

def compute_metrics(dfp: pd.DataFrame):
    if dfp.empty:
        return {"global":{"r2":np.nan,"mae":np.nan,"rmse":np.nan,"n":0},
                "by_year": pd.DataFrame(columns=["Year","R2","MAE","RMSE","N"])}

    y_true = dfp["Actual_Score"].values
    y_pred = dfp["Predicted_Score"].values

    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ss_res, ss_tot = np.sum((y_true-y_pred)**2), np.sum((y_true-np.mean(y_true))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan

    rows = []
    for y, g in dfp.groupby("Year"):
        yt, yp = g["Actual_Score"].values, g["Predicted_Score"].values
        mae_y  = np.mean(np.abs(yt-yp))
        rmse_y = np.sqrt(np.mean((yt-yp)**2))
        ss_res_y, ss_tot_y = np.sum((yt-yp)**2), np.sum((yt-np.mean(yt))**2)
        r2_y = 1 - ss_res_y/ss_tot_y if ss_tot_y>0 else np.nan
        rows.append({"Year":int(y),"R2":r2_y,"MAE":mae_y,"RMSE":rmse_y,"N":len(g)})
    by_year = pd.DataFrame(rows).sort_values("Year")

    return {"global":{"r2":r2,"mae":mae,"rmse":rmse,"n":len(dfp)}, "by_year":by_year}

def linreg_coefficients():
    coefs = {}
    try:
        if hasattr(model, "coef_"):
            coefs = dict(zip(FEATURE_ORDER, model.coef_))
    except Exception as e:
        print("WARN: no se pudo leer coeficientes:", e)
    return coefs

def fmt3(x):  # 3 decimales o raya
    return f"{x:.3f}" if pd.notna(x) else "—"

# =========================
# Dash App
# =========================
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "The Happy Dashboard"

years     = sorted(df_clean["Year"].unique().tolist())
regions   = ["All"] + sorted(df_clean["Region"].dropna().unique().tolist())
countries = ["All"] + sorted(df_clean["Country"].dropna().unique().tolist())

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
        html.Span("World Happiness", className="text-muted")
    ]),
    class_name="mb-3 shadow-sm bg-light border-bottom",
    sticky="top"
)

# ---- CONTROLES por pestaña ----
controls_eda = dbc.Card(
    dbc.CardBody([
        html.Label("Year"),
        dcc.Dropdown([{"label":y,"value":y} for y in years], value=years[-1], id="year_dd", clearable=False),
        html.Br(),
        html.Label("Region"),
        dcc.Dropdown([{"label":r,"value":r} for r in regions], value="All", id="region_dd", clearable=False),
        html.Br(),
        html.Label("Country"),
        dcc.Dropdown([{"label":c,"value":c} for c in countries], value="All", id="country_dd", clearable=False),
    ])
)

controls_stream = dbc.Card(
    dbc.CardBody([
        html.Label("Conjunto (Train/Test/All)"),
        dcc.Dropdown(
            id="dataset_dd",
            options=[
                {"label":"TEST","value":"test"},
                {"label":"TRAIN","value":"train"},
                {"label":"ALL","value":"all"},
            ],
            value="test", clearable=False
        ),
        html.Br(),
        html.Label("Auto-refresh streaming (s)"),
        dcc.Input(id="refresh_secs", type="number", min=0, max=30, step=1, value=5),
        html.Div("0 = desactivado", className="small text-muted mt-1"),
    ])
)

app.layout = dbc.Container([
    header,

    # KPIs
    dbc.Row(id="kpi_row"),

    dbc.Tabs([
        # ================= Overview =================
        dbc.Tab(label="Overview", tab_id="tab-overview", children=[
            dbc.Row([
                dbc.Col(controls_stream, md=3),
                dbc.Col(dcc.Graph(id="pred_vs_real_scatter"), md=9),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="error_by_year_bar"),   md=6),
                dbc.Col(dcc.Graph(id="resid_vs_pred"), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="resid_hist"),    md=6),
                dbc.Col(dcc.Graph(id="calib_scatter"), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="top_err_bar"),   md=6),
                dbc.Col(dcc.Graph(id="region_year_heat"), md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="qq_plot"),         md=12),
            ]),
        ]),

        # ================= EDA =================
        dbc.Tab(label="EDA", tab_id="tab-eda", children=[
            dbc.Row([
                dbc.Col(controls_eda, md=3),
                dbc.Col([
                    dbc.Row([ dbc.Col(dcc.Graph(id="eda_box_by_region"), md=12) ]),
                    dbc.Row([ dbc.Col(dcc.Graph(id="eda_bubble"),       md=12) ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id="eda_region_trend"), md=7),
                        dbc.Col(dcc.Graph(id="eda_top10"),       md=5),
                    ]),
                ], md=9),
            ])
        ]),

        # ================= Model =================
        dbc.Tab(label="Model", tab_id="tab-model", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="coef_bar"),                 md=6),
                dbc.Col(dcc.Graph(id="feature_importance_note"),  md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="perm_importance"), md=6),
                dbc.Col(dcc.Graph(id="pdp_gdp"),         md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="coef_ci"),   md=6),
                dbc.Col(dcc.Graph(id="vif_bar"),   md=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="res_vs_feat"), md=12),
            ]),
        ]),

        # ================= Streaming =================
        dbc.Tab(label="Streaming", tab_id="tab-stream", children=[
            dbc.Row([
                dbc.Col(controls_stream, md=3),
                dbc.Col(dash_table.DataTable(
                    id="pred_table",
                    columns=[{"name": c, "id": c}
                             for c in ["ts","Country","Year","Predicted_Score","Actual_Score","Data_Set"]],
                    page_size=10, sort_action="native",
                    style_table={"overflowX":"auto"}
                ), md=9),
            ])
        ]),
    ], id="tabs", active_tab="tab-overview", className="mb-4"),

    dcc.Interval(id="stream_interval", interval=5_000, n_intervals=0)
], fluid=True)

# =========================
# Callbacks
# =========================
# 0) Interval — SOLO Streaming
@app.callback(Output("stream_interval","interval"), Input("refresh_secs","value"))
def update_interval(secs):
    try: secs = int(secs or 0)
    except Exception: secs = 0
    return (24*60*60*1000) if secs<=0 else secs*1000

# 1) Overview (usa dataset_dd)
@app.callback(
    Output("kpi_row","children"),
    Output("pred_vs_real_scatter","figure"),
    Output("error_by_year_bar","figure"),
    Output("resid_vs_pred","figure"),
    Output("resid_hist","figure"),
    Output("calib_scatter","figure"),
    Output("top_err_bar","figure"),
    Output("region_year_heat","figure"),
    Output("qq_plot","figure"),
    Input("stream_interval","n_intervals"),
    Input("dataset_dd","value"),
)
def refresh_overview(_n, dataset_val):
    dfp = load_predictions(dataset_val)
    metrics = compute_metrics(dfp)
    ds_label = dataset_val.upper()

    kpis = dbc.Row([
        dbc.Col(kpi_card("Registros (streaming)", f"{metrics['global']['n']:,}",
                         "primary", f"Data_Set: {ds_label}"), md=3),
        dbc.Col(kpi_card("R² (global)",  fmt3(metrics["global"]["r2"]),  "success"), md=3),
        dbc.Col(kpi_card("MAE (global)", fmt3(metrics["global"]["mae"]), "warning"), md=3),
        dbc.Col(kpi_card("RMSE (global)",fmt3(metrics["global"]["rmse"]), "danger"), md=3),
    ])

    # Pred vs Real
    if not dfp.empty:
        fig_pvr = px.scatter(
            dfp, x="Actual_Score", y="Predicted_Score",
            color="Year", hover_data=["Country","Region","ts","Data_Set"],
            trendline="ols", title=f"Predicted vs Actual (streaming · {ds_label})"
        )
        fig_pvr.add_trace(go.Scatter(
            x=[dfp["Actual_Score"].min(), dfp["Actual_Score"].max()],
            y=[dfp["Actual_Score"].min(), dfp["Actual_Score"].max()],
            mode="lines", name="Ideal", line=dict(dash="dash")
        ))
    else:
        fig_pvr = go.Figure().update_layout(title=f"Predicted vs Actual — sin datos ({ds_label})")

    # RMSE por año
    if not metrics["by_year"].empty:
        fig_err_year = px.bar(metrics["by_year"], x="Year", y="RMSE", text="N",
                              title=f"RMSE por Año (streaming · {ds_label})")
    else:
        fig_err_year = go.Figure().update_layout(title=f"RMSE por Año — sin datos ({ds_label})")

    if dfp.empty:
        empty = go.Figure().update_layout(title="Esperando datos…")
        return (kpis, fig_pvr, fig_err_year, empty, empty, empty, empty, empty, empty)

    # Residuos
    dfp = dfp.copy()
    dfp["_resid"] = dfp["Actual_Score"] - dfp["Predicted_Score"]

    fig_resid_fitted = px.scatter(
        dfp, x="Predicted_Score", y="_resid", color="Year",
        hover_data=["Country","Region","Actual_Score","ts","Data_Set"],
        title="Residuos vs Predicción"
    )
    fig_resid_fitted.add_hline(y=0, line_dash="dash")

    fig_resid_hist = px.histogram(
        dfp, x="_resid", nbins=40, marginal="box",
        title="Distribución de errores (Actual − Pred)"
    )

    # Calibración por deciles
    try:
        bins = pd.qcut(dfp["Predicted_Score"], q=10, duplicates="drop")
        cal = dfp.groupby(bins).agg(
            pred=("Predicted_Score","mean"),
            real=("Actual_Score","mean"),
            n=("Actual_Score","size")
        ).reset_index(drop=True)
        fig_calib = px.scatter(cal, x="pred", y="real", size="n",
                               title="Calibración por deciles")
        fig_calib.add_trace(go.Scatter(x=cal["pred"], y=cal["pred"],
                                       mode="lines", name="Ideal",
                                       line=dict(dash="dash")))
    except Exception:
        fig_calib = go.Figure().update_layout(title="Calibración — no suficiente variación")

    # Top errores
    topn = dfp.assign(abs_err=(dfp["_resid"]).abs()).nlargest(15, "abs_err")
    fig_top_err = px.bar(topn, x="abs_err", y="Country", color="Year",
                         orientation="h", title="Top 15 | Error absoluto por país")

    # Heatmap Región×Año
    err_reg = (dfp.groupby(["Region","Year"])["_resid"]
                 .apply(lambda s: np.sqrt(np.mean(s**2)))
                 .reset_index(name="RMSE"))
    pivot = err_reg.pivot(index="Region", columns="Year", values="RMSE")
    fig_err_heat = px.imshow(pivot, color_continuous_scale="RdBu_r",
                             aspect="auto", title="RMSE por Región y Año")

    # Q-Q
    res = np.sort(dfp["_resid"].values)
    q = np.sort(np.random.normal(0, res.std(ddof=1) if res.std(ddof=1)>0 else 1e-6, size=len(res)))
    fig_qq = px.scatter(x=q, y=res, title="Q-Q plot de residuos")
    fig_qq.add_trace(go.Scatter(x=[q.min(), q.max()], y=[q.min(), q.max()],
                                mode="lines", name="45°", line=dict(dash="dash")))

    return (kpis, fig_pvr, fig_err_year, fig_resid_fitted, fig_resid_hist,
            fig_calib, fig_top_err, fig_err_heat, fig_qq)

# 2) EDA — 4 figuras (sin cambios)
@app.callback(
    Output("eda_box_by_region","figure"),
    Output("eda_bubble","figure"),
    Output("eda_region_trend","figure"),
    Output("eda_top10","figure"),
    Input("year_dd","value"),
    Input("region_dd","value"),
    Input("country_dd","value"),
)
def refresh_eda(year_val, region_val, country_val):
    df_eda = df_clean.copy()
    if year_val:
        df_eda = df_eda[df_eda["Year"] == year_val]
    if region_val and region_val != "All":
        df_eda = df_eda[df_eda["Region"] == region_val]
    if country_val and country_val != "All":
        df_eda = df_eda[df_eda["Country"] == country_val]

    if not df_eda.empty:
        fig_box_region = px.box(
            df_eda, x="Region", y="Happiness_Score", points="all",
            title="Distribución del Happiness Score por Región"
        ).update_layout(xaxis_title="", yaxis_title="Happiness_Score")
    else:
        fig_box_region = go.Figure().update_layout(title="Distribución por Región — sin datos para el filtro actual")

    if not df_eda.empty:
        fig_bubble = px.scatter(
            df_eda, x="GDP_per_Capita", y="Happiness_Score",
            size="Social_Support", color="Region",
            hover_data=["Country","Year"],
            title="GDP vs Happiness (tamaño = Social_Support)"
        )
    else:
        fig_bubble = go.Figure().update_layout(title="GDP vs Happiness — sin datos para el filtro actual")

    trend = (df_clean.groupby(["Year","Region"], as_index=False)["Happiness_Score"].mean())
    if region_val and region_val != "All":
        trend = trend[trend["Region"] == region_val]
    fig_region_trend = px.line(
        trend, x="Year", y="Happiness_Score", color="Region",
        markers=True, title="Tendencia del Happiness (promedio) por Región"
    )

    top10 = (df_clean.groupby("Country", as_index=False)["Happiness_Score"]
             .mean()
             .sort_values("Happiness_Score", ascending=False)
             .head(10))
    fig_top10 = px.bar(
        top10.sort_values("Happiness_Score"),
        x="Happiness_Score", y="Country", orientation="h",
        title="Top 10 países más felices (promedio 2015–2019)",
        labels={"Happiness_Score":"Promedio","Country":"País"}
    )

    return fig_box_region, fig_bubble, fig_region_trend, fig_top10

# 3) Model — diagnósticos/explicabilidad (igual)
@app.callback(
    Output("coef_bar","figure"),
    Output("feature_importance_note","figure"),
    Output("perm_importance","figure"),
    Output("pdp_gdp","figure"),
    Output("coef_ci","figure"),
    Output("vif_bar","figure"),
    Output("res_vs_feat","figure"),
    Input("tabs","active_tab"),
)
def refresh_model(_tab):
    coefs = linreg_coefficients()
    if coefs:
        coef_ser = pd.Series(coefs).sort_values()
        fig_coef = px.bar(coef_ser, orientation="h",
                          title="Coeficientes del Modelo (Linear Regression)",
                          labels={"value":"Coeficiente","index":"Feature"})
    else:
        fig_coef = go.Figure().update_layout(title="Coeficientes no disponibles")

    fig_note = go.Figure()
    fig_note.add_annotation(
        text=("Nota: Los coeficientes muestran la influencia lineal de cada feature\n"
              "sobre el Happiness_Score (mayor coeficiente ⇒ mayor contribución positiva,\n"
              "manteniendo constantes las otras variables)."),
        xref="paper", yref="paper", x=0, y=0.5, showarrow=False
    )
    fig_note.update_layout(title="Interpretación de coeficientes",
                           xaxis_visible=False, yaxis_visible=False)

    Xtr, Xte, ytr, yte = train_test_split(X_ALL, y_ALL, test_size=0.3, random_state=42)

    try:
        pi = permutation_importance(model, Xte, yte, n_repeats=20,
                                    random_state=0, scoring="neg_mean_squared_error")
        imp = pd.Series(pi.importances_mean, index=X_ALL.columns).sort_values()
        fig_pi = px.bar(imp, orientation="h", title="Importancia por permutación (↓RMSE)")
    except Exception as e:
        fig_pi = go.Figure().update_layout(title=f"Importancia por permutación — no disponible ({e})")

    def pdp_curve(model_, X, feature, grid=40):
        xs = np.linspace(X[feature].min(), X[feature].max(), grid)
        Xc = X.copy()
        preds = []
        for v in xs:
            Xc[feature] = v
            preds.append(model_.predict(Xc).mean())
        return pd.DataFrame({feature: xs, "pred": preds})
    try:
        pdp1 = pdp_curve(model, Xtr, "GDP_per_Capita")
        fig_pdp = px.line(pdp1, x="GDP_per_Capita", y="pred",
                          title="PDP: GDP_per_Capita → Predicción")
    except Exception as e:
        fig_pdp = go.Figure().update_layout(title=f"PDP — no disponible ({e})")

    try:
        B = 200
        coef_samples=[]
        for _ in range(B):
            Xb, yb = resample(Xtr, ytr, replace=True)
            mb = model.__class__().fit(Xb, yb)
            coef_samples.append(mb.coef_)
        coef_df = pd.DataFrame(coef_samples, columns=X_ALL.columns)
        ci = coef_df.quantile([0.025,0.975]).T.reset_index().rename(
            columns={"index":"Feature",0.025:"low",0.975:"high"}
        )
        ci["coef"] = pd.Series(model.coef_, index=X_ALL.columns).values
        fig_ci = px.scatter(ci, x="coef", y="Feature",
                            error_x=ci["coef"]-ci["low"],
                            error_x_minus=ci["high"]-ci["coef"],
                            title="Coeficientes con IC 95% (bootstrap)")
    except Exception as e:
        fig_ci = go.Figure().update_layout(title=f"IC de coeficientes — no disponible ({e})")

    if HAS_STATSMODELS:
        try:
            Xc = sm.add_constant(X_ALL)
            vif_vals = [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
            vif = pd.DataFrame({"Feature": Xc.columns, "VIF": vif_vals})
            vif = vif[vif["Feature"] != "const"].sort_values("VIF", ascending=False)
            fig_vif = px.bar(vif, x="VIF", y="Feature", orientation="h", title="VIF por feature")
        except Exception as e:
            fig_vif = go.Figure().update_layout(title=f"VIF — error ({e})")
    else:
        fig_vif = go.Figure()
        fig_vif.add_annotation(text="Instala statsmodels para ver VIF: pip install statsmodels",
                               xref="paper", yref="paper", x=0, y=0.5, showarrow=False)
        fig_vif.update_layout(title="VIF — no disponible")

    try:
        pred_all = model.predict(X_ALL)
        res_all = y_ALL - pred_all
        fig_res_feat = px.scatter(x=X_ALL["Healthy_Life_Expectancy"], y=res_all,
                                  labels={"x":"Healthy_Life_Expectancy","y":"Residuo"},
                                  title="Residuos vs Healthy_Life_Expectancy")
        fig_res_feat.add_hline(y=0, line_dash="dash")
    except Exception as e:
        fig_res_feat = go.Figure().update_layout(title=f"Residuos vs feature — no disponible ({e})")

    return (fig_coef, fig_note, fig_pi, fig_pdp, fig_ci, fig_vif, fig_res_feat)

# 4) Streaming — tabla en vivo (usa dataset_dd)
@app.callback(
    Output("pred_table","data"),
    Input("stream_interval","n_intervals"),
    Input("dataset_dd","value"),
)
def refresh_stream_table(_n, dataset_val):
    dfp = load_predictions(dataset_val)
    if dfp.empty:
        return []
    cols = ["ts","Country","Year","Predicted_Score","Actual_Score","Data_Set"] if "Data_Set" in dfp.columns else \
           ["ts","Country","Year","Predicted_Score","Actual_Score"]
    dfp_tbl = dfp[cols].sort_values("ts", ascending=False)
    return dfp_tbl.head(200).to_dict("records")

# =========================
# Main
# =========================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
