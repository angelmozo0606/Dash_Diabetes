
import base64
import json
from pathlib import Path

import dash
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html
from scipy.stats import chi2_contingency, mannwhitneyu

from train_models import ART, DATA, FEATURES, RENAME, TARGET, train


APP_TITLE = "GlucoRisk"
TEAM = [  
    dict(name="Angel Mozo", github="https://github.com/angelmozo0606", linkedin="https://www.linkedin.com/in/angel-mozo-24053b441/?isSelfProfile=true"),
    dict(name="Abran Basto", github="https://github.com/abrahanb123", linkedin="https://www.linkedin.com/in/abraham-basto-28048631a/"),
]
VIDEO_URL = ""  # ← ej. "https://drive.google.com/file/d/<ID>/preview"
POP_PREVALENCE = 0.139  


INK, PANEL, PANEL2, LINE = "#0c1730", "#13213f", "#182a4f", "#24365e"
FG, MUTED = "#eef2fb", "#8fa0c2"
ROSE, TEAL, AMBER = "#ff5c8a", "#3ddbc4", "#ffb547"
SANS = "Manrope, system-ui, sans-serif"
SERIF = "Fraunces, Georgia, serif"
GROUP_COLORS = {0: TEAL, 1: ROSE}
GROUP_NAMES = {0: "Sin diabetes", 1: "Con diabetes / prediabetes"}


YES_NO = {0: "No", 1: "Sí"}
AGE_LABELS = {1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49", 7: "50–54",
              8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"}
EDU_LABELS = {1: "Sin escolaridad", 2: "Primaria", 3: "Secundaria incompleta",
              4: "Secundaria completa", 5: "Universidad incompleta", 6: "Universitario graduado"}
INC_LABELS = {1: "< $10k", 2: "$10–15k", 3: "$15–20k", 4: "$20–25k", 5: "$25–35k",
              6: "$35–50k", 7: "$50–75k", 8: "≥ $75k"}
GEN_LABELS = {1: "Excelente", 2: "Muy buena", 3: "Buena", 4: "Regular", 5: "Mala"}

VARS = {
    "PA_alta": dict(label="Presión arterial alta", type="bin", cats=YES_NO,
                    desc="Diagnóstico de hipertensión por un profesional de la salud."),
    "col_alto": dict(label="Colesterol alto", type="bin", cats=YES_NO,
                     desc="Diagnóstico de colesterol alto."),
    "CheckCol": dict(label="Chequeo de colesterol", type="bin", cats=YES_NO,
                     desc="Se midió el colesterol en los últimos 5 años."),
    "IMC": dict(label="Índice de masa corporal", type="num", cats=None,
                desc="Peso (kg) / talla² (m²)."),
    "Fumador": dict(label="Fumador", type="bin", cats=YES_NO,
                    desc="Ha fumado al menos 100 cigarrillos en su vida."),
    "derrame": dict(label="Derrame cerebral", type="bin", cats=YES_NO,
                    desc="Ha tenido un accidente cerebrovascular."),
    "EnfCardi": dict(label="Enfermedad cardíaca", type="bin", cats=YES_NO,
                     desc="Enfermedad coronaria o infarto de miocardio."),
    "ActFisi": dict(label="Actividad física", type="bin", cats=YES_NO,
                    desc="Hizo actividad física fuera del trabajo en los últimos 30 días."),
    "fruta": dict(label="Fruta diaria", type="bin", cats=YES_NO,
                  desc="Consume fruta al menos una vez al día."),
    "vegetales": dict(label="Vegetales diarios", type="bin", cats=YES_NO,
                      desc="Consume vegetales al menos una vez al día."),
    "Alcohlico": dict(label="Consumo alto de alcohol", type="bin", cats=YES_NO,
                      desc="Hombres > 14 tragos/semana, mujeres > 7 tragos/semana."),
    "eps": dict(label="Cobertura de salud", type="bin", cats=YES_NO,
                desc="Tiene algún tipo de seguro o cobertura médica."),
    "AfordabilidadMedica": dict(label="No pudo pagar el médico", type="bin", cats=YES_NO,
                                desc="En el último año necesitó un médico y no fue por el costo."),
    "SaludG": dict(label="Salud general percibida", type="ord", cats=GEN_LABELS,
                   desc="Escala de 1 (excelente) a 5 (mala)."),
    "SaludM": dict(label="Días de mala salud mental", type="num", cats=None,
                   desc="Días con mala salud mental en los últimos 30 días."),
    "SaludF": dict(label="Días de mala salud física", type="num", cats=None,
                   desc="Días con mala salud física en los últimos 30 días."),
    "DificultadCaminar": dict(label="Dificultad para caminar", type="bin", cats=YES_NO,
                              desc="Dificultad seria para caminar o subir escaleras."),
    "Genero": dict(label="Sexo", type="bin", cats={0: "Mujer", 1: "Hombre"},
                   desc="0 = mujer, 1 = hombre."),
    "Edad": dict(label="Grupo de edad", type="ord", cats=AGE_LABELS,
                 desc="13 rangos de 5 años, de 18–24 a 80+."),
    "Educacion": dict(label="Nivel educativo", type="ord", cats=EDU_LABELS,
                      desc="Máximo nivel alcanzado, de 1 a 6."),
    "Ingresos": dict(label="Ingresos del hogar", type="ord", cats=INC_LABELS,
                     desc="Ingreso anual en 8 rangos (USD)."),
}
BINARY_FACTORS = ["PA_alta", "col_alto", "CheckCol", "Fumador", "derrame", "EnfCardi",
                  "ActFisi", "fruta", "vegetales", "Alcohlico", "eps",
                  "AfordabilidadMedica", "DificultadCaminar"]


def vlabel(v):
    return VARS[v]["label"] if v in VARS else v


RAW = pd.read_csv(DATA)
DF = RAW.rename(columns=RENAME)[[TARGET] + FEATURES].drop_duplicates().reset_index(drop=True)
DF = DF.astype(float)
DF[TARGET] = DF[TARGET].astype(int)
N = len(DF)


def load_artifacts():
    try:
        models = joblib.load(ART / "models.joblib")
        metrics = json.loads((ART / "metrics.json").read_text())
    except Exception:  # artefactos ausentes o versión incompatible → reentrenar
        models, metrics = train()
    for m in models.values():  # predicción de una sola fila: evitar overhead de hilos
        est = m.steps[-1][1] if hasattr(m, "steps") else m
        if hasattr(est, "n_jobs"):
            est.set_params(n_jobs=1)
    return models, metrics


MODELS, METRICS = load_artifacts()
BEST = METRICS["best"]
RESULTS = METRICS["results"]
BEST_ROW = RESULTS[0]
MEANS = DF[FEATURES].mean()


def _cramers_v(table):
    chi2 = chi2_contingency(table)[0]
    n = table.values.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


BIV = {}
for v in FEATURES:
    if VARS[v]["type"] == "num":
        g0, g1 = DF.loc[DF[TARGET] == 0, v], DF.loc[DF[TARGET] == 1, v]
        u, p = mannwhitneyu(g1, g0, alternative="two-sided")
        r_rb = 2 * u / (len(g0) * len(g1)) - 1  # correlación biserial por rangos
        BIV[v] = dict(test="Mann-Whitney U", stat=float(u), p=float(p), effect=float(r_rb),
                      effect_name="r biserial por rangos")
    else:
        tab = pd.crosstab(DF[v], DF[TARGET])
        chi2, p, dof, _ = chi2_contingency(tab)
        BIV[v] = dict(test="Chi-cuadrado", stat=float(chi2), p=float(p), dof=int(dof),
                      effect=_cramers_v(tab), effect_name="V de Cramér")

SPEARMAN = DF.corr(method="spearman")[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
PREV_AGE = DF.groupby("Edad")[TARGET].mean()



def base_layout(fig, h=340, legend=True, **kw):
    fig.update_layout(
        height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, color=FG, size=12),
        margin=dict(l=10, r=10, t=44 if legend else 10, b=10),
        showlegend=legend,
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        separators=". ",
        hoverlabel=dict(bgcolor=PANEL2, bordercolor=LINE, font=dict(family=SANS, color=FG)),
        **kw,
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED), automargin=True)
    return fig


def graph(fig, id_=None):
    props = dict(figure=fig, config={"displayModeBar": False},
                 style={"height": f"{fig.layout.height or 340}px"})
    if id_:
        props["id"] = id_
    return dcc.Graph(**props)


def section_head(title, sub=None):
    return html.Div([html.H2(title), html.P(sub) if sub else None], className="section-head")


def panel(children, title=None, sub=None, cls=""):
    head = []
    if title:
        head.append(html.H3(title))
    if sub:
        head.append(html.P(sub, className="sub"))
    return html.Div(head + (children if isinstance(children, list) else [children]),
                    className=f"panel {cls}".strip())


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_int(n):
    return f"{n:,}".replace(",", " ")


def stat_box(value, label):
    return html.Div([html.B(value), html.Span(label)], className="stat")


def cat_text(v, x):
    cats = VARS[v]["cats"]
    return cats.get(int(x), str(x)) if cats else str(x)



def fig_univariate(v):
    meta = VARS[v]
    if meta["type"] == "num":
        fig = go.Figure(go.Histogram(x=DF[v], nbinsx=50 if v == "IMC" else 31,
                                     marker_color=AMBER, marker_line_width=0,
                                     hovertemplate="%{x}: %{y:,} personas<extra></extra>"))
        fig.update_xaxes(title=meta["label"])
        fig.update_yaxes(title="Personas")
        return base_layout(fig, h=360, legend=False, bargap=0.04)
    counts = DF[v].value_counts().sort_index()
    labels = [cat_text(v, k) for k in counts.index]
    pct = counts / counts.sum() * 100
    colors = [TEAL, ROSE] if meta["type"] == "bin" else [AMBER] * len(counts)
    fig = go.Figure(go.Bar(x=labels, y=counts.values, marker_color=colors,
                           text=[f"{p:.1f}%" for p in pct], textposition="outside",
                           textfont=dict(color=FG),
                           hovertemplate="%{x}: %{y:,} personas<extra></extra>"))
    fig.update_yaxes(title="Personas", range=[0, counts.max() * 1.18])
    return base_layout(fig, h=360, legend=False)


def univariate_summary(v):
    meta = VARS[v]
    s = DF[v]
    if meta["type"] == "num":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        boxes = [stat_box(f"{s.mean():.2f}", "Media"), stat_box(f"{s.median():.1f}", "Mediana"),
                 stat_box(f"{q3 - q1:.1f}", f"IQR (Q1 {q1:.0f}, Q3 {q3:.0f})"),
                 stat_box(f"{s.min():.0f} – {s.max():.0f}", "Mínimo – máximo"),
                 stat_box(f"{s.skew():.2f}", "Asimetría"), stat_box(f"{s.std():.2f}", "Desv. estándar")]
        if v == "IMC":
            ob = (s >= 30).mean() * 100
            txt = f"El {ob:.1f}% de la muestra tiene IMC ≥ 30 (obesidad). La cola derecha llega hasta {s.max():.0f}."
        else:
            zero = (s == 0).mean() * 100
            txt = f"El {zero:.1f}% reporta 0 días; la distribución se concentra en cero con picos en 15 y 30."
    else:
        counts = s.value_counts().sort_index()
        mode = counts.idxmax()
        boxes = [stat_box(f"{counts[k] / len(s) * 100:.1f}%", cat_text(v, k)) for k in counts.index]
        txt = f"La categoría más frecuente es «{cat_text(v, mode)}» con {fmt_int(int(counts[mode]))} personas."
    return html.Div([html.Div(boxes, className="stat-list"), html.Div(txt, className="verdict")])


def fig_binary_overview():
    rows = [(vlabel(v), DF[v].mean() * 100) for v in BINARY_FACTORS]
    rows.sort(key=lambda t: t[1])
    fig = go.Figure(go.Bar(y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h",
                           marker_color=TEAL, text=[f"{r[1]:.1f}%" for r in rows],
                           textposition="outside", textfont=dict(color=FG),
                           hovertemplate="%{y}: %{x:.1f}% responde Sí<extra></extra>"))
    fig.update_xaxes(title="% que responde Sí", range=[0, 112])
    return base_layout(fig, h=430, legend=False)


def fig_bivariate(v):
    meta = VARS[v]
    if meta["type"] == "num":
        fig = go.Figure()
        for g in (0, 1):
            fig.add_trace(go.Box(y=DF.loc[DF[TARGET] == g, v], name=GROUP_NAMES[g],
                                 marker_color=GROUP_COLORS[g], boxmean=True,
                                 marker=dict(size=2, opacity=0.25), line=dict(width=1.6)))
        fig.update_yaxes(title=meta["label"])
        return base_layout(fig, h=380, legend=False)
    tab = pd.crosstab(DF[v], DF[TARGET], normalize="index") * 100
    labels = [cat_text(v, k) for k in tab.index]
    counts = DF[v].value_counts().sort_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=tab[1], name="% con diabetes", marker_color=ROSE,
                         text=[f"{p:.0f}%" for p in tab[1]], textposition="outside",
                         textfont=dict(color=FG), customdata=counts.values,
                         hovertemplate="%{x}<br>%{y:.1f}% con diabetes<br>n = %{customdata:,}<extra></extra>"))
    fig.add_hline(y=DF[TARGET].mean() * 100, line_dash="dot", line_color=MUTED,
                  annotation_text="promedio de la muestra", annotation_font_color=MUTED,
                  annotation_position="top left")
    fig.update_yaxes(title="% con diabetes o prediabetes", range=[0, 105])
    return base_layout(fig, h=380, legend=False)


def bivariate_summary(v):
    b = BIV[v]
    meta = VARS[v]
    if meta["type"] == "num":
        g = DF.groupby(TARGET)[v]
        m0, m1 = g.mean()[0], g.mean()[1]
        md0, md1 = g.median()[0], g.median()[1]
        boxes = [stat_box(f"{m0:.2f}", "Media sin diabetes"), stat_box(f"{m1:.2f}", "Media con diabetes"),
                 stat_box(f"{md0:.1f} / {md1:.1f}", "Medianas"),
                 stat_box(f"{b['effect']:.3f}", b["effect_name"])]
        verdict = (f"{b['test']}: p {fmt_p(b['p'])}. Las distribuciones difieren entre grupos; "
                   f"el grupo con diabetes tiene valores más altos de {meta['label'].lower()}.")
    else:
        boxes = [stat_box(f"{b['stat']:,.1f}".replace(",", " "), "χ²"),
                 stat_box(str(b["dof"]), "Grados de libertad"),
                 stat_box(fmt_p(b["p"]), "p-valor"),
                 stat_box(f"{b['effect']:.3f}", b["effect_name"])]
        strength = ("débil" if b["effect"] < 0.1 else "moderada" if b["effect"] < 0.3 else "fuerte")
        verdict = (f"Se rechaza la independencia (p {fmt_p(b['p'])}). Con una muestra tan grande todo resulta "
                   f"significativo; la V de Cramér indica una asociación {strength}.")
    return html.Div([html.Div(boxes, className="stat-list"), html.Div(verdict, className="verdict")])


def fig_dumbbell():
    rows = []
    for v in BINARY_FACTORS:
        p = DF.groupby(v)[TARGET].mean() * 100
        rows.append((vlabel(v), p.get(0.0, np.nan), p.get(1.0, np.nan)))
    rows.sort(key=lambda r: r[2] - r[1])
    fig = go.Figure()
    for name, p0, p1 in rows:
        fig.add_trace(go.Scatter(x=[p0, p1], y=[name, name], mode="lines",
                                 line=dict(color=LINE, width=4), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=[r[1] for r in rows], y=[r[0] for r in rows], mode="markers",
                             name="Factor = No", marker=dict(color=TEAL, size=11),
                             hovertemplate="%{y}<br>Sin el factor: %{x:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r[2] for r in rows], y=[r[0] for r in rows], mode="markers",
                             name="Factor = Sí", marker=dict(color=ROSE, size=11),
                             hovertemplate="%{y}<br>Con el factor: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(title="% con diabetes o prediabetes", range=[0, 100])
    return base_layout(fig, h=470)


def fig_spearman():
    s = SPEARMAN.sort_values(key=np.abs)
    fig = go.Figure(go.Bar(y=[vlabel(v) for v in s.index], x=s.values, orientation="h",
                           marker_color=[ROSE if x > 0 else TEAL for x in s.values],
                           hovertemplate="%{y}: ρ = %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Correlación de Spearman con Diabetes")
    return base_layout(fig, h=560, legend=False)


def fig_heatmap():
    top = [TARGET] + list(SPEARMAN.index[:9])
    c = DF[top].corr(method="spearman")
    names = ["Diabetes"] + [vlabel(v) for v in top[1:]]
    fig = go.Figure(go.Heatmap(z=c.values, x=names, y=names, zmin=-1, zmax=1,
                               colorscale=[[0, TEAL], [0.5, PANEL], [1, ROSE]],
                               text=np.round(c.values, 2), texttemplate="%{text}",
                               textfont=dict(size=10), colorbar=dict(tickfont=dict(color=MUTED))))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    return base_layout(fig, h=520, legend=False)


def fig_funnel():
    m = METRICS
    fig = go.Figure(go.Funnel(
        y=["Registros originales", "Tras eliminar duplicados", "Entrenamiento (80%)", "Prueba (20%)"],
        x=[m["n_raw"], m["n_clean"], m["n_train"], m["n_test"]],
        marker=dict(color=[MUTED, TEAL, ROSE, AMBER]),
        texttemplate="%{value:,}", textfont=dict(color=INK, family=SANS, size=13),
        connector=dict(fillcolor=PANEL2, line=dict(color=LINE))))
    return base_layout(fig, h=320, legend=False)



MODEL_COLORS = [ROSE, TEAL, AMBER, "#8b9dff", "#c792ea", MUTED]


def fig_roc():
    fig = go.Figure()
    for i, r in enumerate(RESULTS):
        roc = METRICS["roc"][r["name"]]
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                 name=f"{r['name']} ({r['auc']:.3f})",
                                 line=dict(color=MODEL_COLORS[i], width=3 if i == 0 else 1.6)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=LINE, dash="dot"),
                             showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(title="Tasa de falsos positivos", range=[0, 1])
    fig.update_yaxes(title="Tasa de verdaderos positivos (recall)", range=[0, 1.02])
    base_layout(fig, h=420)
    fig.update_layout(margin=dict(t=10), legend=dict(orientation="v", x=0.98, xanchor="right", y=0.04,
                                                     yanchor="bottom", bgcolor="rgba(19,33,63,.85)"))
    return fig


def fig_metric_bars():
    keys = [("acc", "Accuracy"), ("prec", "Precisión"), ("rec", "Recall"), ("f1", "F1")]
    fig = go.Figure()
    colors = [MUTED, AMBER, ROSE, TEAL]
    for (k, name), col in zip(keys, colors):
        fig.add_trace(go.Bar(name=name, x=[r["name"] for r in RESULTS], y=[r[k] for r in RESULTS],
                             marker_color=col, hovertemplate=name + ": %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0.6, 0.85], title="Valor (umbral 0.5)")
    return base_layout(fig, h=380, barmode="group", bargap=0.25)


def fig_overfit():
    names = [r["name"] for r in RESULTS]
    fig = go.Figure()
    for k, name, col in [("tr", "AUC entrenamiento", MUTED), ("cv", "AUC validación cruzada (5 folds)", AMBER),
                         ("auc", "AUC prueba", ROSE)]:
        fig.add_trace(go.Scatter(x=names, y=[r[k] for r in RESULTS], mode="lines+markers", name=name,
                                 line=dict(color=col, width=2), marker=dict(size=9)))
    fig.update_yaxes(title="ROC-AUC", range=[0.76, 0.88])
    return base_layout(fig, h=360)


def fig_confusion(name):
    cm = np.array(METRICS["cm"][name])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = ["Sin diabetes", "Con diabetes"]
    text = [[f"{cm[i, j]:,}<br>{pct[i, j]:.1f}%".replace(",", " ") for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=pct, x=labels, y=labels, text=text, texttemplate="%{text}",
                               textfont=dict(size=15, color=FG), showscale=False,
                               colorscale=[[0, PANEL], [1, ROSE]], zmin=0, zmax=100,
                               hovertemplate="Real: %{y}<br>Predicho: %{x}<extra></extra>"))
    fig.update_xaxes(title="Predicción", side="bottom")
    fig.update_yaxes(title="Valor real", autorange="reversed")
    return base_layout(fig, h=340, legend=False)


def fig_importance():
    imp = METRICS["importance"][:12][::-1]
    fig = go.Figure(go.Bar(y=[vlabel(d["f"]) for d in imp], x=[d["v"] for d in imp], orientation="h",
                           marker_color=ROSE, hovertemplate="%{y}: −%{x:.4f} AUC al permutar<extra></extra>"))
    fig.update_xaxes(title="Caída del AUC al permutar la variable")
    return base_layout(fig, h=420, legend=False)


def fig_odds():
    coefs = METRICS["lr_coef"][:12][::-1]
    odds = [np.exp(d["v"]) for d in coefs]
    fig = go.Figure(go.Bar(y=[vlabel(d["f"]) for d in coefs], x=[o - 1 for o in odds], base=1,
                           orientation="h", marker_color=[ROSE if o > 1 else TEAL for o in odds],
                           customdata=odds,
                           hovertemplate="%{y}: OR = %{customdata:.2f} por 1 desv. estándar<extra></extra>"))
    fig.add_vline(x=1, line_color=MUTED, line_dash="dot")
    fig.update_xaxes(title="Odds ratio por aumento de 1 desviación estándar")
    return base_layout(fig, h=420, legend=False)



DEFAULT_PROFILE = dict(IMC=28, Edad=8, SaludG=3, SaludM=0, SaludF=0, Educacion=5, Ingresos=6,
                       Genero=0, chips=["CheckCol", "ActFisi", "fruta", "vegetales", "eps"])
PRESETS = {
    "low": dict(IMC=23, Edad=4, SaludG=1, SaludM=0, SaludF=0, Educacion=6, Ingresos=8, Genero=0,
                chips=["CheckCol", "ActFisi", "fruta", "vegetales", "eps"]),
    "high": dict(IMC=36, Edad=10, SaludG=4, SaludM=5, SaludF=15, Educacion=4, Ingresos=3, Genero=1,
                 chips=["PA_alta", "col_alto", "CheckCol", "Fumador", "EnfCardi", "DificultadCaminar", "eps"]),
}
SLIDERS = ["IMC", "Edad", "SaludG", "SaludM", "SaludF", "Educacion", "Ingresos"]


def build_row(values):
    row = {f: 0.0 for f in FEATURES}
    for k in SLIDERS:
        row[k] = float(values[k])
    row["Genero"] = float(values["Genero"])
    for c in values["chips"] or []:
        row[c] = 1.0
    return pd.DataFrame([row])[FEATURES]


def adjust_prior(p, prior=POP_PREVALENCE):
    """Lleva una probabilidad de un modelo entrenado al 50/50 a la prevalencia poblacional."""
    odds = p / (1 - p) * prior / (1 - prior)
    return odds / (1 + odds)


def fig_gauge(p):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number=dict(suffix="%", font=dict(family=SERIF, size=46, color=FG)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED)),
                   bar=dict(color=FG, thickness=0.22), bgcolor=PANEL2, borderwidth=0,
                   steps=[dict(range=[0, 35], color="rgba(61,219,196,.35)"),
                          dict(range=[35, 65], color="rgba(255,181,71,.35)"),
                          dict(range=[65, 100], color="rgba(255,92,138,.4)")])))
    base_layout(fig, h=250, legend=False)
    fig.update_layout(margin=dict(l=36, r=36, t=24, b=0))
    return fig


def fig_contrib(row):
    lr = MODELS["Regresión Logística"]
    sc, clf = lr.named_steps["scaler"], lr.named_steps["clf"]
    z = (row.values[0] - sc.mean_) / sc.scale_
    contrib = pd.Series(z * clf.coef_[0], index=FEATURES)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10])[::-1]
    fig = go.Figure(go.Bar(y=[vlabel(v) for v in top.index], x=top.values, orientation="h",
                           marker_color=[ROSE if x > 0 else TEAL for x in top.values],
                           hovertemplate="%{y}: %{x:+.2f} en log-odds<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED)
    fig.update_xaxes(title="← baja el riesgo      sube el riesgo →")
    return base_layout(fig, h=360, legend=False)


def fig_all_models(row):
    names = [r["name"] for r in RESULTS]
    probs = [float(MODELS[n].predict_proba(row)[0, 1]) * 100 for n in names]
    fig = go.Figure(go.Bar(x=probs, y=names, orientation="h",
                           marker_color=[ROSE if p >= 50 else TEAL for p in probs],
                           text=[f"{p:.0f}%" for p in probs], textposition="outside", textfont=dict(color=FG),
                           cliponaxis=False, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.add_vline(x=50, line_color=MUTED, line_dash="dot")
    fig.update_xaxes(range=[0, 118], title="Probabilidad estimada (%)", tickvals=[0, 25, 50, 75, 100])
    fig.update_yaxes(autorange="reversed")
    return base_layout(fig, h=280, legend=False)


NAV_ITEMS = [("Inicio", "/"), ("Contexto", "/contexto"), ("Exploratorio", "/exploratorio"),
             ("Modelos", "/modelos"), ("Predicción", "/prediccion"), ("Aplicaciones", "/aplicaciones")]

NAVBAR = html.Header(html.Div([
    dcc.Link([html.Div(className="brand-mark"), html.Span(APP_TITLE, className="brand-name")],
             href="/", className="brand"),
    html.Nav([dcc.Link(t, href=h, id=f"nav-{i}", className="nav-link") for i, (t, h) in enumerate(NAV_ITEMS)],
             className="nav"),
], className="topbar-inner"), className="topbar")

FOOTER = html.Footer([
    html.Span(f"{APP_TITLE} · Proyecto académico de analítica predictiva"),
    html.Span("Datos: CDC BRFSS 2015 (Kaggle, Alex Teboul)"),
], className="footer")


def hero_curve_svg():
    """Curva real: % con diabetes por grupo de edad (se dibuja una vez al cargar)."""
    w, h, pad = 560, 230, 24
    xs = np.linspace(pad, w - pad, len(PREV_AGE))
    ys = h - pad - (PREV_AGE.values) * (h - 2 * pad)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"{xs[0]:.1f},{h - pad} " + pts + f" {xs[-1]:.1f},{h - pad}"
    grid = "".join(
        f'<line x1="{pad}" x2="{w - pad}" y1="{h - pad - g * (h - 2 * pad):.1f}" y2="{h - pad - g * (h - 2 * pad):.1f}" '
        f'stroke="{LINE}" stroke-width="1"/><text x="{w - pad}" y="{h - pad - g * (h - 2 * pad) - 5:.1f}" '
        f'fill="{MUTED}" font-size="11" text-anchor="end" font-family="Manrope,sans-serif">{int(g * 100)}%</text>'
        for g in (0.25, 0.5, 0.75))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">
<style>
.p{{stroke-dasharray:1400;stroke-dashoffset:1400;animation:d 2.6s ease-out .3s forwards}}
.a{{opacity:0;animation:o 1s ease-out 1.8s forwards}}
.t{{opacity:0;animation:o .4s ease-out 2.7s forwards}}
@keyframes d{{to{{stroke-dashoffset:0}}}} @keyframes o{{to{{opacity:1}}}}
@media (prefers-reduced-motion:reduce){{.p{{animation:none;stroke-dashoffset:0}}.a,.t{{animation:none;opacity:1}}}}
</style>
{grid}
<polygon class="a" points="{area}" fill="{ROSE}" fill-opacity="0.10"/>
<polyline class="p" points="{pts}" fill="none" stroke="{ROSE}" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>
<circle class="t" cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="6" fill="{ROSE}" stroke="{INK}" stroke-width="3"/>
<circle class="t" cx="{xs[0]:.1f}" cy="{ys[0]:.1f}" r="5" fill="{TEAL}" stroke="{INK}" stroke-width="3"/>
</svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def page_home():
    p_young, p_old = PREV_AGE.iloc[0] * 100, PREV_AGE.max() * 100
    hero = html.Div([
        html.Div([
            html.H1("¿Quién está en riesgo de diabetes?"),
            html.P(f"Indicadores de salud de {fmt_int(N)} adultos de la encuesta BRFSS 2015 y seis modelos de "
                   "clasificación para estimar el riesgo de diabetes o prediabetes a partir de hábitos, "
                   "antecedentes y condiciones socioeconómicas.", className="hero-lead"),
            html.Div([
                dcc.Link("Calcular un riesgo", href="/prediccion", className="btn btn-primary"),
                dcc.Link("Explorar los datos", href="/exploratorio", className="btn btn-ghost"),
            ], className="cta-row"),
        ]),
        html.Div([
            html.Img(src=hero_curve_svg(), alt="Curva de prevalencia de diabetes por grupo de edad",
                     style={"width": "100%"}),
            html.Div([html.Span(f"18–24 años: {p_young:.0f}%"), html.Span(f"Máximo: {p_old:.0f}%")],
                     className="pulse-legend"),
            html.P("Proporción con diabetes o prediabetes según grupo de edad, en la muestra balanceada.",
                   className="note"),
        ], className="pulse-card"),
    ], className="hero")

    facts = html.Div([
        html.Div([html.B(fmt_int(N)), html.Span("registros únicos")], className="fact"),
        html.Div([html.B("21"), html.Span("variables predictoras")], className="fact"),
        html.Div([html.B("6"), html.Span("modelos comparados")], className="fact"),
        html.Div([html.B(f"{BEST_ROW['auc']:.3f}"), html.Span(f"ROC-AUC de {BEST}")], className="fact"),
    ], className="facts")

    routes = html.Div([
        dcc.Link([html.H4("Contexto"), html.P("El problema, la fuente de los datos y el diccionario de variables.")],
                 href="/contexto", className="route"),
        dcc.Link([html.H4("Exploratorio"), html.P("Distribuciones, pruebas de asociación y el proceso ETL.")],
                 href="/exploratorio", className="route"),
        dcc.Link([html.H4("Modelos"), html.P("Curvas ROC, matrices de confusión e importancia de variables.")],
                 href="/modelos", className="route"),
        dcc.Link([html.H4("Aplicaciones"), html.P("Dónde puede servir un modelo así y cuáles son sus límites.")],
                 href="/aplicaciones", className="route"),
    ], className="grid-4", style={"marginTop": "34px"})

    people = [html.Div([html.B(p["name"]), html.Div([
        html.A("GitHub", href=p["github"], target="_blank", className="btn btn-ghost btn-small"),
        html.A("LinkedIn", href=p["linkedin"], target="_blank", className="btn btn-ghost btn-small"),
    ], className="person-links")], className="person") for p in TEAM]
    video = (html.Iframe(src=VIDEO_URL, className="video-box", allow="autoplay") if VIDEO_URL else
             html.Div("Agrega el enlace del video explicativo en la variable VIDEO_URL de app.py.",
                      className="video-empty"))
    team = html.Div([
        section_head("Equipo"),
        html.Div([panel(people, title="Integrantes"), panel(video, title="Video explicativo")],
                 className="grid-side"),
    ], className="section")
    return html.Div([hero, facts, routes, team])


def page_context():
    dict_rows = [html.Tr([html.Td(html.Code(v)), html.Td(VARS[v]["label"]),
                          html.Td(html.Span({"bin": "Binaria", "ord": "Ordinal", "num": "Numérica"}[VARS[v]["type"]],
                                            className=f"tag tag-{VARS[v]['type']}")),
                          html.Td(VARS[v]["desc"], style={"color": MUTED})]) for v in FEATURES]
    return html.Div([
        html.Div([
            section_head("El problema",
                         "La diabetes tipo 2 avanza sin síntomas durante años. Detectar a tiempo a quienes tienen "
                         "más riesgo permite intervenir con cambios de hábitos antes de que aparezcan complicaciones."),
            html.Div([
                panel(html.P("Clasificar si una persona tiene diabetes o prediabetes (1) o no (0) usando solo "
                             "información de una encuesta: sin exámenes de laboratorio."),
                      title="Objetivo"),
                panel(html.P("Behavioral Risk Factor Surveillance System 2015 del CDC. Versión balanceada 50/50 "
                             f"publicada en Kaggle: {fmt_int(METRICS['n_raw'])} respuestas, "
                             f"{fmt_int(N)} tras quitar duplicados."), title="Fuente"),
                panel(html.P("Como las clases están balanceadas, la exactitud es interpretable, pero priorizamos "
                             "ROC-AUC y recall: en tamizaje es más costoso no detectar a un caso que revisar a "
                             "un falso positivo."), title="Cómo medimos"),
            ], className="grid-3"),
        ], className="section"),
        html.Div([
            section_head("Diccionario de variables",
                         "Nombres usados en los notebooks de análisis. La variable respuesta es Diabetes."),
            panel(html.Div(html.Table([
                html.Thead(html.Tr([html.Th("Variable"), html.Th("Nombre"), html.Th("Tipo"), html.Th("Descripción")])),
                html.Tbody(dict_rows)], className="clean"), className="table-wrap")),
        ], className="section"),
        html.Div([
            section_head("Métricas de evaluación"),
            html.Div([
                panel(html.P("Área bajo la curva ROC. Mide qué tan bien el modelo ordena a las personas por "
                             "riesgo, sin depender de un umbral. 0.5 es azar, 1 es perfecto."), title="ROC-AUC"),
                panel(html.P("De todas las personas con diabetes, qué porcentaje detecta el modelo. "
                             "Es la métrica clave para tamizaje."), title="Recall (sensibilidad)"),
                panel(html.P("De las personas que el modelo marca como en riesgo, qué porcentaje realmente "
                             "tiene diabetes. Controla la carga de falsas alarmas."), title="Precisión"),
            ], className="grid-3"),
        ], className="section"),
    ])


def page_eda():
    var_opts = [{"label": vlabel(v), "value": v} for v in FEATURES]
    tabs = html.Div([
        html.Button("Univariado", id="tab-uni", className="tab-btn on", n_clicks=0),
        html.Button("Bivariado", id="tab-bi", className="tab-btn", n_clicks=0),
        html.Button("ETL", id="tab-etl", className="tab-btn", n_clicks=0),
    ], className="tabs-bar")

    uni = html.Div([
        html.Div([
            panel([dcc.Dropdown(var_opts, "IMC", id="uni-var", clearable=False, style={"marginBottom": "12px"}),
                   dcc.Graph(id="uni-graph", config={"displayModeBar": False})],
                  title="Distribución por variable", sub="Elige una variable para ver su distribución."),
            panel(html.Div(id="uni-summary"), title="Resumen", sub="Medidas descriptivas de la variable."),
        ], className="grid-side"),
        html.Div(panel(graph(fig_binary_overview()), title="Factores binarios en la muestra",
                       sub="Porcentaje que responde Sí en cada variable binaria."),
                 style={"marginTop": "18px"}),
    ], id="pane-uni")

    bi = html.Div([
        html.Div([
            panel([dcc.Dropdown(var_opts, "SaludG", id="bi-var", clearable=False, style={"marginBottom": "12px"}),
                   dcc.Graph(id="bi-graph", config={"displayModeBar": False})],
                  title="Variable frente a Diabetes",
                  sub="Categóricas: % con diabetes en cada categoría. Numéricas: cajas por grupo."),
            panel(html.Div(id="bi-summary"), title="Prueba estadística",
                  sub="Chi-cuadrado para categóricas y Mann-Whitney para numéricas (no normales, Anderson-Darling)."),
        ], className="grid-side"),
        html.Div([
            panel(graph(fig_dumbbell()), title="Prevalencia con y sin cada factor",
                  sub="Distancia entre puntos = cuánto cambia la proporción con diabetes al tener el factor."),
            panel(graph(fig_spearman()), title="Correlación con Diabetes",
                  sub="Spearman, comparable entre variables binarias, ordinales y numéricas."),
        ], className="grid-2", style={"marginTop": "18px"}),
        html.Div(panel(graph(fig_heatmap()), title="Correlaciones entre las variables más asociadas",
                       sub="Útil para detectar redundancia: SaludG, SaludF y DificultadCaminar se mueven juntas."),
                 style={"marginTop": "18px"}),
    ], id="pane-bi", style={"display": "none"})

    m = METRICS
    steps = [
        ("Carga", "Lectura del CSV balanceado de Kaggle.", f"{fmt_int(m['n_raw'])} filas"),
        ("Renombrado", "Columnas en español, iguales a los notebooks (HighBP → PA_alta, BMI → IMC…).", "22 columnas"),
        ("Deduplicación", "Se eliminan respuestas idénticas para no inflar el entrenamiento.",
         f"−{fmt_int(m['n_dup'])} filas"),
        ("Validación de tipos", "Todas las variables son numéricas, sin nulos; rangos verificados (IMC 12–98, días 0–30).",
         "0 nulos"),
        ("Partición estratificada", "80% entrenamiento y 20% prueba, conservando el 50/50 de clases.",
         f"{fmt_int(m['n_train'])} / {fmt_int(m['n_test'])}"),
        ("Escalado", "StandardScaler solo para modelos sensibles a escala: regresión logística y KNN.",
         "2 modelos"),
        ("Validación cruzada", "5 folds estratificados sobre entrenamiento para estimar la estabilidad del AUC.",
         "5 folds"),
    ]
    etl = html.Div([
        html.Div([
            panel(html.Ol([html.Li([html.Div([html.B(t), html.Span(d)]), html.Em(n)]) for t, d, n in steps],
                          className="steps"),
                  title="Pipeline de preparación", sub="Pasos en el orden en que se aplican."),
            panel([graph(fig_funnel()),
                   html.P("No se aplicó tratamiento de atípicos al IMC: los valores extremos son plausibles "
                          "(obesidad severa) y los modelos de árboles son robustos a ellos.", className="note")],
                  title="Registros en cada etapa"),
        ], className="grid-2"),
    ], id="pane-etl", style={"display": "none"})

    return html.Div([
        html.Div([section_head("Análisis exploratorio",
                               "Resultados de los notebooks univariado y bivariado, en versión interactiva."),
                  tabs, uni, bi, etl], className="section"),
    ])


def results_table():
    head = html.Tr([html.Th("Modelo"), html.Th("ROC-AUC", className="num"), html.Th("Accuracy", className="num"),
                    html.Th("Precisión", className="num"), html.Th("Recall", className="num"),
                    html.Th("F1", className="num"), html.Th("AUC CV", className="num"),
                    html.Th("Brecha train–test", className="num")])
    rows = [html.Tr([
        html.Td(html.B(r["name"]) if i == 0 else r["name"]),
        html.Td(f"{r['auc']:.4f}", className="num"), html.Td(f"{r['acc']:.4f}", className="num"),
        html.Td(f"{r['prec']:.4f}", className="num"), html.Td(f"{r['rec']:.4f}", className="num"),
        html.Td(f"{r['f1']:.4f}", className="num"), html.Td(f"{r['cv']:.4f} ± {r['cv_std']:.3f}", className="num"),
        html.Td(f"{r['tr'] - r['auc']:+.3f}", className="num"),
    ], className="best" if i == 0 else "") for i, r in enumerate(RESULTS)]
    return html.Div(html.Table([html.Thead(head), html.Tbody(rows)], className="clean"), className="table-wrap")


def overfit_text():
    gaps = sorted(RESULTS, key=lambda r: r["tr"] - r["auc"])
    worst, best2 = gaps[-1], gaps[:2]
    return (f"{worst['name']} tiene la mayor brecha entre entrenamiento y prueba ({worst['tr'] - worst['auc']:+.3f}); "
            f"{best2[0]['name']} y {best2[1]['name']} casi no se sobreajustan. Las curvas de CV y prueba coinciden: "
            "los resultados son estables.")


def page_models():
    lr = next(r for r in RESULTS if r["name"] == "Regresión Logística")
    return html.Div([
        html.Div([
            section_head("Comparación de modelos",
                         f"Seis clasificadores evaluados sobre {fmt_int(METRICS['n_test'])} personas que no vieron "
                         f"en el entrenamiento. {BEST} obtiene el mejor ROC-AUC ({BEST_ROW['auc']:.3f})."),
            panel(results_table(), title="Resultados en el conjunto de prueba",
                  sub="Ordenados por ROC-AUC. Accuracy, precisión, recall y F1 con umbral 0.5."),
        ], className="section"),
        html.Div([
            html.Div([
                panel(graph(fig_roc()), title="Curvas ROC",
                      sub="Las curvas casi se superponen: el techo lo pone la información de la encuesta, no el algoritmo."),
                panel(graph(fig_metric_bars()), title="Métricas con umbral 0.5",
                      sub="Casi todos priorizan recall sobre precisión, deseable en tamizaje."),
            ], className="grid-2"),
            html.Div([
                panel(graph(fig_overfit()), title="Sobreajuste",
                      sub=overfit_text()),
                panel([dcc.Dropdown([r["name"] for r in RESULTS], BEST, id="cm-model", clearable=False,
                                    style={"marginBottom": "10px"}),
                       dcc.Graph(id="cm-graph", config={"displayModeBar": False})],
                      title="Matriz de confusión", sub="Porcentajes por fila (sobre el valor real)."),
            ], className="grid-2", style={"marginTop": "18px"}),
        ], className="section", style={"paddingTop": "24px"}),
        html.Div([
            section_head("¿Qué variables pesan más?"),
            html.Div([
                panel(graph(fig_importance()), title=f"Importancia por permutación ({BEST})",
                      sub="Cuánto cae el AUC si se desordena cada variable. Salud general, IMC y edad dominan."),
                panel(graph(fig_odds()), title="Odds ratios de la regresión logística",
                      sub=f"Modelo interpretable con AUC {lr['auc']:.3f}, apenas "
                          f"{(BEST_ROW['auc'] - lr['auc']) * 100:.1f} puntos por debajo del mejor."),
            ], className="grid-2"),
        ], className="section"),
    ])


def slider(id_, v, lo, hi, step=1, marks=None):
    return dcc.Slider(lo, hi, step, value=v, id=id_, marks=marks,
                      tooltip={"placement": "bottom", "always_visible": False})


def field(label, control, hint=None):
    return html.Div([html.Div([html.Span(label), html.Small(hint) if hint else None], className="field-label"),
                     control], className="field")


def page_predict():
    d = DEFAULT_PROFILE
    form = panel([
        html.Div([
            html.Button("Perfil de bajo riesgo", id="preset-low", className="btn btn-ghost btn-small", n_clicks=0),
            html.Button("Perfil de alto riesgo", id="preset-high", className="btn btn-ghost btn-small", n_clicks=0),
        ], className="preset-row"),
        field("Índice de masa corporal", slider("in-IMC", d["IMC"], 12, 60, 1,
                                                {12: "12", 18.5: "18.5", 25: "25", 30: "30", 40: "40", 60: "60"}),
              hint="kg/m²"),
        field("Grupo de edad", slider("in-Edad", d["Edad"], 1, 13, 1,
                                      {k: AGE_LABELS[k] for k in (1, 4, 7, 10, 13)})),
        field("Salud general percibida", slider("in-SaludG", d["SaludG"], 1, 5, 1, GEN_LABELS)),
        html.Div([
            field("Días de mala salud física", slider("in-SaludF", d["SaludF"], 0, 30, 1, {0: "0", 15: "15", 30: "30"}),
                  hint="últimos 30"),
            field("Días de mala salud mental", slider("in-SaludM", d["SaludM"], 0, 30, 1, {0: "0", 15: "15", 30: "30"}),
                  hint="últimos 30"),
        ], className="grid-2", style={"gap": "16px"}),
        field("Nivel educativo", slider("in-Educacion", d["Educacion"], 1, 6, 1,
                                        {1: "Ninguno", 4: "Secundaria", 6: "Univ."})),
        field("Ingresos del hogar", slider("in-Ingresos", d["Ingresos"], 1, 8, 1,
                                           {1: "<$10k", 5: "$25–35k", 8: "≥$75k"})),
        field("Sexo", dcc.RadioItems([{"label": "Mujer", "value": 0}, {"label": "Hombre", "value": 1}],
                                     d["Genero"], id="in-Genero", className="seg", inline=True)),
        field("Marca lo que aplica", dcc.Checklist([{"label": vlabel(v), "value": v} for v in BINARY_FACTORS],
                                                   d["chips"], id="in-chips", className="chips")),
    ], title="Perfil de la persona", sub="Mueve los controles; el resultado se actualiza al instante.")

    result = html.Div([
        panel([
            dcc.Dropdown([{"label": f"{r['name']} (AUC {r['auc']:.3f})", "value": r["name"]} for r in RESULTS],
                         BEST, id="in-model", clearable=False, style={"marginBottom": "6px"}),
            dcc.Graph(id="out-gauge", config={"displayModeBar": False}),
            html.Div(id="out-band"),
            html.P("Resultado educativo basado en una encuesta. No reemplaza una prueba de glucosa ni la "
                   "valoración de un profesional de la salud.", className="disclaimer"),
        ], title="Riesgo estimado"),
        panel(dcc.Graph(id="out-contrib", config={"displayModeBar": False}), title="Qué mueve este resultado",
              sub="Aporte de cada variable según la regresión logística, para este perfil frente al promedio."),
        panel(dcc.Graph(id="out-models", config={"displayModeBar": False}), title="Los seis modelos",
              sub="Probabilidad que asigna cada modelo al mismo perfil."),
    ], style={"display": "grid", "gap": "18px"})

    return html.Div([
        html.Div([section_head("Calcula un riesgo",
                               f"Ingresa un perfil y compara lo que estiman los modelos. Por defecto se usa {BEST}."),
                  html.Div([form, result], className="grid-form")], className="section"),
    ])


def page_apps():
    rec, prec = BEST_ROW["rec"], BEST_ROW["prec"]
    uses = [
        ("Tamizaje en EPS y aseguradoras",
         "Ordenar a los afiliados por riesgo con datos que ya existen en encuestas de ingreso, y citar primero "
         "a quienes encabezan la lista para una prueba de HbA1c."),
        ("Priorización de pruebas de laboratorio",
         f"Con recall de {rec:.0%}, el modelo encuentra ocho de cada diez casos en la muestra; útil cuando "
         "las pruebas son limitadas y hay que decidir a quién hacerlas primero."),
        ("Campañas de prevención focalizadas",
         "Salud general, IMC, edad, hipertensión y colesterol explican la mayor parte del riesgo: son los ejes "
         "para diseñar mensajes y programas de actividad física y nutrición."),
        ("Herramienta educativa",
         "La página de predicción muestra cómo cambia el riesgo al modificar hábitos, lo que sirve para "
         "explicar factores modificables a pacientes o estudiantes."),
    ]
    limits = [
        ("Datos autorreportados", "El diagnóstico, el peso y la talla los informa la persona; hay error de memoria y de sesgo."),
        ("Muestra balanceada", f"En la población solo ~{POP_PREVALENCE:.0%} tiene diabetes o prediabetes. Las "
                               "probabilidades del modelo deben ajustarse a esa prevalencia antes de usarse en la práctica."),
        ("Contexto de EE.UU. 2015", "Antes de aplicarlo en Colombia habría que validarlo con datos locales, "
                                    "por ejemplo la ENSIN o registros de EPS."),
        ("Precisión moderada", f"Con precisión de {prec:.0%}, aproximadamente 1 de cada 4 alertas es un falso positivo: "
                               "sirve para priorizar, no para diagnosticar."),
    ]
    return html.Div([
        html.Div([section_head("Aplicaciones reales",
                               "Un modelo de encuesta no diagnostica, pero sí ayuda a decidir a quién evaluar primero."),
                  html.Div([panel(html.P(t), title=h) for h, t in uses], className="grid-2")], className="section"),
        html.Div([section_head("Límites a tener en cuenta"),
                  html.Div([panel(html.P(t), title=h, cls="panel-quiet") for h, t in limits], className="grid-2")],
                 className="section"),
    ])



app = dash.Dash(
    __name__, title=f"{APP_TITLE} · Predicción de diabetes", suppress_callback_exceptions=True,
    external_stylesheets=["https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600"
                          "&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap"],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    NAVBAR,
    html.Main([html.Div(id="page-content"), FOOTER], className="page"),
])

PAGES = {"/": page_home, "/contexto": page_context, "/exploratorio": page_eda,
         "/modelos": page_models, "/prediccion": page_predict, "/aplicaciones": page_apps}


@app.callback(Output("page-content", "children"),
              [Output(f"nav-{i}", "className") for i in range(len(NAV_ITEMS))],
              Input("url", "pathname"))
def route(path):
    path = path if path in PAGES else "/"
    classes = ["nav-link active" if h == path else "nav-link" for _, h in NAV_ITEMS]
    return PAGES[path](), *classes


@app.callback(Output("pane-uni", "style"), Output("pane-bi", "style"), Output("pane-etl", "style"),
              Output("tab-uni", "className"), Output("tab-bi", "className"), Output("tab-etl", "className"),
              Input("tab-uni", "n_clicks"), Input("tab-bi", "n_clicks"), Input("tab-etl", "n_clicks"))
def switch_tab(*_):
    active = {"tab-uni": 0, "tab-bi": 1, "tab-etl": 2}.get(ctx.triggered_id, 0)
    styles = [{"display": "block"} if i == active else {"display": "none"} for i in range(3)]
    classes = ["tab-btn on" if i == active else "tab-btn" for i in range(3)]
    return *styles, *classes


@app.callback(Output("uni-graph", "figure"), Output("uni-summary", "children"), Input("uni-var", "value"))
def update_uni(v):
    return fig_univariate(v), univariate_summary(v)


@app.callback(Output("bi-graph", "figure"), Output("bi-summary", "children"), Input("bi-var", "value"))
def update_bi(v):
    return fig_bivariate(v), bivariate_summary(v)


@app.callback(Output("cm-graph", "figure"), Input("cm-model", "value"))
def update_cm(name):
    return fig_confusion(name)


@app.callback([Output(f"in-{k}", "value") for k in SLIDERS] + [Output("in-Genero", "value"), Output("in-chips", "value")],
              Input("preset-low", "n_clicks"), Input("preset-high", "n_clicks"), prevent_initial_call=True)
def apply_preset(*_):
    p = PRESETS["low" if ctx.triggered_id == "preset-low" else "high"]
    return [p[k] for k in SLIDERS] + [p["Genero"], p["chips"]]


@app.callback(Output("out-gauge", "figure"), Output("out-band", "children"),
              Output("out-contrib", "figure"), Output("out-models", "figure"),
              [Input(f"in-{k}", "value") for k in SLIDERS] +
              [Input("in-Genero", "value"), Input("in-chips", "value"), Input("in-model", "value")])
def predict(*args):
    values = dict(zip(SLIDERS, args[:len(SLIDERS)]))
    values["Genero"], values["chips"], model = args[len(SLIDERS)], args[len(SLIDERS) + 1], args[-1]
    row = build_row(values)
    p = float(MODELS[model].predict_proba(row)[0, 1])
    if p < 0.35:
        band, color, text = "Riesgo bajo", TEAL, "El perfil se parece más al de personas sin diabetes."
    elif p < 0.65:
        band, color, text = "Riesgo intermedio", AMBER, "Perfil mixto: conviene vigilar los factores modificables."
    else:
        band, color, text = "Riesgo alto", ROSE, "Perfil similar al de personas con diabetes o prediabetes."
    band_div = html.Div([
        html.P(band, className="risk-band", style={"color": color}),
        html.P(text, style={"margin": "0 0 8px", "color": FG}),
        html.P(f"Ajustada a la prevalencia poblacional (~{POP_PREVALENCE:.0%}), la probabilidad sería "
               f"{adjust_prior(p):.0%}.", className="note"),
    ])
    return fig_gauge(p), band_div, fig_contrib(row), fig_all_models(row)


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
