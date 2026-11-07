"""App Streamlit del ACA 2: red neuronal (ONNX) para predecir enfermedad cardiaca.

Uso:  streamlit run app.py
Requiere en esta carpeta: model.onnx, metrics.json, preproc.json, heart.db, deep_utils.py
(se generan al ejecutar el notebook en Colab).
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st

from deep_utils import FEATURES, transform

BASE = Path(__file__).parent

st.set_page_config(page_title="Red neuronal: enfermedad cardíaca", page_icon="🫀", layout="wide")


@st.cache_resource
def load_session():
    return ort.InferenceSession(str(BASE / "model.onnx"), providers=["CPUExecutionProvider"])


@st.cache_data
def load_json(name):
    return json.loads((BASE / name).read_text(encoding="utf-8"))


def run_query(sql):
    con = sqlite3.connect((BASE / "heart.db").as_uri() + "?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


missing = [f for f in ["model.onnx", "metrics.json", "preproc.json", "heart.db"] if not (BASE / f).exists()]
if missing:
    st.error(f"Faltan archivos: {', '.join(missing)}. Ejecuta el notebook en Colab y copia los resultados aquí.")
    st.stop()

session, metrics, prep = load_session(), load_json("metrics.json"), load_json("preproc.json")
best = metrics["best_model"]

st.title("🫀 Red neuronal para la detección asistida de enfermedad cardíaca")
st.caption(
    "ACA N° 2 · Aprendizaje Automático · CUN · Especialización en IA · "
    f"Modelo desplegado: **{best}** (ONNX). Herramienta académica: no sustituye un diagnóstico médico."
)

tab_pred, tab_metrics, tab_arch, tab_sql = st.tabs(
    ["🔮 Predicción", "📊 Métricas y curvas", "🧠 Arquitectura", "🗄️ Datos (SQL)"]
)

# ------------------------------------------------------------------ Predicción
with tab_pred:
    st.subheader("Ingresa los datos del paciente")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Edad", 20, 90, 55)
        sex = st.radio("Sexo", [1, 0], format_func=lambda v: "Masculino" if v else "Femenino", horizontal=True)
        cp = st.selectbox(
            "Tipo de dolor de pecho",
            [1, 2, 3, 4],
            format_func=lambda v: {1: "Angina típica", 2: "Angina atípica", 3: "Dolor no anginoso", 4: "Asintomático"}[v],
        )
        trestbps = st.slider("Presión arterial en reposo (mm Hg)", 80, 220, 130)
    with c2:
        chol = st.slider("Colesterol sérico (mg/dl)", 100, 600, 240)
        fbs = st.radio("Glucemia en ayunas > 120 mg/dl", [0, 1], format_func=lambda v: "Sí" if v else "No", horizontal=True)
        restecg = st.selectbox(
            "ECG en reposo",
            [0, 1, 2],
            format_func=lambda v: {0: "Normal", 1: "Anomalía onda ST-T", 2: "Hipertrofia ventricular izq."}[v],
        )
        thalach = st.slider("Frecuencia cardíaca máxima", 60, 220, 150)
    with c3:
        exang = st.radio("Angina inducida por ejercicio", [0, 1], format_func=lambda v: "Sí" if v else "No", horizontal=True)
        oldpeak = st.slider("Depresión del ST (oldpeak)", 0.0, 7.0, 1.0, 0.1)
        slope = st.selectbox(
            "Pendiente del segmento ST",
            [1, 2, 3],
            format_func=lambda v: {1: "Ascendente", 2: "Plana", 3: "Descendente"}[v],
        )
        ca = st.selectbox("Vasos mayores coloreados (0-3)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Talasemia",
            [3, 6, 7],
            format_func=lambda v: {3: "Normal", 6: "Defecto fijo", 7: "Defecto reversible"}[v],
        )

    if st.button("Predecir", type="primary"):
        row = pd.DataFrame(
            [{
                "age": age, "trestbps": trestbps, "chol": chol, "thalach": thalach, "oldpeak": oldpeak,
                "sex": sex, "cp": cp, "fbs": fbs, "restecg": restecg, "exang": exang,
                "slope": slope, "ca": float(ca), "thal": float(thal),
            }]
        )[FEATURES]
        x_num, x_cat = transform(row, prep)
        logit = float(session.run(None, {"x_num": x_num, "x_cat": x_cat})[0].reshape(-1)[0])
        proba = 1.0 / (1.0 + np.exp(-logit))
        st.metric("Probabilidad estimada de enfermedad cardíaca", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
        if proba >= 0.5:
            st.error("La red neuronal clasifica este caso como **riesgo alto** de enfermedad cardíaca.")
        else:
            st.success("La red neuronal clasifica este caso como **riesgo bajo** de enfermedad cardíaca.")

# --------------------------------------------------------------------- Métricas
with tab_metrics:
    res = metrics["results"]
    st.subheader(f"Modelo seleccionado: {best}")
    st.caption(f"Selección por menor pérdida en validación. Partición: {metrics['split']}")
    table = pd.DataFrame(
        {n: {k: r[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "test_loss", "epochs", "params"]}
         for n, r in res.items()}
    ).T
    st.dataframe(table)

    cols = st.columns(2)
    for col, name in zip(cols, res):
        hist = res[name]["history"]
        with col:
            st.write(f"**{name}** · curva de pérdida")
            st.line_chart(pd.DataFrame({"Entrenamiento": hist["loss"], "Validación": hist["val_loss"]}))
            st.write(f"**{name}** · curva de exactitud")
            st.line_chart(pd.DataFrame({"Entrenamiento": hist["acc"], "Validación": hist["val_acc"]}))

    st.write(f"**Matriz de confusión de {best}** (conjunto de prueba)")
    st.dataframe(
        pd.DataFrame(
            res[best]["confusion_matrix"],
            index=["Real: sano", "Real: enfermo"],
            columns=["Pred: sano", "Pred: enfermo"],
        )
    )
    st.caption("En un contexto clínico es clave el recall: un falso negativo es más costoso que un falso positivo.")

# ------------------------------------------------------------------ Arquitectura
with tab_arch:
    st.subheader("Arquitecturas entrenadas")
    for name, text in metrics["architectures"].items():
        st.markdown(f"**{name}**  \n{text}")
    st.info(
        "Flujo: variables clínicas → preprocesamiento (imputación, estandarización, índices de categorías) "
        "→ red neuronal (PyTorch) → exportación a ONNX → inferencia en Streamlit con ONNX Runtime."
    )

# -------------------------------------------------------------------------- SQL
with tab_sql:
    st.subheader("Base de datos SQL (SQLite): tabla `pacientes`")
    queries = {
        "Resumen general": "SELECT COUNT(*) AS pacientes, SUM(target) AS con_enfermedad, "
                           "ROUND(100.0*AVG(target),1) AS pct_enfermos FROM pacientes",
        "Enfermedad por sexo": "SELECT sex AS sexo, COUNT(*) AS n, ROUND(100.0*AVG(target),1) AS pct_enfermos "
                               "FROM pacientes GROUP BY sex",
        "Enfermedad por tipo de dolor de pecho": "SELECT cp AS dolor_pecho, COUNT(*) AS n, "
                                                 "ROUND(100.0*AVG(target),1) AS pct_enfermos FROM pacientes GROUP BY cp ORDER BY cp",
        "Enfermedad por grupo de edad": "SELECT CASE WHEN age<40 THEN '<40' WHEN age<55 THEN '40-54' "
                                        "WHEN age<65 THEN '55-64' ELSE '65+' END AS edad, COUNT(*) AS n, "
                                        "ROUND(100.0*AVG(target),1) AS pct_enfermos FROM pacientes GROUP BY 1 ORDER BY 1",
        "Primeros 50 registros": "SELECT * FROM pacientes LIMIT 50",
    }
    choice = st.selectbox("Consulta predefinida", list(queries))
    st.code(queries[choice], language="sql")
    st.dataframe(run_query(queries[choice]))
