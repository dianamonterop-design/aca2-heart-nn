"""Utilidades compartidas por el notebook y la app Streamlit (solo pandas + numpy).

Dataset: UCI Heart Disease (Janosi et al., 1988). Objetivo binario: 1 = enfermedad cardiaca.
El preprocesamiento se aprende con el conjunto de entrenamiento, se guarda en preproc.json
y la app lo reaplica sin depender de scikit-learn.
"""
import numpy as np
import pandas as pd

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"


def fit_preprocessor(df: pd.DataFrame) -> dict:
    """Aprende medianas, medias, desviaciones, modas y categorias (solo con train)."""
    prep = {
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "median": {},
        "mean": {},
        "std": {},
        "mode": {},
        "categories": {},
    }
    for c in NUMERIC:
        med = float(df[c].median())
        filled = df[c].astype(float).fillna(med)
        std = float(filled.std(ddof=0))
        prep["median"][c] = med
        prep["mean"][c] = float(filled.mean())
        prep["std"][c] = std if std > 0 else 1.0
    for c in CATEGORICAL:
        prep["mode"][c] = float(df[c].mode().iloc[0])
        prep["categories"][c] = sorted({float(v) for v in df[c].dropna().unique()})
    return prep


def transform(df: pd.DataFrame, prep: dict):
    """DataFrame -> (x_num float32 [N, 5] estandarizado, x_cat int64 [N, 8] con indices)."""
    num = np.column_stack(
        [
            (df[c].astype(float).fillna(prep["median"][c]).to_numpy() - prep["mean"][c]) / prep["std"][c]
            for c in prep["numeric"]
        ]
    ).astype(np.float32)

    cols = []
    for c in prep["categorical"]:
        index = {float(v): i for i, v in enumerate(prep["categories"][c])}
        default = index[float(prep["mode"][c])]
        values = df[c].astype(float).fillna(prep["mode"][c]).to_numpy()
        cols.append([index.get(float(v), default) for v in values])
    cat = np.array(cols, dtype=np.int64).T
    return num, cat
