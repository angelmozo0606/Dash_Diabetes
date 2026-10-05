
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
ART = BASE / "artifacts"
SEED = 42

RENAME = {
    "Diabetes_binary": "Diabetes", "HighBP": "PA_alta", "HighChol": "col_alto",
    "CholCheck": "CheckCol", "BMI": "IMC", "Smoker": "Fumador", "Stroke": "derrame",
    "HeartDiseaseorAttack": "EnfCardi", "PhysActivity": "ActFisi", "Fruits": "fruta",
    "Veggies": "vegetales", "HvyAlcoholConsump": "Alcohlico", "AnyHealthcare": "eps",
    "NoDocbcCost": "AfordabilidadMedica", "GenHlth": "SaludG", "MentHlth": "SaludM",
    "PhysHlth": "SaludF", "DiffWalk": "DificultadCaminar", "Sex": "Genero",
    "Age": "Edad", "Education": "Educacion", "Income": "Ingresos",
}
TARGET = "Diabetes"
FEATURES = [c for c in RENAME.values() if c != TARGET]


def load_data():
    raw = pd.read_csv(DATA)
    df = raw.rename(columns=RENAME)[[TARGET] + FEATURES]
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df.astype(float), n_raw


def build_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
            random_state=SEED, n_jobs=-1),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=10,
            random_state=SEED, n_jobs=-1),
        "Regresión Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000))]),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, random_state=SEED),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=35))]),
        "Naive Bayes": GaussianNB(),
    }


def train():
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    df, n_raw = load_data()
    X, y = df[FEATURES], df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models, results, rocs, cms = {}, [], {}, {}

    for name, model in build_models().items():
        t = time.time()
        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        p_tr = model.predict_proba(X_tr)[:, 1]
        y_hat = (p_te >= 0.5).astype(int)
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)

        results.append(dict(
            name=name,
            auc=round(roc_auc_score(y_te, p_te), 4),
            acc=round(accuracy_score(y_te, y_hat), 4),
            prec=round(precision_score(y_te, y_hat), 4),
            rec=round(recall_score(y_te, y_hat), 4),
            f1=round(f1_score(y_te, y_hat), 4),
            cv=round(cv_auc.mean(), 4),
            cv_std=round(cv_auc.std(), 4),
            tr=round(roc_auc_score(y_tr, p_tr), 4),
        ))
        fpr, tpr, _ = roc_curve(y_te, p_te)
        idx = np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int)
        rocs[name] = dict(fpr=fpr[idx].round(4).tolist(), tpr=tpr[idx].round(4).tolist())
        cms[name] = confusion_matrix(y_te, y_hat).tolist()
        models[name] = model
        print(f"  {name:<22} AUC={results[-1]['auc']:.4f}  ({time.time() - t:.1f}s)")

    results.sort(key=lambda r: r["auc"], reverse=True)
    best = results[0]["name"]

    # Importancia por permutación del mejor modelo (muestra del test para rapidez)
    sample = X_te.sample(5000, random_state=SEED)
    perm = permutation_importance(models[best], sample, y_te.loc[sample.index],
                                  scoring="roc_auc", n_repeats=5, random_state=SEED)
    imp = sorted(zip(FEATURES, perm.importances_mean), key=lambda t: t[1], reverse=True)

    # Coeficientes de la regresión logística (variables estandarizadas)
    lr = models["Regresión Logística"]
    coefs = sorted(zip(FEATURES, lr.named_steps["clf"].coef_[0]),
                   key=lambda t: abs(t[1]), reverse=True)

    metrics = dict(
        n_raw=n_raw, n_clean=len(df), n_dup=n_raw - len(df),
        n_train=len(X_tr), n_test=len(X_te), features=FEATURES, best=best,
        results=results, roc=rocs, cm=cms,
        importance=[dict(f=f, v=round(float(v), 5)) for f, v in imp],
        lr_coef=[dict(f=f, v=round(float(v), 4)) for f, v in coefs],
    )
    joblib.dump(models, ART / "models.joblib", compress=3)
    (ART / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1))
    print(f"Listo en {time.time() - t0:.0f}s → mejor modelo: {best}")
    return models, metrics


if __name__ == "__main__":
    train()
