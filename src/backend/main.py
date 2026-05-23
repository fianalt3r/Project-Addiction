import os
import sys
import json
import joblib
import shap
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT             = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH        = os.path.join(ROOT, "data", "students_social_media_addiction.csv")
SHAP_BG_PATH     = os.path.join(ROOT, "data", "shap_background.csv")
PRED_LOG_PATH    = os.path.join(ROOT, "data", "predictions_log.csv")
MODEL_PATH       = os.path.join(ROOT, "model", "social_media_model.pkl")

# ── Feature definitions ────────────────────────────────────────────────────
NUMERICAL_COLS = [
    "Age",
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Mental_Health_Score",
    "Conflicts_Over_Social_Media",
]

CATEGORICAL_COLS = [
    "Gender",
    "Academic_Level",
    "Country",
    "Most_Used_Platform",
    "Affects_Academic_Performance",
    "Relationship_Status",
]

ALL_FEATURES = NUMERICAL_COLS + CATEGORICAL_COLS
CLASSES      = ["Low", "Moderate", "High"]


# ── Pydantic schemas ───────────────────────────────────────────────────────
class UserInput(BaseModel):
    Age: int
    Gender: str
    Academic_Level: str
    Country: str
    Avg_Daily_Usage_Hours: float
    Most_Used_Platform: str
    Affects_Academic_Performance: str
    Sleep_Hours_Per_Night: float
    Mental_Health_Score: float
    Conflicts_Over_Social_Media: int
    Relationship_Status: str
    Model_Name: str = "Random Forest"


class PredictionOutput(BaseModel):
    prediction: str
    probability_low: float
    probability_moderate: float
    probability_high: float
    shap_values: List[float]
    feature_names: List[str]


# ── Data loading & preparation ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])

    df["Addiction_Level"] = pd.cut(
        df["Addicted_Score"],
        bins=[0, 4, 7, 10],
        labels=CLASSES,
    ).astype(str)

    df = df.dropna(subset=["Addiction_Level"])
    return df


# ── Pipeline builder ───────────────────────────────────────────────────────
def build_pipeline(classifier):
    num_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_tf, NUMERICAL_COLS),
        ("cat", cat_tf, CATEGORICAL_COLS),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])


# ── Training ───────────────────────────────────────────────────────────────
def train_model():
    print("📦 Loading data...")
    df = load_data()
    X  = df[ALL_FEATURES]
    
    label_map = {"Low": 0, "Moderate": 1, "High": 2}
    rev_map   = {0: "Low", 1: "Moderate", 2: "High"}
    y  = df["Addiction_Level"].map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_test_str = [rev_map[val] for val in y_test]

    print("🤖 Tuning Random Forest Classifier with GridSearchCV...")
    rf_base = build_pipeline(
        RandomForestClassifier(random_state=42, class_weight="balanced")
    )
    rf_param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5],
    }
    rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    rf_cv_score = rf_grid.best_score_
    rf_best_params = {k.replace("classifier__", ""): v for k, v in rf_grid.best_params_.items()}

    print("📊 Tuning Logistic Regression with GridSearchCV...")
    lr_base = build_pipeline(
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    )
    lr_param_grid = {
        "classifier__C": [0.1, 1.0, 10.0],
    }
    lr_grid = GridSearchCV(lr_base, lr_param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    lr_best = lr_grid.best_estimator_
    lr_cv_score = lr_grid.best_score_
    lr_best_params = {k.replace("classifier__", ""): v for k, v in lr_grid.best_params_.items()}

    print("🚀 Tuning XGBoost with GridSearchCV...")
    xgb_base = build_pipeline(
        XGBClassifier(random_state=42, eval_metric="mlogloss")
    )
    xgb_param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
    }
    xgb_grid = GridSearchCV(xgb_base, xgb_param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_
    xgb_cv_score = xgb_grid.best_score_
    xgb_best_params = {k.replace("classifier__", ""): v for k, v in xgb_grid.best_params_.items()}

    rf_pred = [rev_map[p] for p in rf_best.predict(X_test)]
    lr_pred = [rev_map[p] for p in lr_best.predict(X_test)]
    xgb_pred = [rev_map[p] for p in xgb_best.predict(X_test)]

    metrics = {
        "rf_accuracy":    accuracy_score(y_test_str, rf_pred),
        "rf_f1":          f1_score(y_test_str, rf_pred, average="weighted"),
        "rf_cv_score":    rf_cv_score,
        "rf_best_params": rf_best_params,
        "rf_report":      classification_report(y_test_str, rf_pred, output_dict=True),
        "rf_confusion":   confusion_matrix(y_test_str, rf_pred, labels=CLASSES).tolist(),
        
        "lr_accuracy":    accuracy_score(y_test_str, lr_pred),
        "lr_f1":          f1_score(y_test_str, lr_pred, average="weighted"),
        "lr_cv_score":    lr_cv_score,
        "lr_best_params": lr_best_params,
        "lr_report":      classification_report(y_test_str, lr_pred, output_dict=True),
        "lr_confusion":   confusion_matrix(y_test_str, lr_pred, labels=CLASSES).tolist(),

        "xgb_accuracy":    accuracy_score(y_test_str, xgb_pred),
        "xgb_f1":          f1_score(y_test_str, xgb_pred, average="weighted"),
        "xgb_cv_score":    xgb_cv_score,
        "xgb_best_params": xgb_best_params,
        "xgb_report":      classification_report(y_test_str, xgb_pred, output_dict=True),
        "xgb_confusion":   confusion_matrix(y_test_str, xgb_pred, labels=CLASSES).tolist(),

        "X_test":         X_test,
        "y_test":         y_test_str,
        "classes":        CLASSES,
    }

    # Save SHAP background data (100 random training samples)
    bg = X_train.sample(min(100, len(X_train)), random_state=42)
    bg.to_csv(SHAP_BG_PATH, index=False)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"rf_pipeline": rf_best, "lr_pipeline": lr_best, "xgb_pipeline": xgb_best, "metrics": metrics}, MODEL_PATH)

    print(f"✅ RF  → Acc: {metrics['rf_accuracy']:.3f}  CV: {metrics['rf_cv_score']:.3f}  Params: {metrics['rf_best_params']}")
    print(f"✅ LR  → Acc: {metrics['lr_accuracy']:.3f}  CV: {metrics['lr_cv_score']:.3f}  Params: {metrics['lr_best_params']}")
    print(f"✅ XGB → Acc: {metrics['xgb_accuracy']:.3f}  CV: {metrics['xgb_cv_score']:.3f}  Params: {metrics['xgb_best_params']}")
    print(f"💾 Models saved → {MODEL_PATH}")
    return rf_best, lr_best, xgb_best, metrics


# ── SHAP helpers ───────────────────────────────────────────────────────────
def get_shap_for_input(pipeline, input_df, model_choice="Random Forest"):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]
    X_t          = preprocessor.transform(input_df)

    if model_choice == "Logistic Regression":
        bg_df = pd.read_csv(SHAP_BG_PATH)
        bg_t = preprocessor.transform(bg_df)
        explainer = shap.LinearExplainer(classifier, bg_t)
    else:
        explainer = shap.TreeExplainer(classifier)

    shap_vals = explainer.shap_values(X_t)
    # the predictions are now 0, 1, 2, so the index matches
    pred_idx  = int(np.argmax(pipeline.predict_proba(input_df)[0]))

    if isinstance(shap_vals, list):
        sv = shap_vals[pred_idx][0]
    else:
        if len(shap_vals.shape) == 3:
            sv = shap_vals[0, :, pred_idx]
        elif len(shap_vals.shape) == 2:
            sv = shap_vals[0]
        else:
            sv = shap_vals
    return sv.tolist(), ALL_FEATURES


def get_global_shap(pipeline, X_test, model_choice="Random Forest"):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier   = pipeline.named_steps["classifier"]
    X_t          = preprocessor.transform(X_test)

    if model_choice == "Logistic Regression":
        bg_df = pd.read_csv(SHAP_BG_PATH)
        bg_t = preprocessor.transform(bg_df)
        explainer = shap.LinearExplainer(classifier, bg_t)
    else:
        explainer = shap.TreeExplainer(classifier)

    shap_vals = explainer.shap_values(X_t)

    if isinstance(shap_vals, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
    else:
        if len(shap_vals.shape) == 3:
            mean_abs = np.abs(shap_vals).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(shap_vals).mean(axis=0)

    return mean_abs.tolist(), ALL_FEATURES


# ── Prediction logger ──────────────────────────────────────────────────────
def log_prediction(input_dict: dict, prediction: str, probabilities: dict):
    row = {**input_dict, "prediction": prediction, "timestamp": datetime.now().isoformat()}
    row.update({f"prob_{k.lower()}": v for k, v in probabilities.items()})
    df_row = pd.DataFrame([row])

    if os.path.exists(PRED_LOG_PATH):
        df_row.to_csv(PRED_LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(PRED_LOG_PATH, index=False)


# ── Load or train on startup ───────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    try:
        print("✅ Loading existing model...")
        bundle      = joblib.load(MODEL_PATH)
        rf_pipeline = bundle["rf_pipeline"]
        lr_pipeline = bundle["lr_pipeline"]
        xgb_pipeline = bundle["xgb_pipeline"]
        metrics     = bundle["metrics"]
    except Exception as e:
        print(f"⚠️ Failed to load model ({e}) — retraining...")
        rf_pipeline, lr_pipeline, xgb_pipeline, metrics = train_model()
else:
    print("🔧 No model found — training now...")
    rf_pipeline, lr_pipeline, xgb_pipeline, metrics = train_model()


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Social Media Addiction Predictor",
    description="Predicts student addiction risk: Low / Moderate / High",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "running", "message": "Social Media Addiction API is live 🚀"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: UserInput):
    try:
        input_dict = data.dict()
        model_choice = input_dict.pop("Model_Name", "Random Forest")
        input_df   = pd.DataFrame([input_dict])[ALL_FEATURES]

        if model_choice == "Logistic Regression":
            selected_pipeline = lr_pipeline
        elif model_choice == "XGBoost":
            selected_pipeline = xgb_pipeline
        else:
            selected_pipeline = rf_pipeline

        rev_map = {0: "Low", 1: "Moderate", 2: "High"}
        pred_num = selected_pipeline.predict(input_df)[0]
        prediction = rev_map[pred_num]
        
        # Pipeline predict_proba outputs probabilities for classes 0, 1, 2
        prob_vals = selected_pipeline.predict_proba(input_df)[0].tolist()
        probabilities = {rev_map[i]: prob_vals[i] for i in range(len(prob_vals))}

        shap_vals, feat_names = get_shap_for_input(selected_pipeline, input_df, model_choice)
        log_prediction(input_dict, prediction, probabilities)

        return PredictionOutput(
            prediction=prediction,
            probability_low=float(probabilities.get("Low", 0)),
            probability_moderate=float(probabilities.get("Moderate", 0)),
            probability_high=float(probabilities.get("High", 0)),
            shap_values=shap_vals,
            feature_names=feat_names,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-metrics")
def get_metrics():
    return {
        "rf_accuracy":              round(metrics["rf_accuracy"], 4),
        "rf_f1":                    round(metrics["rf_f1"], 4),
        "rf_cv_score":              round(metrics.get("rf_cv_score", 0), 4),
        "rf_best_params":           metrics.get("rf_best_params", {}),
        "lr_accuracy":              round(metrics["lr_accuracy"], 4),
        "lr_f1":                    round(metrics["lr_f1"], 4),
        "lr_cv_score":              round(metrics.get("lr_cv_score", 0), 4),
        "lr_best_params":           metrics.get("lr_best_params", {}),
        "xgb_accuracy":             round(metrics["xgb_accuracy"], 4),
        "xgb_f1":                   round(metrics["xgb_f1"], 4),
        "xgb_cv_score":             round(metrics.get("xgb_cv_score", 0), 4),
        "xgb_best_params":          metrics.get("xgb_best_params", {}),
        "rf_confusion_matrix":      metrics["rf_confusion"],
        "lr_confusion_matrix":      metrics["lr_confusion"],
        "xgb_confusion_matrix":     metrics["xgb_confusion"],
        "rf_classification_report": metrics["rf_report"],
        "xgb_report":               metrics["xgb_report"],
        "classes":                  CLASSES,
    }


@app.get("/feature-importance")
def feature_importance(model_name: str = "Random Forest"):
    try:
        X_test               = metrics["X_test"]
        if model_name == "Logistic Regression":
            selected_pipeline = lr_pipeline
        elif model_name == "XGBoost":
            selected_pipeline = xgb_pipeline
        else:
            selected_pipeline = rf_pipeline
        mean_shap, feat_names = get_global_shap(selected_pipeline, X_test, model_name)
        return {"feature_names": feat_names, "mean_shap_values": mean_shap}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions-log")
def predictions_log():
    if not os.path.exists(PRED_LOG_PATH):
        return {"data": [], "message": "No predictions made yet."}
    df = pd.read_csv(PRED_LOG_PATH)
    return {"data": df.tail(50).to_dict(orient="records")}
