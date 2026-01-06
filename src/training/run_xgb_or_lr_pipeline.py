# src/training/run_xgb_or_lr_pipeline.py

import time
import os
import json
import logging
import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any, Callable, Tuple
from datetime import datetime

# Importiere Concrete-ML und Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef, classification_report
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier, LogisticRegression as ConcreteLogisticRegression
from tqdm.auto import tqdm

# Lokale Modul-Importe (angenommen, diese sind im sys.path)
from src.utils.logger import setup_logger, log_training_summary
from src.data.edge_iiot_dataset import load_edgeiiot_data # Annahme: load_edgeiiot_data wurde verschoben oder ist zugänglich

# --- Globale Model Map (für zukünftige Erweiterbarkeit) ---
MODEL_MAP = {
    "xgb": ConcreteXGBClassifier,
    "lr": ConcreteLogisticRegression
}

# --- Hilfsfunktionen aus dem Notebook/Projekt ---

def _load_raw_data_and_preprocess(config: Dict[str, Any], log: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, MinMaxScaler]:
    """Lädt, skaliert und splittet die EdgeIIoT/CICIoT Daten."""
    data_cfg = config.get("data_params", {})
    npz_file_path = data_cfg.get("npz_file_path")

    # Verwende die Funktion aus der data-Sektion deines Projekts
    # Wir brauchen die Rohdaten, aber ohne DataLoader
    _, _, _, raw = load_edgeiiot_data(
        npz_path=npz_file_path, 
        batch_size=1, # Irrelevant, da wir keine Loader verwenden
        return_raw_data=True, 
        logger=log
    )
    
    if raw is None:
        raise ValueError("Fehler beim Laden der Rohdaten aus NPZ.")

    X_train_raw = raw["X_train"].astype("float32")
    y_train_raw = raw["y_train"]
    X_test = raw["X_test"].astype("float32")
    y_test = raw["y_test"]
    scaler = raw["scaler"]
    label_encoder = raw["label_encoder"]
    
    # Concrete-ML Modelle arbeiten am besten mit skalierten Daten
    X_train_scaled = scaler.transform(X_train_raw)
    
    # Subset-Logik aus dem QAT-Runner (falls in data_params definiert)
    if data_cfg.get("use_subset_training", False):
        subset_fraction = data_cfg.get("subset_fraction", 0.1)
        subset_size = int(len(X_train_scaled) * subset_fraction)
        log.info(f"Verwende Subset von {subset_size} Trainingssamples ({subset_fraction*100:.1f}%).")
        X_train_scaled = X_train_scaled[:subset_size]
        y_train_raw = y_train_raw[:subset_size]
        
    # Sicherstellen, dass Labels int64 sind (wichtig für Concrete-ML)
    y_train_encoded = y_train_raw.astype("int64") 
    y_test_encoded = y_test.astype("int64")

    return X_train_scaled, y_train_encoded, X_test, y_test_encoded, label_encoder, scaler


def _train_and_compile_model(
    X_train: np.ndarray, y_train: np.ndarray, model_type: str, 
    optimal_params: Dict[str, Any], log: logging.Logger
) -> Tuple[Any, float]:
    """Trainiert das Concrete-ML Modell mit optimalen Parametern und kompiliert es."""
    
    ModelClass = MODEL_MAP[model_type]
    
    # 1. Training
    log.info(f"Starte Training des {model_type.upper()} mit optimalen Parametern: {optimal_params}")
    model = ModelClass(**optimal_params)
    
    start_train = time.time()
    with tqdm(total=1, desc=f"Training {model_type.upper()}") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)
    train_time = time.time() - start_train
    log.info(f"Training Time: {train_time:.4f} s")

    # 2. Kompilierung
    log.info("Starte FHE Kompilierung...")
    start_compile = time.time()
    # Kompilierung verwendet immer Trainingsdaten
    fhe_circuit = model.compile(X_train) 
    compile_time = time.time() - start_compile
    log.info(f"FHE Compile Time: {compile_time:.4f} s")

    return model, compile_time


def _evaluate_fhe_model_execute(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder,
    n_samples_execute: int, log_inference_time: bool, log: logging.Logger
) -> Dict[str, Any]:
    """Führt FHE-Inferenz (Execute Mode) durch und loggt Zeiten und Metriken."""
    
    if n_samples_execute <= 0:
        return {"status": "skipped", "reason": "n_samples_execute <= 0"}

    log.info(f"Starte FHE Evaluierung (Execute Mode) mit {n_samples_execute} Samples...")
    
    X_exec = X_test[:n_samples_execute]
    y_exec_true = y_test[:n_samples_execute]
    
    # FHE Prediction
    start_time = time.time()
    y_exec_pred = model.predict(X_exec, fhe="execute") # FHE Execute Mode
    total_inference_time_s = time.time() - start_time

    # Metriken berechnen
    acc = accuracy_score(y_exec_true, y_exec_pred)
    f1_weighted = f1_score(y_exec_true, y_exec_pred, average="weighted", zero_division=0)
    
    results = {
        "status": "success",
        "n_samples": n_samples_execute,
        "accuracy": acc,
        "f1_weighted": f1_weighted,
    }
    
    if log_inference_time:
        results["total_inference_time_s"] = total_inference_time_s
        results["time_per_sample_s"] = total_inference_time_s / n_samples_execute
        results["time_per_1000_samples_s"] = (total_inference_time_s / n_samples_execute) * 1000
        log.info(f"Execute Time ({n_samples_execute} samples): {total_inference_time_s:.3f} s")
        log.info(f"Execute Time / 1000 samples: {results['time_per_1000_samples_s']:.3f} s")

    # Log das Classification Report für den execute-Lauf
    log.info(f"Classification Report (Execute, {n_samples_execute} Samples):\n{classification_report(y_exec_true, y_exec_pred, target_names=label_encoder.classes_, zero_division=0)}")
    
    return results

# --- HAUPTFUNKTION FÜR DEN EXPERIMENT RUNNER ---

def run_concrete_ml_pipeline(config_path: str) -> Dict[str, Any]:
    """
    Führt die vollständige Concrete-ML Pipeline (Train, Compile, Execute Eval) aus.
    Diese Funktion wird vom experiment_runner.py aufgerufen.
    """
    config = {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"FEHLER: Konnte Config nicht laden: {e}")
        return {"status": "critical_error", "message": f"Config load failed: {str(e)}"}

    run_cfg = config.get("run_settings", {})
    model_cfg = config.get("model_params", {}) # Nicht verwendet, da XGBoost/LR params direkt im root liegen
    
    # Der experiment_runner.py steuert den logger_name.
    log_level_str = run_cfg.get("logger_level", "INFO").upper()
    logger_name = run_cfg.get("logger_instance_name", "concrete_ml_pipeline_run")
    log = setup_logger(name=logger_name, level=log_level_str)
    
    log.info(f"Starte Concrete-ML Pipeline mit Config: {config_path}")

    # --- 1. Parameter aus Config extrahieren ---
    if config.get("run_xgboost", False):
        model_type = "xgb"
        model_params_cfg = config.get("xgboost_params", {})
    elif config.get("run_logistic_regression", False):
        model_type = "lr"
        model_params_cfg = config.get("logistic_regression_params", {})
    else:
        log.error("Weder run_xgboost noch run_logistic_regression ist auf 'true' gesetzt. Abbruch.")
        return {"status": "error", "message": "No model run flag set."}
    
    optimal_params = model_params_cfg.get("optimal_params", {})
    if not optimal_params:
        log.error("FEHLER: 'optimal_params' nicht gefunden. Abbruch.")
        return {"status": "error", "message": "Optimal params missing."}
    
    fhe_eval_cfg = model_params_cfg.get("fhe_evaluation_params", {})
    n_samples_execute = fhe_eval_cfg.get("n_samples_execute", 0)
    log_inference_time = fhe_eval_cfg.get("log_inference_time", False)
    
    # --- 2. Daten laden und vorbereiten ---
    try:
        X_train_scaled, y_train_encoded, X_test, y_test_encoded, label_encoder, scaler = _load_raw_data_and_preprocess(config, log)
        log.info(f"Daten geladen. Train Shape: {X_train_scaled.shape}, Test Shape: {X_test.shape}")
    except Exception as e:
        log.error(f"Fehler beim Laden/Vorbereiten der Daten: {e}")
        return {"status": "error", "message": f"Data preparation failed: {str(e)}"}
    
    # --- 3. Training und Kompilierung ---
    try:
        model, compile_time = _train_and_compile_model(
            X_train_scaled, y_train_encoded, model_type, optimal_params, log
        )
    except Exception as e:
        log.error(f"Fehler beim Training/Kompilierung: {e}")
        return {"status": "error", "message": f"Training/Compilation failed: {str(e)}"}

    # --- 4. FHE Execute Evaluierung ---
    try:
        fhe_exec_results = _evaluate_fhe_model_execute(
            model, X_test, y_test_encoded, label_encoder,
            n_samples_execute, log_inference_time, log
        )
    except Exception as e:
        log.error(f"Fehler bei FHE Execute Evaluierung: {e}")
        fhe_exec_results = {"status": "error", "message": f"FHE Execute failed: {str(e)}"}


    # --- 5. Ergebnisse konsolidieren und zurückgeben (simuliert QAT-Runner-Format) ---
    
    # Wir führen KEINE Grid Search oder Val-Runs durch, aber wir brauchen Metriken für den Runner.
    # Wir verwenden die Test-Genauigkeit (unverschlüsselt) als Pseudo-Val-Metrik.
    # Achtung: Die 'fhe_simulate_f1_weighted' Logik des Runners würde fehlschlagen,
    # da wir die Simulation überspringen, aber wir loggen die execute-Metriken.
    
    test_pred_plain = model.predict(scaler.transform(X_test))
    test_f1_weighted = f1_score(y_test_encoded, test_pred_plain, average="weighted", zero_division=0)
    
    final_summary = {
        "status": "success",
        "model_type": model_type,
        "optimal_params": optimal_params,
        "compile_time_s": compile_time,
        "pytorch_test_f1_weighted": test_f1_weighted,
        "fhe_execute_evaluation": fhe_exec_results,
        # Für den Runner (der nach FHE-Metriken sucht):
        "fhe_simulate_f1_weighted": "N/A (Skipped)",
        "fhe_execute_f1_weighted": fhe_exec_results.get("f1_weighted", "N/A"),
    }
    
    # Logge die gesamte Zusammenfassung als JSON
    # Da wir KEINEN vollen QAT-Runner-JSON-Log erstellen, geben wir nur die Metriken zurück.
    
    log.info("Concrete-ML Pipeline abgeschlossen. Logge Metriken für Runner.")
    return final_summary