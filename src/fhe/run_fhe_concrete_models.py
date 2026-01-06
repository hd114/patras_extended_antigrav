#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_fhe_concrete_models.py

Dieses Skript trainiert und evaluiert XGBoost- und LogisticRegression-Modelle
unter Verwendung von Concrete ML für Fully Homomorphic Encryption (FHE).
Es beinhaltet Hyperparameter-Suche, FHE-Simulation und -Ausführungszeitmessung.
"""

import os
import shutil
from pathlib import Path
import time
from typing import Callable, Dict, Tuple, Any, Optional, List
import argparse
import logging
import yaml

import matplotlib.pyplot as plt # Nur wenn Sie Plots direkt speichern wollen
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # MinMaxScaler direkt hier definiert
from sklearn.metrics import accuracy_score
from tqdm import tqdm # Geändert von tqdm.notebook.tqdm

import torch # Für Typ-Annotationen und ggf. Basis-Dataset
from torch.utils.data import Dataset # Basis-Dataset, falls EdgeIIotDataset davon erbt

from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier
# from concrete.ml.deployment import FHEModelDev, FHEModelServer # Falls Deployment benötigt wird

# --- Globale Konfigurationen (ggf. in eine YAML-Datei auslagern) ---

# Pfad zur NPZ-Datei. Passen Sie dies an Ihren tatsächlichen Pfad an.
# Der Pfad aus Ihrem Notebook: "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
# oder für EdgeIIoT: "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
DEFAULT_NPZ_PATH = "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz" # Beispiel, anpassen!

# Basisverzeichnis für alle Ergebnisse dieses Skripts
RESULTS_BASE_DIR = "results/fhe_concrete_model_evaluations"
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)

# Logging Konfiguration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# --- Daten-Klassen und Ladefunktionen (aus dem Notebook übernommen) ---
# Hinweis: Wenn Sie bereits eine EdgeIIotDataset-Klasse in Ihrem Projekt haben,
# überlegen Sie, diese zu verwenden oder zu vereinheitlichen.

class EdgeIIotDataset(Dataset): # Erbt von torch.utils.data.Dataset
    """
    Benutzerdefiniertes Dataset für EdgeIIoT-Daten, kompatibel mit PyTorch DataLoader,
    obwohl Concrete-ML Sklearn-Modelle primär NumPy-Arrays erwarten.
    Die Skalierung und Label-Kodierung erfolgt hier.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 label_encoder: Optional[LabelEncoder] = None,
                 scaler: Optional[MinMaxScaler] = None,
                 is_pre_encoded: bool = False,
                 is_pre_scaled: bool = False):
        self.label_encoder = label_encoder
        self.scaler = scaler

        if is_pre_encoded:
            encoded_labels = labels
            if self.label_encoder is None: # Encoder muss trotzdem gesetzt werden für inverse_transform
                logger.warning("Labels sind als 'is_pre_encoded' markiert, aber es wurde kein LabelEncoder übergeben. "
                               "Inverse Transformation wird nicht möglich sein.")
        else:
            if self.label_encoder is None:
                logger.info("Erstelle und fitte neuen LabelEncoder.")
                self.label_encoder = LabelEncoder().fit(labels)
            encoded_labels = self.label_encoder.transform(labels)
        
        self.labels = encoded_labels

        if is_pre_scaled:
            scaled_features = features
            if self.scaler is None:
                logger.warning("Features sind als 'is_pre_scaled' markiert, aber es wurde kein Scaler übergeben.")
        else:
            if self.scaler is None:
                logger.info("Erstelle und fitte neuen MinMaxScaler.")
                self.scaler = MinMaxScaler().fit(features)
            scaled_features = self.scaler.transform(features)
        
        self.features = scaled_features


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        # Gibt NumPy-Arrays zurück, wie für Sklearn-Modelle erwartet
        return self.features[idx], self.labels[idx]

    def inverse_transform_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """Transformiert kodierte Labels zurück zu den Original-Labels."""
        if self.label_encoder:
            return self.label_encoder.inverse_transform(encoded_labels)
        raise ValueError("LabelEncoder nicht initialisiert.")


def load_and_prepare_data(
    path: str,
    refit_scaler: bool = False, # Ob der Scaler neu gefittet werden soll
    refit_encoder: bool = False # Ob der Encoder neu gefittet werden soll (benötigt Roh-String-Labels)
) -> Dict[str, Any]:
    """
    Lädt und bereitet die EdgeIIoT-Daten aus einer NPZ-Datei auf.
    Die Skalierung und Label-Kodierung wird hier verwaltet.

    Args:
        path (str): Pfad zur NPZ-Datei.
        refit_scaler (bool): Wenn True, wird der MinMaxScaler auf X_train_raw neu gefittet.
                             Andernfalls werden gespeicherte Scaler-Parameter verwendet.
        refit_encoder (bool): Wenn True, wird der LabelEncoder neu gefittet.
                              ACHTUNG: Benötigt originale String-Labels im NPZ,
                              die aktuell nicht im Notebook-Schema enthalten sind.

    Returns:
        Dict[str, Any]: Ein Dictionary mit den aufbereiteten Daten (X_train, y_train etc.),
                        dem Scaler und dem LabelEncoder.
    """
    try:
        data_npz = np.load(path, allow_pickle=True)
        logger.info(f"Rohdaten geladen von: {path}")
    except FileNotFoundError:
        logger.error(f"NPZ-Datei nicht gefunden unter: {path}")
        raise
    except Exception as e:
        logger.error(f"Fehler beim Laden der NPZ-Datei {path}: {e}")
        raise

    X_train_raw = data_npz["X_train_raw"]
    X_val_raw = data_npz["X_val_raw"]
    X_test_raw = data_npz["X_test_raw"]

    # Labels sind im NPZ bereits als 'encoded' gespeichert
    y_train_encoded = data_npz["y_train_encoded"]
    y_val_encoded = data_npz["y_val_encoded"]
    y_test_encoded = data_npz["y_test_encoded"]

    label_encoder = LabelEncoder()
    if "label_classes" in data_npz:
        label_encoder.classes_ = data_npz["label_classes"]
        logger.info(f"LabelEncoder Klassen geladen: {label_encoder.classes_}")
    else:
        logger.warning("Keine 'label_classes' im NPZ gefunden. LabelEncoder wird ggf. neu gefittet.")
        # Dies würde fehlschlagen, da y_train_encoded bereits Zahlen sind.
        # Für ein Refit bräuchte man die originalen String-Labels.
        if refit_encoder:
             raise NotImplementedError(
                "Refit von LabelEncoder erfordert originale String-Labels im NPZ, die aktuell fehlen."
            )
        # Versuche, aus den kombinierten encoded Labels zu fitten (nicht ideal, aber ein Fallback)
        all_encoded_labels = np.concatenate([y_train_encoded, y_val_encoded, y_test_encoded])
        label_encoder.fit(all_encoded_labels) # Lernt die numerischen 'Klassen'
        logger.info(f"LabelEncoder auf numerischen Labels gefittet. Klassen: {label_encoder.classes_}")


    if refit_scaler:
        logger.info("MinMaxScaler wird auf X_train_raw neu gefittet.")
        scaler = MinMaxScaler().fit(X_train_raw)
    else:
        if "scaler_min" in data_npz and "scaler_scale" in data_npz:
            scaler = MinMaxScaler()
            scaler.min_ = data_npz["scaler_min"]
            scaler.scale_ = data_npz["scaler_scale"]
            # Notwendig für korrekte inverse_transform Funktionalität des Scalers
            scaler.data_min_ = data_npz.get("scaler_data_min", scaler.min_)
            scaler.data_max_ = data_npz.get("scaler_data_max", (scaler.min_ + 1.0 / scaler.scale_) if scaler.scale_ is not None and not np.all(scaler.scale_ == 0) else scaler.min_ + 1.0)
            scaler.data_range_ = scaler.data_max_ - scaler.data_min_
            logger.info("Gespeicherte MinMaxScaler Parameter geladen.")
        else:
            logger.warning("Keine Scaler-Parameter im NPZ gefunden und refit_scaler ist False. "
                           "MinMaxScaler wird auf X_train_raw gefittet.")
            scaler = MinMaxScaler().fit(X_train_raw)

    # Skalieren der Feature-Sets
    X_train_scaled = scaler.transform(X_train_raw).astype("float32")
    X_val_scaled = scaler.transform(X_val_raw).astype("float32")
    X_test_scaled = scaler.transform(X_test_raw).astype("float32")

    logger.info(f"Datenaufbereitung abgeschlossen. Shapes: "
                f"X_train: {X_train_scaled.shape}, X_val: {X_val_scaled.shape}, X_test: {X_test_scaled.shape}")

    return {
        "X_train": X_train_scaled, "y_train": y_train_encoded,
        "X_val": X_val_scaled, "y_val": y_val_encoded,
        "X_test": X_test_scaled, "y_test": y_test_encoded,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "num_features": X_train_scaled.shape[1],
        "num_classes": len(label_encoder.classes_) if hasattr(label_encoder, 'classes_') else 0
    }


def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray,
                        clf_class: Callable, grid_param: Dict,
                        cv_folds: int = 3, n_jobs: int = -1) -> GridSearchCV:
    """
    Führt eine Grid Search für einen gegebenen Klassifikator durch.
    """
    logger.info(f"Starte GridSearchCV für {clf_class.__name__} mit Parametern: {grid_param}")
    grid_search_obj = GridSearchCV(
        clf_class(),  # Instanziiere den Klassifikator hier
        grid_param,
        cv=cv_folds,
        verbose=1,
        n_jobs=n_jobs,
        scoring='f1_weighted' # oder eine andere passende Metrik
    )
    grid_search_obj.fit(X_train, y_train)

    logger.info(f"GridSearchCV abgeschlossen für {clf_class.__name__}.")
    logger.info(f"Beste Parameter: {grid_search_obj.best_params_}")
    logger.info(f"Bester Score (f1_weighted CV): {grid_search_obj.best_score_:.4f}")

    return grid_search_obj


def evaluate_fhe_performance(
    grid_search_results: pd.DataFrame, # DataFrame der GridSearch Ergebnisse
    clf_class: Callable, # Die Klasse des Klassifikators (z.B. ConcreteXGBClassifier)
    X_train: np.ndarray, y_train: np.ndarray, # Trainingsdaten zum Fitten
    X_test_sample: np.ndarray, # Ein einzelnes oder wenige Samples für die FHE-Zeitmessung
    parameter_keys: List[str], # Schlüssel der Parameter im DataFrame
    results_dir: str,
    model_name_prefix: str
) -> pd.DataFrame:
    """
    Misst die Inferenzzeit im FHE-Modus ('execute') für jede Parameterkombination
    aus den GridSearch-Ergebnissen. Kompiliert das Modell bei jedem Mal neu.
    """
    logger.info(f"Starte FHE Performance Evaluierung für {model_name_prefix}...")
    
    # Stelle sicher, dass X_test_sample die richtige Form hat (z.B. für Einzel-Sample (1, n_features))
    if X_test_sample.ndim == 1:
        X_test_sample = X_test_sample.reshape(1, -1)

    fhe_inference_times = []
    fhe_compilation_times = []
    all_params_configs = []

    for _, param_row in tqdm(grid_search_results[parameter_keys].iterrows(), 
                             total=len(grid_search_results),
                             desc=f"FHE Eval {model_name_prefix}"):
        
        current_params = param_row.to_dict()
        all_params_configs.append(current_params) # Für spätere Referenz

        # Instanziierung und Training des Modells mit der aktuellen Parameterkombination
        try:
            clf = clf_class(**current_params)
            clf.fit(X_train, y_train)

            # Kompilierung für FHE (benötigt Trainingsdaten für Quantisierungsbereiche)
            compile_start_time = time.time()
            # Verwende einen kleinen, repräsentativen Teil von X_train für die Kompilierung, falls X_train sehr groß ist
            compile_sample = X_train[:min(1000, len(X_train))]
            clf.compile(compile_sample) # Concrete-ML's compile
            compilation_time = time.time() - compile_start_time
            fhe_compilation_times.append(compilation_time)

            # FHE Ausführungszeitmessung (mit 'execute')
            # Key Generierung ist implizit in Concrete-ML oder muss separat gehandhabt werden,
            # je nach Version und genauer Nutzung. Für .predict(fhe="execute") ist sie oft intern.
            # Für reale Szenarien würde man Client/Server-Keys explizit verwalten.
            inference_start_time = time.time()
            _ = clf.predict(X_test_sample, fhe="execute")
            inference_time = time.time() - inference_start_time
            fhe_inference_times.append(inference_time)
            
            logger.debug(f"Params: {current_params}, Compile: {compilation_time:.2f}s, Infer: {inference_time:.2f}s")

        except Exception as e:
            logger.error(f"Fehler bei FHE Evaluierung für Parameter {current_params}: {e}")
            logger.error(traceback.format_exc())
            fhe_inference_times.append(np.nan) # NaN bei Fehler
            fhe_compilation_times.append(np.nan)

    # Füge die Zeiten zu einer Kopie des Ergebnis-DataFrames hinzu
    # Es ist wichtig, dass die Reihenfolge der Iteration über grid_search_results erhalten bleibt.
    results_with_fhe = grid_search_results.copy()
    # Wenn die Längen nicht übereinstimmen, gab es Fehler
    if len(fhe_inference_times) == len(results_with_fhe):
        results_with_fhe["fhe_compilation_time_s"] = fhe_compilation_times
        results_with_fhe["fhe_inference_time_s_execute"] = fhe_inference_times
    else:
        logger.error("Längenunterschied zwischen Ergebnissen und gemessenen FHE-Zeiten. Fülle mit NaN.")
        results_with_fhe["fhe_compilation_time_s"] = pd.Series(fhe_compilation_times, index=results_with_fhe.index) # Versuche, Index anzupassen
        results_with_fhe["fhe_inference_time_s_execute"] = pd.Series(fhe_inference_times, index=results_with_fhe.index)
        # Fülle fehlende Werte mit NaN, falls die Serien kürzer sind
        results_with_fhe["fhe_compilation_time_s"] = results_with_fhe["fhe_compilation_time_s"].reindex(results_with_fhe.index)
        results_with_fhe["fhe_inference_time_s_execute"] = results_with_fhe["fhe_inference_time_s_execute"].reindex(results_with_fhe.index)


    # Speichere die erweiterten Ergebnisse
    output_filename = os.path.join(results_dir, f"{model_name_prefix}_grid_search_fhe_results.csv")
    results_with_fhe.to_csv(output_filename, index=False)
    logger.info(f"{model_name_prefix} FHE Evaluierungsergebnisse gespeichert in: {output_filename}")
    
    return results_with_fhe


def run_experiment(config: Dict[str, Any]):
    """
    Hauptfunktion zur Durchführung eines einzelnen Experimentsatzes
    (z.B. für XGBoost oder Logistic Regression).
    """
    dataset_path = config.get("dataset_path", DEFAULT_NPZ_PATH)
    experiment_name = config.get("experiment_name", "FHE_Experiment")
    
    # Erstelle ein Unterverzeichnis für die Ergebnisse dieses Experiments
    current_experiment_dir = os.path.join(RESULTS_BASE_DIR, experiment_name)
    os.makedirs(current_experiment_dir, exist_ok=True)
    logger.info(f"Ergebnisse für '{experiment_name}' werden in '{current_experiment_dir}' gespeichert.")

    # 1. Daten laden und vorbereiten
    logger.info(f"Lade und verarbeite Daten von: {dataset_path}")
    try:
        data_dict = load_and_prepare_data(path=dataset_path)
    except Exception as e:
        logger.error(f"Kritischer Fehler beim Laden der Daten: {e}. Experiment wird abgebrochen.")
        return

    X_train, y_train = data_dict["X_train"], data_dict["y_train"]
    X_val, y_val = data_dict["X_val"], data_dict["y_val"] # Validierungsset für finale Modellwahl nutzen
    X_test, y_test = data_dict["X_test"], data_dict["y_test"]
    
    # Stichprobe für FHE-Zeitmessung (z.B. die ersten paar Test-Samples)
    # Concrete-ML erwartet oft ein 2D-Array, auch für ein einzelnes Sample.
    fhe_timing_sample = X_test[:min(5, len(X_test))] if len(X_test) > 0 else X_train[:min(5, len(X_train))]
    if fhe_timing_sample.size == 0:
        logger.error("Keine Daten für FHE Timing Sample verfügbar. Überspringe FHE Zeitmessungen.")
        # Hier könnte man entscheiden, ob man abbricht oder ohne FHE-Zeiten weitermacht
        # Fürs Erste wird evaluate_fhe_performance dann mit einem leeren Array aufgerufen oder übersprungen.


    # --- XGBoost Experiment ---
    if config.get("run_xgboost", False):
        logger.info("\n--- Starte XGBoost Experiment ---")
        xgb_grid_params = config.get("xgboost_params", {}).get("grid_search_params", {
            "n_bits": [8, 10], "max_depth": [6, 7], "n_estimators": [5, 6] # Kleinere Standardwerte
        })
        
        start_time_xgb_gs = time.time()
        gs_xgb_results_obj = perform_grid_search(X_train, y_train, ConcreteXGBClassifier, xgb_grid_params)
        logger.info(f"XGBoost Grid Search Dauer: {time.time() - start_time_xgb_gs:.2f}s")
        
        xgb_gs_df = pd.DataFrame(gs_xgb_results_obj.cv_results_)
        # Parameter-Spaltennamen aufräumen (entferne "param_")
        xgb_gs_df.columns = xgb_gs_df.columns.str.replace("param_", "")
        xgb_parameter_keys = list(xgb_grid_params.keys())
        xgb_results_filtered = xgb_gs_df[xgb_parameter_keys + ["mean_test_score", "std_test_score"]]
        xgb_results_sorted = xgb_results_filtered.sort_values(by="mean_test_score", ascending=False)
        
        if fhe_timing_sample.size > 0:
            evaluate_fhe_performance(
                xgb_results_sorted, ConcreteXGBClassifier,
                X_train, y_train, fhe_timing_sample,
                xgb_parameter_keys, current_experiment_dir, "XGBoost"
            )
        else:
            logger.warning("Überspringe FHE Performance Evaluierung für XGBoost da kein Timing Sample.")


    # --- Logistic Regression Experiment ---
    if config.get("run_logistic_regression", False):
        logger.info("\n--- Starte Logistic Regression Experiment ---")
        lr_grid_params = config.get("logistic_regression_params", {}).get("grid_search_params", {
            "C": [0.5, 1.0], "n_bits": [10, 14], "solver": ["saga", "lbfgs"], "multi_class": ["auto"]
        })

        start_time_lr_gs = time.time()
        gs_lr_results_obj = perform_grid_search(X_train, y_train, ConcreteLogisticRegression, lr_grid_params)
        logger.info(f"Logistic Regression Grid Search Dauer: {time.time() - start_time_lr_gs:.2f}s")

        lr_gs_df = pd.DataFrame(gs_lr_results_obj.cv_results_)
        lr_gs_df.columns = lr_gs_df.columns.str.replace("param_", "")
        lr_parameter_keys = list(lr_grid_params.keys())
        lr_results_filtered = lr_gs_df[lr_parameter_keys + ["mean_test_score", "std_test_score"]]
        lr_results_sorted = lr_results_filtered.sort_values(by="mean_test_score", ascending=False)

        if fhe_timing_sample.size > 0:
            evaluate_fhe_performance(
                lr_results_sorted, ConcreteLogisticRegression,
                X_train, y_train, fhe_timing_sample,
                lr_parameter_keys, current_experiment_dir, "LogisticRegression"
            )
        else:
            logger.warning("Überspringe FHE Performance Evaluierung für LogReg da kein Timing Sample.")

    logger.info(f"\nExperiment '{experiment_name}' abgeschlossen.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Führt FHE Concrete-ML Experimente durch.")
    parser.add_argument(
        "--config",
        type=str,
        default=None, # Kein Standard-Config-Pfad, um explizite Angabe zu fördern
        help="Pfad zur YAML-Konfigurationsdatei für das Experiment."
    )
    args = parser.parse_args()

    if args.config:
        logger.info(f"Lade Konfiguration von: {args.config}")
        try:
            with open(args.config, 'r') as f:
                main_config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Konfigurationsdatei {args.config} nicht gefunden!")
            exit(1)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Konfigurationsdatei {args.config}: {e}")
            exit(1)
    else:
        # Fallback auf eine Standard-In-Skript-Konfiguration, wenn keine Datei angegeben wird
        logger.warning("Keine Konfigurationsdatei über '--config' angegeben. "
                       "Verwende interne Standardkonfiguration für ein Demo-Experiment.")
        main_config = {
            "dataset_path": DEFAULT_NPZ_PATH, # Verwendet den globalen Standard
            "experiment_name": "FHE_Demo_XGB_LogReg",
            "run_xgboost": True,
            "xgboost_params": {
                "grid_search_params": {
                    "n_bits": [8, 10], # Kleinere Auswahl für Demo
                    "max_depth": [3, 5], 
                    "n_estimators": [3, 5]
                }
            },
            "run_logistic_regression": True,
            "logistic_regression_params": {
                "grid_search_params": {
                    "C": [1.0], 
                    "n_bits": [10], 
                    "solver": ["saga"],
                    "multi_class": ["auto"]
                }
            }
        }

    run_experiment(main_config)