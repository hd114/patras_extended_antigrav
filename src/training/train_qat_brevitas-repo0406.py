#!/usr/bin/env python
# coding: utf-8

"""
Hauptskript für das Quantization-Aware Training (QAT) einer Pipeline
für das EdgeIIoT-Dataset. Dieses Skript handhabt das Laden von Daten,
Modellinitialisierung, Training, Evaluierung (PyTorch und FHE-simuliert),
und das Logging von Ergebnissen und Artefakten, inklusive einer detaillierten
Text-Logdatei.
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import RocCurveDisplay, f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import brevitas.nn as qnn
from concrete.ml.torch.compile import compile_brevitas_qat_model

# Eigene Modul-Importe
from src.data.edge_iiot_dataset import load_edgeiiot_data
from src.evaluation.concrete_evaluate import (
    _calculate_robust_roc_auc,
    evaluate_fhe_model,
    evaluate_torch_model,
)
from src.models.qat_model import QATPrunedSimpleNet, get_pruning_summary
from src.training.custom_losses import FocalLoss
from src.utils.config_loader import load_config
from src.utils.logger import (
    log_training_epoch,
    log_training_summary,
    setup_logger,
)


# --- Hilfsfunktionen für Trainings-Epochen ---
def _train_epoch_step(
    model: nn.Module,
    data_loader_or_tensor: Union[DataLoader, torch.Tensor],
    labels_tensor_if_manual: Optional[torch.Tensor],
    batch_size_if_manual: Optional[int],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int,
    total_epochs: int,
    logger: logging.Logger,
    use_loader: bool,
) -> float:
    """
    Führt einen einzelnen Trainingsschritt (Epoche) für das gegebene Modell durch.

    Args:
        model: Das zu trainierende PyTorch-Modell.
        data_loader_or_tensor: Entweder ein DataLoader für Batch-Verarbeitung
                               oder ein Tensor mit allen Trainingsdaten.
        labels_tensor_if_manual: Tensor mit Labels, falls Trainingsdaten
                                 manuell als Tensor übergeben werden.
        batch_size_if_manual: Batch-Größe für manuelles Slicing der Tensoren.
        criterion: Die Verlustfunktion.
        optimizer: Der Optimierer.
        device: Das Gerät (CPU/GPU), auf dem trainiert wird.
        epoch_num: Die aktuelle Epochennummer (0-basiert).
        total_epochs: Die Gesamtanzahl der Epochen.
        logger: Der Logger für Trainingsinformationen.
        use_loader: True, wenn ein DataLoader verwendet wird, sonst False.

    Returns:
        Der durchschnittliche Trainingsverlust für diese Epoche.
    """
    model.train()
    running_loss = 0.0
    num_batches_processed = 0

    if use_loader:
        data_iterator = data_loader_or_tensor
        num_total_batches = len(data_loader_or_tensor)  # type: ignore
        iterable_for_tqdm = enumerate(data_iterator)
    else:
        X_train_tensor = data_loader_or_tensor
        if batch_size_if_manual and batch_size_if_manual > 0:
            num_total_batches = (
                len(X_train_tensor) + batch_size_if_manual - 1  # type: ignore
            ) // batch_size_if_manual
        else:
            if logger:
                logger.error(
                    "Ungültige batch_size_if_manual für manuelles Slicing. "
                    "Setze Batches auf 0."
                )
            num_total_batches = 0
        iterable_for_tqdm = range(num_total_batches)

    if num_total_batches == 0:
        log_msg = "Keine Batches zum Trainieren in _train_epoch_step."
        if logger:
            logger.warning(log_msg)
        else:
            print(f"WARNUNG: {log_msg}")
        return 0.0

    progress_bar = tqdm(
        iterable_for_tqdm,
        total=num_total_batches,
        desc=f"Epoch {epoch_num + 1}/{total_epochs} [T]",
        leave=False,
    )

    if use_loader:
        for i, batch_data in progress_bar:
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            num_batches_processed += 1
    else:
        y_train_tensor = labels_tensor_if_manual
        if y_train_tensor is None:
            if logger:
                logger.error(
                    "labels_tensor_if_manual ist None bei manuellem Slicing."
                )
            return 0.0
        # Sicherstellen, dass batch_size_if_manual ein gültiger int > 0 ist
        if not isinstance(batch_size_if_manual, int) or batch_size_if_manual <= 0:
            if logger:
                logger.error(
                    f"Ungültiger batch_size_if_manual ({batch_size_if_manual}) "
                    "für manuelles Slicing in der Schleife."
                )
            return 0.0

        for i in progress_bar:
            start_idx = i * batch_size_if_manual
            end_idx = start_idx + batch_size_if_manual
            inputs = X_train_tensor[start_idx:end_idx]  # type: ignore
            labels = y_train_tensor[start_idx:end_idx]
            if inputs.shape[0] == 0:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            num_batches_processed += 1

    if num_batches_processed > 0:
        return running_loss / num_batches_processed
    else:
        if logger:
            logger.warning("Keine Batches in der Epoche verarbeitet.")
        return 0.0


def _validate_epoch_step(
    model: nn.Module,
    X_val_tensor: torch.Tensor,
    y_val_tensor: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    label_encoder_global: LabelEncoder,
    epoch_num: int,
    logger: logging.Logger,
) -> Tuple[float, float, float, Union[str, float], np.ndarray, np.ndarray]:
    """
    Führt einen einzelnen Validierungsschritt (Epoche) für das gegebene Modell durch.

    Args:
        model: Das zu validierende PyTorch-Modell.
        X_val_tensor: Tensor mit Validierungsmerkmalen.
        y_val_tensor: Tensor mit Validierungslabels.
        criterion: Die Verlustfunktion.
        device: Das Gerät (CPU/GPU), auf dem validiert wird.
        label_encoder_global: Der globale LabelEncoder für Klasseninformationen.
        epoch_num: Die aktuelle Epochennummer (0-basiert).
        logger: Der Logger für Validierungsinformationen.

    Returns:
        Ein Tupel mit:
        - Durchschnittlicher Validierungsverlust.
        - Gewichteter F1-Score.
        - Makro F1-Score.
        - Makro ROC AUC Score (oder Fehlermeldung als String).
        - Array der wahren Labels.
        - Array der vorhergesagten Wahrscheinlichkeiten.
    """
    model.eval()
    current_val_loss = 0.0
    f1_weighted, f1_macro = 0.0, 0.0
    true_labels_val_list, probs_val_list = [], []
    val_batch_size = 512  # Feste Batch-Größe für interne Validierungs-DataLoader

    val_dataset_for_loader = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader_internal = DataLoader(
        val_dataset_for_loader, batch_size=val_batch_size, shuffle=False
    )

    running_val_loss_agg = 0.0
    total_val_samples = 0

    with torch.no_grad():
        for X_batch_val, y_batch_val in val_loader_internal:
            X_batch_val_dev = X_batch_val.to(device)
            y_batch_val_dev = y_batch_val.to(device)

            val_outputs_batch = model(X_batch_val_dev)
            loss_val_item = criterion(val_outputs_batch, y_batch_val_dev).item()
            running_val_loss_agg += loss_val_item * X_batch_val.size(0)
            total_val_samples += X_batch_val.size(0)

            softmax_probs_batch = torch.softmax(val_outputs_batch, dim=1)
            true_labels_val_list.append(y_batch_val.cpu().numpy())
            probs_val_list.append(softmax_probs_batch.cpu().numpy())

    if total_val_samples > 0:
        current_val_loss = running_val_loss_agg / total_val_samples
    else:
        current_val_loss = 0.0

    roc_auc_macro_val: Union[str, float] = "n/a (no_val_samples)"

    if total_val_samples > 0:
        true_labels_val = np.concatenate(true_labels_val_list).astype(int)
        probs_val = np.concatenate(probs_val_list, axis=0)
        preds_val = np.argmax(probs_val, axis=1).astype(int)
        metric_labels_val = np.unique(np.concatenate((true_labels_val, preds_val)))

        if len(metric_labels_val) == 0:
            f1_weighted, f1_macro = 0.0, 0.0
            log_msg_val = "Keine Labels für Metrikberechnung in Validierung."
            if logger:
                logger.warning(log_msg_val)
            else:
                print(f"WARNUNG: {log_msg_val}")
        else:
            f1_weighted = f1_score(
                true_labels_val,
                preds_val,
                labels=metric_labels_val,
                average="weighted",
                zero_division=0,
            )
            f1_macro = f1_score(
                true_labels_val,
                preds_val,
                labels=metric_labels_val,
                average="macro",
                zero_division=0,
            )
            roc_auc_macro_val = _calculate_robust_roc_auc(
                true_labels_val,
                probs_val,
                logger,
                context_msg=f"Ep{epoch_num + 1} Val",
            )
    else:
        true_labels_val, probs_val = np.array([]), np.array([])

    return (
        current_val_loss,
        f1_weighted,
        f1_macro,
        roc_auc_macro_val,
        true_labels_val,
        probs_val,
    )


# --- Haupt-Trainings- und Kompilierungsfunktion ---
def run_qat_training_pipeline(config_path: str = "config.yaml") -> Optional[str]:
    """
    Führt die gesamte Quantization-Aware Training (QAT) Pipeline aus.

    Dies beinhaltet das Laden der Konfiguration, Datenvorbereitung,
    Modellinitialisierung, Training, Speichern von Artefakten,
    Evaluierung und detailliertes Logging.

    Args:
        config_path: Pfad zur YAML-Konfigurationsdatei.

    Returns:
        Der Pfad zum Ergebnisverzeichnis des Laufs oder None bei einem Fehler.
    """
    # --- 1. Konfiguration und initiales Setup ---
    config = load_config(config_path)
    if not config:
        # Fallback-Logging, falls der Hauptlogger noch nicht initialisiert werden kann
        print(f"FEHLER: Konnte Config von '{config_path}' nicht laden. Pipeline-Abbruch.")
        return None

    run_cfg = config.get("run_settings", {})
    data_cfg = config.get("data_params", {})
    model_cfg = config.get("model_params", {})
    train_cfg = config.get("training_params", {})
    eval_cfg = config.get("evaluation_params", {})

    log_level_str = run_cfg.get("logger_level", "INFO").upper()
    # Verwende den logger_instance_name aus der Config oder einen Standardwert
    logger_name_from_cfg = run_cfg.get("logger_instance_name", "qat_pipeline_run") 
    log = setup_logger(name=logger_name_from_cfg, level=log_level_str)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Zeitstempel für Dateinamen relevant
    dataset_name = run_cfg.get("dataset_name", "UnknownDataset")

    # Basis-Identifikator für DATEINAMEN (ohne Laufzeit-Metriken wie Epochen/F1)
    # Diese Parameter kommen aus der spezifischen Konfiguration dieses Laufs.
    cfg_h_early = model_cfg.get("n_hidden", 100) 
    cfg_qb_early = model_cfg.get("quantization_bits", 3)
    cfg_up_early = model_cfg.get("unpruned_neurons", 16)

    base_identifier_for_filenames = (
        f"QAT_{dataset_name}_{cfg_h_early}h_qb{cfg_qb_early}_p{cfg_up_early}"
    )
    
    # WICHTIG: Der Ergebnisordner wird jetzt direkt aus der Konfiguration übernommen.
    # Der `experiment_runner.py` muss sicherstellen, dass dieser Pfad für jeden
    # einzelnen Lauf (jede Wiederholung jeder Konfiguration) eindeutig ist.
    current_run_results_dir = run_cfg.get("results_base_dir") 
    
    if not current_run_results_dir:
        log.error("FEHLER: 'results_base_dir' nicht in run_settings der Config gefunden oder ist leer. Pipeline-Abbruch.")
        # Hier sollten ggf. bereits geöffnete Handler geschlossen werden, falls der Logger schon welche hat.
        # Da detailed_log_file_handler noch nicht erstellt wurde, ist das hier noch nicht nötig.
        return None 
        
    # Erstelle den (jetzt eindeutigen) Ergebnisordner, falls er nicht existiert.
    # Dieser Ordner wird vom experiment_runner.py so gesetzt, dass er eindeutig ist.
    try:
        os.makedirs(current_run_results_dir, exist_ok=True)
    except OSError as e:
        log.error(f"FEHLER beim Erstellen des Ergebnisordners '{current_run_results_dir}': {e}. Pipeline-Abbruch.")
        return None

    # --- Detaillierten Text-Log-Handler einrichten ---
    # Der Dateiname der Text-Logdatei verwendet den Zeitstempel und den Basis-Identifikator (ohne Laufzeit-Metriken).
    # Er landet direkt im `current_run_results_dir`.
    detailed_log_filename = f"{run_timestamp}_{base_identifier_for_filenames}_pipeline_full.log"
    detailed_log_path = os.path.join(current_run_results_dir, detailed_log_filename)
    
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S,%03d",
    )

    # Entferne existierende FileHandler für diesen Logger, um Duplikate zu vermeiden,
    # falls die Funktion im selben Python-Prozess mehrfach aufgerufen wird und der Logger persistiert.
    # Dies ist besonders wichtig, wenn der experiment_runner die Pipeline-Funktion mehrfach aufruft.
    for handler in list(log.handlers): # Iteriere über eine Kopie der Handler-Liste
        if isinstance(handler, logging.FileHandler):
            # Schließe und entferne alle FileHandler, um sicherzustellen, dass wir nicht in alte Dateien schreiben
            # oder mehrere Handler für dieselbe Datei haben.
            # Eine spezifischere Prüfung wäre, ob handler.baseFilename == detailed_log_path,
            # aber es ist sicherer, alle FileHandler zu entfernen und neu hinzuzufügen.
            log.removeHandler(handler)
            handler.close()
            
    detailed_log_file_handler = logging.FileHandler(detailed_log_path, mode="w") # "w" um bei jedem Lauf neu zu schreiben
    detailed_log_file_handler.setFormatter(file_formatter)
    detailed_log_file_handler.setLevel(log.level)
    log.addHandler(detailed_log_file_handler)

    log.info(f"Detailliertes Text-Log wird nach '{detailed_log_path}' geschrieben.")
    log.info(f"Alle Ergebnisse für diesen spezifischen Run werden in '{current_run_results_dir}' gespeichert.")

    # Gerätekonfiguration
    device_str = run_cfg.get("device_str", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info(f"Verwende Device: {device}")


    # --- 2. Daten laden und vorbereiten ---
    npz_file_path = data_cfg.get("npz_file_path")
    if not npz_file_path:
        log.error("FEHLER: 'npz_file_path' nicht in data_params der Config. Abbruch.")
        if detailed_log_file_handler:
            log.removeHandler(detailed_log_file_handler)
            detailed_log_file_handler.close()
        return None

    log.info(f"Lade Daten für Dataset: {dataset_name} von {npz_file_path}")
    dataloader_batch_size = data_cfg.get("dataloader_batch_size", 256)

    # Lade Daten mit der Hilfsfunktion
    # Explicitly type raw_data if possible, or handle potential None carefully
    loaded_data = load_edgeiiot_data(
        npz_path=npz_file_path,
        batch_size=dataloader_batch_size,
        refit_scaler=data_cfg.get("refit_scaler", False),
        refit_encoder=data_cfg.get("refit_encoder", False),
        return_raw_data=True,
        logger=log,
    )
    if loaded_data is None or loaded_data[3] is None: # loaded_data[3] is raw_data
        log.error("Fehler beim Laden der Daten. Abbruch der Pipeline.")
        if detailed_log_file_handler:
            log.removeHandler(detailed_log_file_handler)
            detailed_log_file_handler.close()
        return None
    train_loader, val_loader, test_loader, raw_data = loaded_data

    input_size = raw_data["num_features"]
    num_classes = raw_data["num_classes"]
    label_encoder = raw_data["label_encoder"]
    X_train_np_all, y_train_np_all = raw_data["X_train"], raw_data["y_train"]

    # Optionales Subsetting der Trainingsdaten
    if data_cfg.get("use_subset_training", False):
        subset_fraction = data_cfg.get("subset_fraction", 0.1)
        if not (0 < subset_fraction <= 1.0):
            log.warning(
                f"Ungültige subset_fraction ({subset_fraction}). Verwende 0.1."
            )
            subset_fraction = 0.1
        subset_size = int(len(X_train_np_all) * subset_fraction)
        X_train_np, y_train_np = (
            X_train_np_all[:subset_size],
            y_train_np_all[:subset_size],
        )
        log.info(
            f"Verwende Subset von {len(X_train_np)} Trainingssamples "
            f"({subset_fraction*100:.1f}%)."
        )
    else:
        X_train_np, y_train_np = X_train_np_all, y_train_np_all
        log.info(
            f"Verwende vollen Trainingsdatensatz ({len(X_train_np)} Samples)."
        )

    # --- 3. Modellinitialisierung und -konfiguration ---
    # Modellparameter aus der Konfiguration lesen (cfg_..._early bereits definiert)
    dropout_config = model_cfg.get("dropout", {})
    dropout_rate1 = dropout_config.get("rate1", 0.0)
    dropout_rate2 = dropout_config.get("rate2", 0.0)
    log.info(
        f"Modellparameter (aus Config): n_hidden={cfg_h_early}, "
        f"qbits={cfg_qb_early}, unpruned={cfg_up_early}, "
        f"dropout1={dropout_rate1}, dropout2={dropout_rate2}"
    )

    qlinear_args = {
        "weight_bit_width": cfg_qb_early,
        "bias": model_cfg.get("qlinear_bias", True),
    }
    qidentity_args = {
        "bit_width": cfg_qb_early,
        "return_quant_tensor": model_cfg.get(
            "qidentity_return_quant_tensor", True
        ),
    }
    qrelu_args = {
        "bit_width": cfg_qb_early,
        "return_quant_tensor": model_cfg.get("qrelu_return_quant_tensor", True),
    }

    torch_model = QATPrunedSimpleNet(
        input_size,
        num_classes,
        cfg_h_early, # Verwende die früh definierte Variable
        qlinear_args_config=qlinear_args,
        qidentity_args_config=qidentity_args,
        qrelu_args_config=qrelu_args,
        dropout_rate1=dropout_rate1,
        dropout_rate2=dropout_rate2,
    ).to(device)

    # Optionales Pruning des Modells
    if cfg_up_early > 0 and input_size > cfg_up_early:
        torch_model.prune(max_non_zero_per_neuron=cfg_up_early)
    initial_pruning_summary = get_pruning_summary(torch_model)
    log.info(f"Initiales Pruning Summary: {initial_pruning_summary}")

    # Extraktion der Layer-Struktur für das Logging
    layer_structure = []
    for name, m in torch_model.named_modules():
        is_relevant = False
        layer_type = type(m).__name__
        in_f, out_f, has_b = None, None, None
        if isinstance(m, qnn.QuantLinear):
            is_relevant = True
        elif isinstance(m, torch.nn.Linear):
            is_relevant = True
            layer_type = "torch.nn.Linear" # Für Klarheit im Log
        elif hasattr(torch.nn, "qat") and isinstance(m, torch.nn.qat.Linear):
            is_relevant = True
            layer_type = "torch.nn.qat.Linear" # Für Klarheit im Log

        if is_relevant:
            if hasattr(m, "in_features"):
                in_f = m.in_features
            if hasattr(m, "out_features"):
                out_f = m.out_features
            if hasattr(m, "bias"):
                has_b = m.bias is not None
            layer_structure.append(
                {"name": name, "type": layer_type, "in": in_f, "out": out_f, "bias": has_b}
            )
    log.info(f"Modell Layer Struktur: {layer_structure}")

    # --- 4. Trainingsparameter und Optimierungskomponenten ---
    learning_rate = train_cfg.get("learning_rate", 0.001)
    num_epochs = train_cfg.get("num_epochs", 10)
    early_stopping_cfg = train_cfg.get("early_stopping", {})
    patience = early_stopping_cfg.get("patience", 15)
    min_delta = early_stopping_cfg.get("min_delta", 1e-5)
    use_train_loader = train_cfg.get("use_train_loader_for_batches", False)
    manual_batch_size = train_cfg.get("manual_batch_size", 256)

    # Konfiguration der Verlustfunktion (Criterion)
    criterion_cfg = train_cfg.get("criterion", {})
    criterion_name = criterion_cfg.get("name", "CrossEntropyLoss").lower()
    log.debug(f"DEBUG: Konfigurierter Criterion-Name: '{criterion_cfg.get('name')}'")

    # Handhabung von Klassengewichten
    weights_tensor = None
    class_weights_list = criterion_cfg.get("class_weights")
    if class_weights_list is not None and isinstance(class_weights_list, list):
        try:
            if len(class_weights_list) == num_classes:
                weights_tensor = torch.tensor(
                    class_weights_list, dtype=torch.float32
                ).to(device)
                log.info(
                    "Verwende manuell konfigurierte Klassengewichte: "
                    f"{weights_tensor.tolist()}"
                )
            else:
                log.warning(
                    f"Länge der manuellen class_weights ({len(class_weights_list)}) "
                    f"entspricht nicht num_classes ({num_classes}). "
                    "Ignoriere manuelle Gewichte."
                )
        except Exception as e_cw:
            log.warning(
                "Konnte manuell konfigurierte class_weights nicht als Tensor "
                f"konvertieren: {e_cw}. Fahre ohne manuelle Gewichte fort."
            )
            weights_tensor = None

    if weights_tensor is None and criterion_cfg.get("calculate_class_weights", False):
        method = criterion_cfg.get(
            "class_weight_calculation_method", "inverse_frequency"
        )
        log.info(f"Berechne Klassengewichte (Methode: {method})...")
        if y_train_np is not None and len(y_train_np) > 0:
            if num_classes > 0:
                counts = np.bincount(y_train_np.astype(int), minlength=num_classes)
                calc_weights_list = []
                total_samples_for_w = len(y_train_np)
                if method == "effective_number":
                    beta = criterion_cfg.get("class_weight_beta", 0.999)
                    if not (0 <= beta < 1.0):
                        log.warning(f"Ungültiger Beta-Wert ({beta}). Verwende 0.999.")
                        beta = 0.999
                    for i in range(num_classes):
                        count = counts[i] if i < len(counts) else 0
                        if count > 0:
                            eff_num = (1.0 - beta**count) / (1.0 - beta)
                        else: # Vermeide beta**0 wenn beta=1
                            eff_num = (1.0 - beta**1) / (1.0 - beta) if beta < 1.0 else 1.0
                        calc_weights_list.append(1.0 / eff_num if eff_num > 1e-9 else 1.0)
                    temp_w = torch.tensor(calc_weights_list, dtype=torch.float32)
                    if temp_w.sum() > 1e-9:
                        weights_tensor = (temp_w / temp_w.sum() * num_classes).to(device)
                    else:
                        weights_tensor = torch.ones(num_classes, dtype=torch.float32).to(device)
                        log.warning(
                            "Summe der 'Effective Number' Gewichte war 0. "
                            "Verwende neutrale Gewichte."
                            )
                    log.info(
                        "Automatisch berechnete 'Effective Number' Klassengewichte "
                        f"(beta={beta}, normalisiert): {weights_tensor.tolist()}"
                    )
                elif method == "inverse_frequency":
                    for i in range(num_classes):
                        count = counts[i] if i < len(counts) else 0
                        calc_weights_list.append(
                            total_samples_for_w / (num_classes * count) if count > 0 else 1.0
                        )
                    weights_tensor = torch.tensor(
                        calc_weights_list, dtype=torch.float32
                    ).to(device)
                    log.info(
                        "Automatisch berechnete 'Inverse Frequency' Klassengewichte: "
                        f"{weights_tensor.tolist()}"
                    )
                else:
                    log.warning(
                        f"Unbekannte class_weight_calculation_method: {method}. "
                        "Keine Gewichte berechnet."
                    )
            else:
                log.error(
                    "num_classes ist <= 0, kann Klassengewichte nicht berechnen."
                )
        else:
            log.warning(
                "Keine Trainingslabels (y_train_np) zur Berechnung der "
                "Klassengewichte verfügbar."
            )

    if weights_tensor is not None and criterion_cfg.get("calculate_class_weights", False):
        scale = criterion_cfg.get("auto_weight_overall_scale", 1.0)
        if scale != 1.0:
            weights_tensor *= scale
            log.info(
                f"Automatische Klassengewichte skaliert mit Faktor {scale}: "
                f"{weights_tensor.tolist()}"
            )
    if weights_tensor is None:
        log.info("Keine Klassengewichte werden für die Verlustfunktion verwendet.")

    reduction = criterion_cfg.get("reduction", "mean")
    if criterion_name == "focalloss":
        alpha = criterion_cfg.get("focal_loss_alpha", 0.25)
        gamma = criterion_cfg.get("focal_loss_gamma", 2.0)
        try:
            criterion = FocalLoss(
                alpha=alpha, gamma=gamma, weight=weights_tensor, reduction=reduction
            )
            log.info(
                f"Verwende FocalLoss (alpha={alpha}, gamma={gamma}, "
                f"reduction='{reduction}', mit Gewichten: {weights_tensor is not None})"
            )
        except Exception as e_crit:
            log.error(
                f"FEHLER bei Instanziierung von FocalLoss: {e_crit}. "
                "Fallback zu CrossEntropyLoss."
            )
            criterion = nn.CrossEntropyLoss(weight=weights_tensor, reduction=reduction)
            log.info(
                "Verwende CrossEntropyLoss als Fallback "
                f"(reduction='{reduction}', mit Gewichten: {weights_tensor is not None})."
            )
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor, reduction=reduction)
        log.info(
            f"Verwende CrossEntropyLoss (Config war: "
            f"'{criterion_cfg.get('name', 'CrossEntropyLoss')}'), "
            f"reduction='{reduction}', mit Gewichten: {weights_tensor is not None}"
        )

    # Konfiguration des Optimierers
    optimizer_cfg = train_cfg.get("optimizer", {})
    optimizer_name = optimizer_cfg.get("name", "AdamW").lower()
    opt_weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    opt_momentum = optimizer_cfg.get("momentum", 0.9)  # Für SGD relevant

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            torch_model.parameters(), lr=learning_rate, weight_decay=opt_weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            torch_model.parameters(), lr=learning_rate, weight_decay=opt_weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            torch_model.parameters(),
            lr=learning_rate,
            momentum=opt_momentum,
            weight_decay=opt_weight_decay,
        )
    else:
        log.warning(f"Unbekannter Optimizer '{optimizer_name}'. Verwende AdamW.")
        optimizer = optim.AdamW(
            torch_model.parameters(), lr=learning_rate, weight_decay=opt_weight_decay
        )
    log.info(
        f"Verwende Optimizer: {type(optimizer).__name__} "
        f"mit initialer lr={learning_rate}"
    )

    # Konfiguration des Lernraten-Schedulers
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_name_from_cfg = scheduler_cfg.get("name")
    scheduler = None
    if (
        scheduler_name_from_cfg
        and isinstance(scheduler_name_from_cfg, str)
        and scheduler_name_from_cfg.lower() not in ["null", "none", ""]
    ):
        s_name_lower = scheduler_name_from_cfg.lower()
        log.debug(f"DEBUG: Verarbeite Scheduler-Namen: '{s_name_lower}'")
        # Parameter für Scheduler sollten direkt unter scheduler_cfg stehen,
        # nicht in einem weiteren "params" Sub-Dict wie im alten Code.
        if s_name_lower == "reducelronplateau":
            params = {
                "factor": scheduler_cfg.get("reduce_lr_factor", 0.1),
                "patience": scheduler_cfg.get("reduce_lr_patience", 10),
                "min_lr": scheduler_cfg.get("reduce_lr_min_lr", 1e-6),
                #"verbose": scheduler_cfg.get("reduce_lr_verbose", True),
            }
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", **params
            )
            log.info(
                f"Verwende ReduceLROnPlateau Scheduler mit Parametern: {params}"
            )
        elif s_name_lower == "steplr":
            params = {
                "step_size": scheduler_cfg.get("step_lr_step_size", 10), # Angepasst an neue Config-Struktur
                "gamma": scheduler_cfg.get("step_lr_gamma", 0.1),       # Angepasst
            }
            scheduler = optim.lr_scheduler.StepLR(optimizer, **params)
            log.info(f"Verwende StepLR Scheduler mit Parametern: {params}")
        else:
            log.warning(
                f"Unbekannter Scheduler-Name '{scheduler_name_from_cfg}'. "
                "Kein Scheduler verwendet."
            )
    else:
        log.info("Kein Scheduler konfiguriert.")

    # --- 5. Trainingsschleife ---
    train_losses_hist, val_losses_hist, lr_history_hist = [], [], []
    best_val_loss_for_early_stop = float("inf")
    patience_counter_early_stop = 0
    best_f1_weighted_overall = 0.0
    best_model_state_dict = None
    best_true_val_for_roc, best_probs_val_for_roc = None, None
    best_f1_macro_at_best_f1_weighted: Union[str, float] = "N/A"
    best_roc_auc_at_best_f1_weighted: Union[str, float] = "N/A"
    epoch_of_best_f1w = -1

    # Daten in Tensoren umwandeln und auf das Gerät verschieben
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(raw_data["X_val"], dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(raw_data["y_val"], dtype=torch.long).to(device)

    log.info(f"Starte Training für {num_epochs} Epochen...")
    actual_epochs_run = 0
    for epoch_idx in range(num_epochs):
        actual_epochs_run = epoch_idx + 1
        epoch_start_time = time.time()
        current_lr_for_epoch = optimizer.param_groups[0]["lr"]
        lr_history_hist.append(current_lr_for_epoch)

        # Trainingsschritt
        train_data_input = train_loader if use_train_loader else X_train_tensor
        train_labels_input = y_train_tensor if not use_train_loader else None
        current_manual_batch_size_for_step = manual_batch_size if not use_train_loader else None

        avg_train_loss_epoch = _train_epoch_step(
            torch_model,
            train_data_input,
            train_labels_input,
            current_manual_batch_size_for_step,
            criterion,
            optimizer,
            device,
            epoch_idx,
            num_epochs,
            log,
            use_loader=use_train_loader,
        )
        train_losses_hist.append(avg_train_loss_epoch)

        # Validierungsschritt
        (
            avg_val_loss_epoch,
            f1_w_epoch,
            f1_m_epoch,
            roc_auc_m_epoch,
            true_val_labels,
            probs_val,
        ) = _validate_epoch_step(
            torch_model,
            X_val_tensor,
            y_val_tensor,
            criterion,
            device,
            label_encoder,
            epoch_idx,
            log,
        )
        val_losses_hist.append(avg_val_loss_epoch)

        log_training_epoch(
            log,
            epoch_idx,
            avg_train_loss_epoch,
            avg_val_loss_epoch,
            f1_w_epoch,
            f1_m_epoch,
            roc_auc_m_epoch, # roc_auc_macro_val umbenannt zu roc_auc_m_epoch
            epoch_start_time,
        )

        # Scheduler-Schritt
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss_epoch)
            else:
                scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]
            is_plateau_verbose = isinstance(
                scheduler, optim.lr_scheduler.ReduceLROnPlateau
            ) and getattr(scheduler, "verbose", False)
            if new_lr != current_lr_for_epoch and not is_plateau_verbose:
                log.info(
                    f"LR durch Scheduler in Epoche {epoch_idx + 1} "
                    f"angepasst auf: {new_lr:.7f}"
                )
        # Bestes Modell basierend auf F1-Weighted speichern
        if f1_w_epoch > best_f1_weighted_overall:
            best_f1_weighted_overall = f1_w_epoch
            best_model_state_dict = torch_model.state_dict()
            best_true_val_for_roc, best_probs_val_for_roc = true_val_labels, probs_val
            best_f1_macro_at_best_f1_weighted = f1_m_epoch
            best_roc_auc_at_best_f1_weighted = roc_auc_m_epoch
            epoch_of_best_f1w = epoch_idx
            log.info(
                f"Neues bestes Modell in Epoche {epoch_idx + 1} gefunden "
                f"(F1-Weighted: {f1_w_epoch:.4f}, Val Loss: {avg_val_loss_epoch:.4f})"
            )

        # Early Stopping Logik
        if avg_val_loss_epoch < best_val_loss_for_early_stop - min_delta:
            best_val_loss_for_early_stop = avg_val_loss_epoch
            patience_counter_early_stop = 0
        else:
            patience_counter_early_stop += 1
            log.debug(
                f"EarlyStopping: Val loss nicht verbessert. "
                f"Patience: {patience_counter_early_stop}/{patience}"
            )
            if patience_counter_early_stop >= patience:
                log.info(f"Early Stopping ausgelöst in Epoche {epoch_idx + 1}.")
                break
    # Ende der Trainingsschleife

    final_pruning_summary = get_pruning_summary(torch_model)
    log.info(f"Finales Pruning Summary: {final_pruning_summary}")

    
    # --- 6. Artefakt-Management (Modell speichern, Plots, Logs) ---
    # artifact_file_prefix wird NACH dem Training erstellt und enthält Laufzeit-Metriken.
    # Dieser Präfix wird für die DATEINAMEN der Artefakte verwendet.
    artifact_file_prefix: str 
    if best_model_state_dict: # best_model_state_dict wird in der Trainingsschleife gesetzt
        f1_val_for_name = 0.0
        # Robuste Umwandlung des F1-Scores für den Dateinamen
        if isinstance(best_f1_weighted_overall, (float, int)):
            f1_val_for_name = float(best_f1_weighted_overall)
        elif isinstance(best_f1_weighted_overall, str):
            # Versuche, String in Float zu konvertieren, wenn es keine Fehler-Zeichenkette ist
            # Erweitere die Liste der Fehler-Strings bei Bedarf
            error_strings_for_f1 = ["N/A", "n/a", "n/a (init)", "n/a (no_val_samples)", 
                                    "n/a (FewClassesInSubset)", 
                                    "n/a (LabelOutOfBoundsForProbs)", 
                                    "n/a (SklearnValErr)", "n/a (OtherErr)"]
            if not any(err_str in best_f1_weighted_overall for err_str in error_strings_for_f1):
                try:
                    f1_val_for_name = float(best_f1_weighted_overall)
                except ValueError:
                    log.warning(f"Konnte best_f1_weighted_overall ('{best_f1_weighted_overall}') nicht in float umwandeln für Dateinamen.")
            else:
                log.info(f"best_f1_weighted_overall ('{best_f1_weighted_overall}') ist ein Fehlerstring, verwende 0.0 für Dateinamen.")
        
        f1_str = f"{f1_val_for_name:.4f}".replace("0.", "").replace(".", "")
        # Stelle sicher, dass base_identifier_for_filenames hier korrekt verwendet wird
        artifact_file_prefix = ( 
            f"{base_identifier_for_filenames}_ep{actual_epochs_run}_f1w{f1_str}"
        )
    else:
        artifact_file_prefix = (
            f"{base_identifier_for_filenames}_ep{actual_epochs_run}_no_best_model"
        )
    
    # Alle Artefakte werden direkt in current_run_results_dir gespeichert.
    # Die Dateinamen enthalten den artifact_file_prefix.
    model_path_final, loss_plot_path_final, roc_plot_path_final = "n/a", "n/a", "n/a"

    if best_model_state_dict:
        # Modell speichern
        model_path_final = os.path.join(
            current_run_results_dir, f"{artifact_file_prefix}.pth" # Dateiname mit Metriken
        )
        try:
            torch.save(best_model_state_dict, model_path_final)
            log.info(f"Bestes Modell gespeichert: {model_path_final}")
        except Exception as e_save_model:
            log.error(f"Fehler beim Speichern des Modells: {e_save_model}\n{traceback.format_exc()}")
            model_path_final = "n/a (SaveError)"


        # Loss-Plot speichern
        loss_plot_path_final = os.path.join(
            current_run_results_dir, f"{artifact_file_prefix}_loss.png" # Dateiname mit Metriken
        )
        try:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(
                range(1, actual_epochs_run + 1),
                train_losses_hist,
                label="Train Loss",
                marker=".",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Trainingsverlauf")
            plt.grid(True)
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(
                range(1, actual_epochs_run + 1),
                val_losses_hist,
                label="Val Loss",
                marker=".",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Validierungsverlauf")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(loss_plot_path_final)
            plt.close() # Schließe die Figur, um Speicher freizugeben
            log.info(f"Loss-Plot gespeichert: {loss_plot_path_final}")
        except Exception as e_plot_loss:
            log.error(f"Fehler beim Erstellen/Speichern des Loss-Plots: {e_plot_loss}\n{traceback.format_exc()}")
            loss_plot_path_final = "n/a (PlotError)"

        # ROC-Plot speichern
        roc_plot_path_final = "n/a (init_roc_plot)" # Initialwert
        can_plot_roc = (
            best_true_val_for_roc is not None
            and best_probs_val_for_roc is not None
            and isinstance(best_true_val_for_roc, np.ndarray) # Zusätzliche Typprüfung
            and isinstance(best_probs_val_for_roc, np.ndarray) # Zusätzliche Typprüfung
            and len(np.unique(best_true_val_for_roc)) >= 2
            and best_probs_val_for_roc.shape[0] > 0
            and best_probs_val_for_roc.ndim == 2 # Sicherstellen, dass es 2D ist
            and best_probs_val_for_roc.shape[1] >= 2
        )
        if can_plot_roc:
            try:
                roc_plot_path_final = os.path.join(
                    current_run_results_dir, f"{artifact_file_prefix}_roc_curves.png"
                )
                fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
                num_plot_classes_roc = best_probs_val_for_roc.shape[1]
                plot_class_names_roc = [
                    f"Class {i}" for i in range(num_plot_classes_roc)
                ]
                if (
                    hasattr(label_encoder, "classes_")
                    and label_encoder.classes_ is not None
                    and len(label_encoder.classes_) == num_plot_classes_roc
                ):
                    plot_class_names_roc = [str(cn) for cn in label_encoder.classes_]
                else:
                    log.warning(
                        f"Klassennamen vom LabelEncoder nicht für ROC-Plot verwendbar "
                        f"(erwartet {num_plot_classes_roc}, hat "
                        f"{len(getattr(label_encoder, 'classes_', [])) }). "
                        "Generische Namen."
                    )

                for i in range(num_plot_classes_roc):
                    y_true_ovr_roc = (best_true_val_for_roc == i).astype(int)
                    if len(np.unique(y_true_ovr_roc)) < 2:
                        log.info(
                            f"ROC für Klasse '{str(plot_class_names_roc[i])}' (Idx {i}) "
                            "nicht geplottet: Benötigt pos/neg Samples in Val-Daten."
                        )
                        continue
                    RocCurveDisplay.from_predictions(
                        y_true_ovr_roc,
                        best_probs_val_for_roc[:, i],
                        name=f"ROC Klasse '{str(plot_class_names_roc[i])}' (idx {i})",
                        ax=ax_roc,
                    )
                ax_roc.plot([0, 1], [0, 1], "k--", label="Chance (AUC=0.5)")
                ax_roc.axis("square")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title(
                    f"ROC Curves (Best Model F1-w: {best_f1_weighted_overall:.4f})"
                )
                ax_roc.legend(loc="lower right", fontsize="small")
                fig_roc.tight_layout()
                fig_roc.savefig(roc_plot_path_final)
                plt.close(fig_roc) # Schließe die Figur, um Speicher freizugeben
                log.info(f"ROC-Plot gespeichert: {roc_plot_path_final}")
            except Exception as e_roc_plot:
                log.error(
                    f"ROC Plot Erstellung fehlgeschlagen: {e_roc_plot}\n"
                    f"{traceback.format_exc()}"
                )
                roc_plot_path_final = "n/a (PlotError)"
        else:
            log.warning(
                "Bedingungen für ROC-Plot nicht erfüllt. Validierungsdaten: "
                f"True Labels unique {len(np.unique(best_true_val_for_roc)) if best_true_val_for_roc is not None else 'None'}, "
                f"Probs shape {best_probs_val_for_roc.shape if best_probs_val_for_roc is not None else 'None'}."
            )
            roc_plot_path_final = "n/a (NoValDataForROC)"
    else:
        log.warning(
            "Kein bestes Modell während des Trainings gespeichert. "
            "Keine Modell-Artefakte erstellt."
        )

    # --- 7. Evaluierung des besten PyTorch-Modells ---
    torch_eval_results: Dict[str, Any] = {}
    pytorch_eval_cfg = eval_cfg.get("pytorch_model_eval", {})
    run_pytorch_eval_flag = pytorch_eval_cfg.get("run_eval", True)

    if run_pytorch_eval_flag and best_model_state_dict:
        pytorch_eval_n_samples = pytorch_eval_cfg.get("n_samples", None)
        eval_torch_model_instance = QATPrunedSimpleNet(
            input_size,
            num_classes,
            cfg_h_early, # Korrigiert
            qlinear_args_config=qlinear_args,
            qidentity_args_config=qidentity_args,
            qrelu_args_config=qrelu_args,
            dropout_rate1=dropout_rate1,
            dropout_rate2=dropout_rate2,
        ).to(device)
        if cfg_up_early > 0 and input_size > cfg_up_early: # Korrigiert
            eval_torch_model_instance.prune(max_non_zero_per_neuron=cfg_up_early) # Korrigiert
        eval_torch_model_instance.load_state_dict(best_model_state_dict)

        torch_eval_results = evaluate_torch_model( # Gibt jetzt das ausführliche Dict zurück
            eval_torch_model_instance,
            test_loader,
            device,
            label_encoder,
            model_name=f"{artifact_file_prefix}_pytorch_eval",
            results_dir=current_run_results_dir,
            n_samples=pytorch_eval_n_samples,
            logger=log
        )
    elif not run_pytorch_eval_flag:
        log.info("PyTorch Modell-Evaluierung übersprungen (run_eval=false).")
    else:
        log.warning("Kein bestes PyTorch-Modell zum Evaluieren vorhanden.")

    
    # --- 8. FHE-Modell Kompilierung und Evaluierung ---
    quantized_numpy_module = None
    fhe_eval_cfg_main = eval_cfg.get("fhe_model_eval", {})
    # Initialisiere ein Dictionary, um Ergebnisse für mehrere FHE-Modi zu speichern
    all_fhe_eval_results: Dict[str, Any] = {} 
    fhe_compilation_duration_for_log: Union[str, float] = "N/A (Skipped or Not Run)"
    
    run_fhe_pipeline_flag = fhe_eval_cfg_main.get("run_fhe_pipeline", True)
    run_fhe_eval_overall_flag = fhe_eval_cfg_main.get("run_fhe_eval", True) # Gesamtflag für FHE-Eval

    if run_fhe_pipeline_flag and best_model_state_dict:
        log.info("Vorbereitung des besten Modells für FHE Kompilierung...")
        fhe_model_instance = QATPrunedSimpleNet(
            input_size,
            num_classes,
            cfg_h_early,
            qlinear_args_config=qlinear_args,
            qidentity_args_config=qidentity_args,
            qrelu_args_config=qrelu_args,
            dropout_rate1=dropout_rate1,
            dropout_rate2=dropout_rate2,
        )
        if cfg_up_early > 0 and input_size > cfg_up_early:
            fhe_model_instance.prune(max_non_zero_per_neuron=cfg_up_early)
        fhe_model_instance.load_state_dict(best_model_state_dict)
        fhe_model_instance.cpu()

        if cfg_up_early > 0 and input_size > cfg_up_early:
            log.info("Unpruning FHE model instance before compilation.")
            fhe_model_instance.unprune()

        compile_sample_size = min(
            fhe_eval_cfg_main.get("compilation_sample_size", 128), len(X_train_np)
        )
        if compile_sample_size == 0 and len(X_train_np) > 0:
            compile_sample_size = 1
        elif len(X_train_np) == 0:
            log.error("Keine Trainingsdaten für FHE-Kompilierungs-Sample. Überspringe FHE.")
            compile_sample_size = 0
            # run_fhe_eval_overall_flag = False # Keine Evaluierung ohne Kompilierung -> wird durch quantized_numpy_module = None gehandhabt
            fhe_compilation_duration_for_log = "N/A (No Compile Sample)"
            quantized_numpy_module = None # Wichtig für die nachfolgende Prüfung

        if compile_sample_size > 0:
            compile_data_sample = torch.tensor(
                X_train_np[:compile_sample_size], dtype=torch.float32
            ).cpu()
            log.info(f"Starte FHE Kompilierung mit {compile_sample_size} Samples...")
            start_compile_time = time.time()
            try:
                quantized_numpy_module = compile_brevitas_qat_model(
                    fhe_model_instance, compile_data_sample
                )
                fhe_compilation_duration_for_log = round(time.time() - start_compile_time, 2)
                log.info(f"FHE Kompilierung erfolgreich. Dauer: {fhe_compilation_duration_for_log}s")
            except Exception as e_fhe_compile:
                fhe_compilation_duration_for_log = "N/A (Failed)"
                log.error(f"Fehler bei FHE Kompilierung: {e_fhe_compile}\n{traceback.format_exc()}")
                quantized_numpy_module = None # Wichtig für die nachfolgende Prüfung
        
        # FHE Evaluierung für mehrere Modi, falls Kompilierung erfolgreich und Evaluierung gewünscht
        if quantized_numpy_module and run_fhe_eval_overall_flag:
            fhe_modes_to_evaluate = ["simulate", "execute"] # Die Modi, die wir evaluieren wollen
            
            for fhe_mode_to_run in fhe_modes_to_evaluate:
                n_samples_key = f"n_samples_{fhe_mode_to_run}"
                # Fallback auf 0, wenn keine Samples für den Modus spezifiziert sind
                fhe_n_samples_to_run = fhe_eval_cfg_main.get(n_samples_key, 0) 

                if fhe_n_samples_to_run is None or fhe_n_samples_to_run <= 0 : # type: ignore
                    log.info(f"FHE Modell-Evaluierung für Modus '{fhe_mode_to_run}' übersprungen (n_samples nicht > 0).")
                    all_fhe_eval_results[fhe_mode_to_run] = {"status": "skipped", "reason": "n_samples_not_positive"}
                    continue

                log.info(f"Starte FHE Modell-Evaluierung (Modus '{fhe_mode_to_run}', {fhe_n_samples_to_run} Samples)...")
                current_mode_results = evaluate_fhe_model(
                    quantized_numpy_module,
                    raw_data["X_test"],
                    raw_data["y_test"],
                    label_encoder,
                    model_name=f"{artifact_file_prefix}_fhe_eval_{fhe_mode_to_run}",
                    results_dir=current_run_results_dir,
                    fhe_mode=fhe_mode_to_run,
                    n_samples=fhe_n_samples_to_run,
                    logger=log,
                    fhe_compilation_time_s=fhe_compilation_duration_for_log # Wird in evaluate_fhe_model verwendet
                )
                all_fhe_eval_results[fhe_mode_to_run] = current_mode_results
        
        elif not run_fhe_eval_overall_flag and run_fhe_pipeline_flag : # Pipeline lief, aber Eval explizit deaktiviert
             log.info("FHE Modell-Evaluierung übersprungen (run_fhe_eval=false in config).")
        elif not quantized_numpy_module and run_fhe_pipeline_flag and compile_sample_size > 0:
            log.warning(
                "FHE Kompilierung nicht erfolgreich oder übersprungen. "
                "FHE Evaluierung wird nicht durchgeführt."
            )
            
    elif not run_fhe_pipeline_flag:
        log.info("FHE Kompilierung und Evaluierung übersprungen (run_fhe_pipeline=false in config).")
    else: # Kein best_model_state_dict
        log.warning("Kein bestes Modell für FHE Kompilierung und Evaluierung vorhanden.")


    # --- 9. Zusammenfassendes Logging (JSON) ---
    effective_batch_size_value = (
        dataloader_batch_size if use_train_loader else manual_batch_size
    )
    batch_size_source_param = (
        "data_params.dataloader_batch_size"
        if use_train_loader
        else "training_params.manual_batch_size"
    )
    logged_training_params_for_json = train_cfg.copy()
    if use_train_loader:
        logged_training_params_for_json[
            "INFO_effective_dataloader_batch_size_from_data_params"
        ] = dataloader_batch_size
        if "manual_batch_size" in logged_training_params_for_json:
            logged_training_params_for_json[
                "INFO_manual_batch_size_ignored_due_to_use_train_loader"
            ] = logged_training_params_for_json["manual_batch_size"]

    class_weights_values_for_log = "N/A"
    if hasattr(criterion, "weight") and criterion.weight is not None:
        try:
            class_weights_values_for_log = criterion.weight.cpu().tolist()
        except Exception:
            class_weights_values_for_log = "Error converting weights to list"

    val_loss_at_best_f1w_log = "N/A"
    if epoch_of_best_f1w != -1 and epoch_of_best_f1w < len(val_losses_hist):
        val_loss_at_best_f1w_log = round(val_losses_hist[epoch_of_best_f1w], 5)

    log_config_dict = {
        "run_overview": {
            "run_timestamp": run_timestamp,
            "dataset_configured": run_cfg.get("dataset_name"),
            "model_type_configured": model_cfg.get("type"),
            "device_actually_used": str(device),
            "config_file_path_source": config_path,
        },
        "input_configuration_from_yaml": {
            "run_settings": run_cfg,
            "data_params": data_cfg,
            "model_params": model_cfg,
            "training_params": logged_training_params_for_json,
            "evaluation_params": eval_cfg,
        },
        "data_summary_runtime": {
            "npz_file_path_used": npz_file_path,
            "input_features_detected": input_size,
            "output_classes_detected": num_classes,
            "train_samples_processed": len(X_train_tensor),
            "val_samples_processed": len(X_val_tensor),
            "test_samples_in_dataset": len(raw_data["X_test"]),
        },
        "model_details_runtime": {
            "layer_structure_generated": layer_structure,
            "dropout_rate1_applied": dropout_rate1,
            "dropout_rate2_applied": dropout_rate2,
            "n_hidden_applied": cfg_h_early, # Korrigiert/Hinzugefügt
            "quantization_bits_applied": cfg_qb_early, # Korrigiert
            "unpruned_neurons_applied": cfg_up_early, # Korrigiert
        },
        "training_execution_summary": {
            "epochs_run_actual": actual_epochs_run,
            "early_stopping_details_runtime": {
                "patience_used": patience,
                "min_delta_used": min_delta,
                "triggered": (
                    actual_epochs_run < num_epochs
                    and patience_counter_early_stop >= patience
                ),
                "stopped_at_epoch_if_triggered": (
                    actual_epochs_run
                    if (
                        actual_epochs_run < num_epochs
                        and patience_counter_early_stop >= patience
                    )
                    else None
                ),
            },
            "criterion_details_runtime": {
                "name_runtime": type(criterion).__name__,
                "reduction_runtime": getattr(criterion, "reduction", "N/A"),
                "focal_gamma_runtime": (
                    getattr(criterion, "gamma", "N/A")
                    if isinstance(criterion, FocalLoss)
                    else "N/A"
                ),
                "focal_alpha_runtime": (
                    getattr(criterion, "alpha", "N/A")
                    if isinstance(criterion, FocalLoss)
                    else "N/A"
                ),
                "class_weights_applied_runtime": (
                    "yes" if isinstance(class_weights_values_for_log, list) else "no"
                ),
                "class_weights_tensor_values": class_weights_values_for_log,
            },
            "optimizer_details_runtime": {
                "name_runtime": type(optimizer).__name__,
                "initial_learning_rate_runtime": learning_rate,
                "weight_decay_runtime": optimizer.param_groups[0].get(
                    "weight_decay", "N/A"
                ),
                "momentum_runtime": optimizer.param_groups[0].get("momentum", "N/A"),
            },
            "scheduler_details_runtime": {
                "name_runtime": (
                    type(scheduler).__name__ if scheduler is not None else "None"
                ),
                "factor_runtime": (
                    getattr(scheduler, "factor", "N/A")
                    if hasattr(scheduler, "factor")
                    else "N/A"
                ),
                "patience_runtime": (
                    getattr(scheduler, "patience", "N/A")
                    if hasattr(scheduler, "patience")
                    else "N/A"
                ),
                "min_lr_runtime": (
                    getattr(scheduler, "min_lrs", [{"N/A"}])[0] # type: ignore
                    if hasattr(scheduler, "min_lrs") and scheduler.min_lrs # type: ignore
                    else "N/A"
                ),
                "step_size_runtime": (
                    getattr(scheduler, "step_size", "N/A")
                    if hasattr(scheduler, "step_size")
                    else "N/A"
                ),
                "gamma_runtime": (
                    getattr(scheduler, "gamma", "N/A")
                    if hasattr(scheduler, "gamma")
                    else "N/A"
                ),
                "last_lr_runtime": (
                    lr_history_hist[-1] if lr_history_hist else learning_rate
                ),
            },
            "batching_details_runtime": {
                "mode_used_for_training": (
                    "DataLoader" if use_train_loader else "ManualTensorSlicing"
                ),
                "batch_size_parameter_source_in_config": batch_size_source_param,
                "effective_batch_size_applied": effective_batch_size_value,
            },
        },
        "evaluation_execution_details": {
            # PyTorch-Teil
            "pytorch_eval_run_configured": pytorch_eval_cfg.get("run_eval", True), # Ob es in der Config gewünscht war
            "pytorch_eval_actually_run": ( # Ob es tatsächlich gelaufen ist
                pytorch_eval_cfg.get("run_eval", True) and bool(best_model_state_dict)
            ),
            "pytorch_eval_summary": torch_eval_results, # Enthält alle Metriken inkl. samples_processed, Inferenzzeiten etc.

            # FHE-Teil
            "fhe_pipeline_run_configured": fhe_eval_cfg_main.get("run_fhe_pipeline", True), # Ob es in der Config gewünscht war
            "fhe_pipeline_actually_run": ( # Ob die Kompilierung zumindest versucht wurde (Modell musste existieren)
                fhe_eval_cfg_main.get("run_fhe_pipeline", True) and bool(best_model_state_dict)
            ),
            "fhe_compilation_time_s": fhe_compilation_duration_for_log,
            "fhe_compilation_status": (
                "Success"
                if isinstance(fhe_compilation_duration_for_log, (int, float))
                else str(fhe_compilation_duration_for_log) # Sicherstellen, dass es ein String ist bei Fehlern
            ),
            "fhe_eval_run_configured": run_fhe_eval_overall_flag, # Ob FHE-Eval in Config gewünscht war
            "fhe_evaluations": all_fhe_eval_results 
        },
        "best_model_metrics_achieved_val": {
            "best_f1_weighted_val": (
                round(best_f1_weighted_overall, 5)
                if isinstance(best_f1_weighted_overall, float)
                else best_f1_weighted_overall
            ),
            "f1_macro_at_best_f1w_val": (
                round(best_f1_macro_at_best_f1_weighted, 5)
                if isinstance(best_f1_macro_at_best_f1_weighted, float)
                else best_f1_macro_at_best_f1_weighted
            ),
            "roc_auc_at_best_f1w_val": (
                round(best_roc_auc_at_best_f1_weighted, 5)
                if isinstance(best_roc_auc_at_best_f1_weighted, float)
                else best_roc_auc_at_best_f1_weighted
            ),
            "val_loss_at_best_f1w_epoch": val_loss_at_best_f1w_log,
            "epoch_of_best_f1w": (
                epoch_of_best_f1w + 1 if epoch_of_best_f1w != -1 else "N/A"
            ),
        },
        "output_artifact_paths": {
            "results_run_directory": current_run_results_dir, # Korrigiert
            "model_file_saved": model_path_final,
            "loss_plot_file_saved": loss_plot_path_final,
            "roc_plot_file_saved": roc_plot_path_final,
            "detailed_text_log": detailed_log_path, # Hinzugefügt für Vollständigkeit
        },
        "pruning_log_details": {
            "initial_sparsity": initial_pruning_summary,
            "final_sparsity": final_pruning_summary,
            "pruned_layers_list": (
                list(torch_model.pruned_layers) # type: ignore
                if hasattr(torch_model, "pruned_layers") and torch_model.pruned_layers # type: ignore
                else []
            ),
        },
        "epoch_history_data": {
            "train_loss_per_epoch": [
                round(l_val, 5) for l_val in train_losses_hist
            ],
            "val_loss_per_epoch": [round(l_val, 5) for l_val in val_losses_hist],
            "lr_per_epoch": lr_history_hist,
        },
    }

    json_log_file_name = f"{run_timestamp}_{artifact_file_prefix}_full_run_log.json" # Enthält Metriken
    json_log_path_final = os.path.join(current_run_results_dir, json_log_file_name)
    log_config_dict["output_artifact_paths"]["json_log_file_self"] = json_log_path_final

    def default_json_converter(obj: Any) -> Any:
        """Konvertiert nicht-standard JSON-Typen für json.dump."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        # Vorsicht mit __dict__, kann bei komplexen Objekten zu Problemen führen
        if hasattr(obj, "__dict__"):
            try:
                return obj.__dict__
            except TypeError: # Falls __dict__ nicht serialisierbar ist
                return str(obj)
        if isinstance(
            obj,
            (
                optim.lr_scheduler.ReduceLROnPlateau, # type: ignore
                optim.AdamW,
                FocalLoss,
                nn.CrossEntropyLoss,
            ),
        ):
            return f"Object<{obj.__class__.__name__}>"
        if callable(obj):
            return f"Callable<{obj.__name__ if hasattr(obj, '__name__') else str(obj)}>"

        if log.isEnabledFor(logging.DEBUG): # Nur loggen, wenn DEBUG aktiv ist
            log.debug(
                f"Objekt vom Typ {type(obj).__name__} nicht direkt JSON "
                f"serialisierbar. Fallback zu str(). Wert: {str(obj)[:100]}"
            )
        return str(obj) # Fallback

    try:
        with open(json_log_path_final, "w") as f_json:
            json.dump(log_config_dict, f_json, indent=4, default=default_json_converter)
        log.info(f"JSON-Log gespeichert unter: {json_log_path_final}")
    except Exception as e_json_save:
        log.error(
            f"Fehler beim Speichern des JSON-Logs: {e_json_save}\n"
            f"{traceback.format_exc()}"
        )

    log_training_summary(
        log,
        model_path_final,
        best_f1_weighted_overall,
        loss_plot_path_final,
        roc_plot_path_final,
        json_log_path_final,
    )
    log.info(f"QAT Pipeline für '{artifact_file_prefix}' abgeschlossen.")

    # --- Detaillierten Text-Log-Handler entfernen und schließen ---
    if detailed_log_file_handler:
        log.info(f"Schließe detailliertes Text-Log: {detailed_log_path}")
        log.removeHandler(detailed_log_file_handler)
        detailed_log_file_handler.close()
        # detailed_log_file_handler = None # Optional: Zurücksetzen

    return current_run_results_dir


# --- Hauptausführungspunkt des Skripts ---
if __name__ == "__main__":
    """
    Dieser Block wird ausgeführt, wenn das Skript direkt gestartet wird.
    Er dient primär zum Testen der Pipeline mit einer Standard- oder
    einer temporär generierten Konfigurationsdatei.
    """
    cli_logger = setup_logger("QAT_Pipeline_CLI_Runner", level=logging.INFO)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Annahme: Skript liegt in src/training/, Projektwurzel ist zwei Ebenen höher
    project_root = os.path.dirname(os.path.dirname(script_dir))

    default_config_filename = "config.yaml"
    config_path_for_cli = os.path.join(project_root, default_config_filename)

    # Standardpfad für NPZ-Datei, falls nicht in Config spezifiziert
    default_npz_filename = "edgeiiot_dataset_all.npz"
    npz_path_for_dummy_config_default = os.path.join(
        project_root, "data", "processed", default_npz_filename
    )
    npz_path_for_dummy_config = npz_path_for_dummy_config_default # Initialwert

    main_loaded_config: Dict[str, Any] = {} # Hier wird die geladene Konfig gespeichert

    # Versuche, die existierende Hauptkonfiguration zu laden
    if os.path.exists(config_path_for_cli):
        try:
            with open(config_path_for_cli, "r") as f_base_config:
                main_loaded_config = yaml.safe_load(f_base_config) or {} # Sicherstellen, dass es ein Dict ist
            # NPZ-Pfad aus der geladenen Konfiguration extrahieren und validieren
            if main_loaded_config.get("data_params", {}).get("npz_file_path"):
                loaded_npz_path_from_cfg = main_loaded_config["data_params"]["npz_file_path"]
                if not os.path.isabs(loaded_npz_path_from_cfg):
                    loaded_npz_path_from_cfg = os.path.join(project_root, loaded_npz_path_from_cfg)

                if os.path.exists(loaded_npz_path_from_cfg):
                    npz_path_for_dummy_config = loaded_npz_path_from_cfg
                    cli_logger.info(
                        f"NPZ-Pfad aus '{config_path_for_cli}' geladen und "
                        f"validiert: '{npz_path_for_dummy_config}'"
                    )
                else:
                    cli_logger.warning(
                        f"NPZ-Pfad '{loaded_npz_path_from_cfg}' aus Config "
                        f"'{config_path_for_cli}' existiert nicht. "
                        f"Verwende Default: '{npz_path_for_dummy_config_default}'"
                    )
            else:
                cli_logger.info(
                    f"Kein NPZ-Pfad in Config '{config_path_for_cli}' gefunden. "
                    f"Verwende Default: '{npz_path_for_dummy_config_default}'"
                )
        except Exception as e_load_main_cfg:
            cli_logger.error(
                f"Fehler beim Laden der Basis-Config '{config_path_for_cli}': "
                f"{e_load_main_cfg}. Verwende Default NPZ-Pfad."
            )
            main_loaded_config = {} # Im Fehlerfall leeres Dict
    else:
        cli_logger.info(
            f"Haupt-Config '{config_path_for_cli}' nicht gefunden. "
            "Verwende Default NPZ-Pfad für Dummy-Config."
        )
        main_loaded_config = {}

    config_to_run_pipeline_with = main_loaded_config
    # Wenn keine Config geladen wurde oder sie leer ist, erstelle eine temporäre Dummy-Config
    if not os.path.exists(config_path_for_cli) or not main_loaded_config:
        cli_logger.warning(
            f"Erstelle temporäre Test-Konfigurationsdatei, da "
            f"'{config_path_for_cli}' nicht gefunden oder leer war."
        )
        temp_config_filename = "temp_config_for_cli.yaml"
        # Die Pipeline wird mit dieser temporären Config-Datei aufgerufen
        config_path_for_cli = os.path.join(project_root, temp_config_filename)
        dummy_results_base_dir = os.path.join(
            project_root, "results", "script_temp_runs"
        )

        if not os.path.exists(npz_path_for_dummy_config):
            cli_logger.error(
                f"KRITISCH: NPZ-Datei für Dummy-Config NICHT GEFUNDEN unter: "
                f"{npz_path_for_dummy_config}"
            )
            cli_logger.error(
                "Pipeline wird wahrscheinlich fehlschlagen. "
                "Bitte Pfad korrigieren oder Datei bereitstellen."
            )
        # Inhalt der Dummy-Konfiguration
        dummy_config_content = {
            "run_settings": {
                "dataset_name": "EdgeIIoT_CLI_Temp",
                "results_base_dir": dummy_results_base_dir,
                "device_str": "auto",
                "logger_instance_name": "QAT_Pipeline_TempCfg",
                "logger_level": "DEBUG",
            },
            "data_params": {
                "npz_file_path": npz_path_for_dummy_config, # Verwendet den validierten oder Default-Pfad
                "dataloader_batch_size": 64,
                "use_subset_training": True,
                "subset_fraction": 0.002, # Kleines Subset für schnelle Tests
                "refit_scaler": False,
                "refit_encoder": False,
            },
            "model_params": {
                "type": "QATPrunedSimpleNet",
                "n_hidden": 16, # Kleinere Werte für schnellen Test
                "quantization_bits": 2,
                "unpruned_neurons": 4,
                "dropout": {"rate1": 0.0, "rate2": 0.0},
                "qlinear_bias": True,
                "qidentity_return_quant_tensor": True,
                "qrelu_return_quant_tensor": True,
            },
            "training_params": {
                "num_epochs": 1, # Nur eine Epoche für schnellen Test
                "manual_batch_size": 16,
                "use_train_loader_for_batches": True,
                "learning_rate": 0.001,
                "criterion": {
                    "name": "FocalLoss",
                    "calculate_class_weights": True,
                    "class_weight_calculation_method": "inverse_frequency",
                    "focal_loss_alpha": 0.25,
                    "focal_loss_gamma": 2.0,
                    "reduction": "mean",
                },
                "optimizer": {"name": "AdamW", "weight_decay": 0.01},
                "scheduler": {"name": "null", "params": {}}, # Kein Scheduler für 1 Epoche
                "early_stopping": {"patience": 1, "min_delta": 0.01},
            },
            "evaluation_params": {
                "pytorch_model_eval": {"run_eval": True, "n_samples": 20},
                "fhe_model_eval": {
                    "run_fhe_pipeline": False, # FHE oft zeitaufwändig
                    "compilation_sample_size": 16,
                    "run_fhe_eval": False,
                    "mode": "simulate",
                    "n_samples_simulate": 10,
                    "n_samples_execute": 1,
                },
            },
        }
        try:
            os.makedirs(dummy_results_base_dir, exist_ok=True)
            with open(config_path_for_cli, "w") as cf_temp_write:
                yaml.dump(dummy_config_content, cf_temp_write, indent=2, sort_keys=False)
            cli_logger.info(
                f"Temporäre Konfigurationsdatei '{config_path_for_cli}' erstellt."
            )
            # config_to_run_pipeline_with wird hier nicht aktualisiert, da
            # run_qat_training_pipeline direkt den Pfad config_path_for_cli verwendet,
            # der auf die temporäre Datei zeigt.
        except Exception as e_yaml_write_cli:
            cli_logger.error(
                "Konnte temporäre Konfigurationsdatei nicht schreiben: "
                f"{e_yaml_write_cli}"
            )
            exit(1) # Kritischer Fehler, wenn Config nicht geschrieben werden kann

    # Finale Überprüfung des NPZ-Pfads aus der Konfiguration, die tatsächlich verwendet wird
    # (entweder die geladene Hauptconfig oder die gerade erstellte Dummy-Config).
    # Dafür muss die Config ggf. neu geladen werden, wenn sie temporär war.
    final_config_for_npz_check: Dict[str, Any] = {}
    try:
        with open(config_path_for_cli, "r") as f_final_cfg_check:
            final_config_for_npz_check = yaml.safe_load(f_final_cfg_check) or {}
    except Exception as e_load_final_cfg:
        cli_logger.error(f"Konnte endgültige Config '{config_path_for_cli}' für NPZ-Check nicht laden: {e_load_final_cfg}")
        # exit(1) # Exit, if this is critical.

    final_npz_path_from_config_check = final_config_for_npz_check.get("data_params", {}).get(
        "npz_file_path"
    )
    if final_npz_path_from_config_check:
        path_to_verify = final_npz_path_from_config_check
        if not os.path.isabs(path_to_verify):
            path_to_verify = os.path.join(project_root, path_to_verify)
        if not os.path.exists(path_to_verify):
            cli_logger.error(
                f"KRITISCH: Die endgültig konfigurierte NPZ-Datendatei "
                f"'{path_to_verify}' (abgeleitet von "
                f"'{final_npz_path_from_config_check}') wurde nicht gefunden."
            )
    elif not final_npz_path_from_config_check:
        cli_logger.error(
            "KRITISCH: Kein NPZ-Pfad ('npz_file_path') in der endgültigen "
            "Konfiguration gefunden, die für den Lauf verwendet wird."
        )

    cli_logger.info(
        f"Starte QAT Trainingspipeline als Skript mit Config: {config_path_for_cli}"
    )
    try:
        # Stelle sicher, dass run_qat_training_pipeline die korrekte config_path verwendet
        run_qat_training_pipeline(config_path=config_path_for_cli)
    except FileNotFoundError as e_fnf_in_main:
        cli_logger.error(
            f"FileNotFoundError: {e_fnf_in_main}. Sicherstellen, dass Pfade "
            "(insb. NPZ) korrekt sind."
        )
        cli_logger.error(traceback.format_exc())
    except Exception as e_main_run_exc:
        cli_logger.error(
            "Unerwarteter Fehler während der Pipeline-Ausführung: "
            f"{e_main_run_exc}"
        )
        cli_logger.error(traceback.format_exc())
    cli_logger.info("QAT Trainingspipeline Skriptlauf beendet.")