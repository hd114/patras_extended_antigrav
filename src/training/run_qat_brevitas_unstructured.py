#!/usr/bin/env python
# coding: utf-8

"""
Hauptskript für das Quantization-Aware Training (QAT) einer Pipeline
für das EdgeIIoT-Dataset. Dieses Skript handhabt das Laden von Daten,
Modellinitialisierung, Training, Evaluierung (PyTorch und FHE-simuliert),
und das Logging von Ergebnissen und Artefakten, inklusive Weights & Biases.
"""

import time
import os
import json
import logging
import yaml
import traceback
from datetime import datetime
from typing import Union, Optional, Tuple, Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import RocCurveDisplay, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

import brevitas.nn as qnn
from concrete.ml.torch.compile import compile_brevitas_qat_model

# Weights & Biases Import
import wandb

# Eigene Modul-Importe
from src.utils.logger import (
    setup_logger,
    log_training_epoch,
    log_training_summary
)
from src.utils.config_loader import load_config
from src.data.edge_iiot_dataset import (
    EdgeIIoTDataset,
    load_edgeiiot_data
)
from src.models.qat_model import QATPrunedSimpleNet, get_pruning_summary
from src.evaluation.concrete_evaluate import (
    evaluate_torch_model,
    evaluate_fhe_model,
    _calculate_robust_roc_auc
)
from src.training.custom_losses import FocalLoss


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
    use_loader: bool
) -> float:
    """
    Führt einen einzelnen Trainingsschritt (Epoche) für das gegebene Modell durch.
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
        leave=False
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
        if logger: logger.warning("Keine Batches in der Epoche verarbeitet.")
        return 0.0


def _validate_epoch_step(
    model: nn.Module,
    X_val_tensor: torch.Tensor,
    y_val_tensor: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    label_encoder_global: LabelEncoder,
    epoch_num: int,
    logger: logging.Logger
) -> Tuple[float, float, float, Union[str, float], np.ndarray, np.ndarray]:
    """
    Führt einen einzelnen Validierungsschritt (Epoche) für das gegebene Modell durch.
    """
    model.eval()
    current_val_loss = 0.0
    f1_weighted, f1_macro = 0.0, 0.0
    true_labels_val_list, probs_val_list = [], []
    val_batch_size = 512
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
    roc_auc_macro = "n/a (no_val_samples)"
    if total_val_samples > 0:
        true_labels_val = np.concatenate(true_labels_val_list).astype(int)
        probs_val = np.concatenate(probs_val_list, axis=0)
        preds_val = np.argmax(probs_val, axis=1).astype(int)
        metric_labels_val = np.unique(np.concatenate((true_labels_val, preds_val)))
        if len(metric_labels_val) == 0:
            f1_weighted, f1_macro = 0.0, 0.0
            log_msg_val = "Keine Labels für Metrikberechnung in Validierung."
            if logger: logger.warning(log_msg_val)
            else: print(f"WARNUNG: {log_msg_val}")
        else:
            f1_weighted = f1_score(true_labels_val, preds_val, labels=metric_labels_val, average="weighted", zero_division=0)
            f1_macro = f1_score(true_labels_val, preds_val, labels=metric_labels_val, average="macro", zero_division=0)
            roc_auc_macro = _calculate_robust_roc_auc(true_labels_val, probs_val, logger, context_msg=f"Ep{epoch_num + 1} Val")
    else:
        true_labels_val, probs_val = np.array([]), np.array([])
    return (current_val_loss, f1_weighted, f1_macro, roc_auc_macro, true_labels_val, probs_val)


# --- Haupt-Trainings- und Kompilierungsfunktion ---

def run_qat_training_pipeline(config_path: str = "config.yaml") -> Optional[str]:
    """
    Führt die gesamte QAT-Trainingspipeline aus.
    """
    # --- 1. Konfiguration und initiales Setup ---
    config = load_config(config_path)
    if not config:
        return None

    run_cfg = config.get("run_settings", {})
    data_cfg = config.get("data_params", {})
    model_cfg = config.get("model_params", {})
    train_cfg = config.get("training_params", {})
    eval_cfg = config.get("evaluation_params", {})
    wandb_cfg = config.get("wandb_settings", {})

    log_level_str = run_cfg.get("logger_level", "INFO").upper()
    logger_name = run_cfg.get("logger_instance_name", "qat_pipeline_run")
    log = setup_logger(name=logger_name, level=log_level_str)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = run_cfg.get("dataset_name", "UnknownDataset")

    # Basis-Identifikator für Dateinamen (ohne Laufzeit-Metriken wie F1-Score)
    cfg_h_early = model_cfg.get("n_hidden", "N")
    cfg_qb_early = model_cfg.get("quantization_bits", "N")
    cfg_up_early = model_cfg.get("unpruned_neurons", "N")
    base_run_identifier = f"QAT_{dataset_name}_{cfg_h_early}h_qb{cfg_qb_early}_p{cfg_up_early}"
    
    # `current_run_results_dir` wird vom `experiment_runner.py` über `run_cfg` übergeben
    # und ist der eindeutige Ordner für diesen spezifischen Lauf (inkl. Wiederholungsnummer).
    # Fallback, falls nicht vom Runner gesetzt (z.B. bei direktem Aufruf des Skripts).
    '''results_base_dir_from_cfg = run_cfg.get("results_base_dir", "results/default_runs")
    current_run_results_dir = run_cfg.get(
        "current_run_actual_results_dir", 
        os.path.join(results_base_dir_from_cfg, f"{run_timestamp}_{base_run_identifier}")
    )'''

    # Fallback, falls nicht vom Runner gesetzt (z.B. bei direktem Aufruf des Skripts).
    results_base_dir_from_cfg = run_cfg.get("results_base_dir", "results/default_runs")

    # Logik für den Ergebnisordner:
    # 1. Versuche, den vom Runner vorgesehenen EXAKTEN Pfad zu verwenden (beste Praxis).
    current_run_results_dir = run_cfg.get("current_run_actual_results_dir")

    # 2. Wenn der Pfad nicht explizit gesetzt ist, prüfe, ob es sich um einen
    #    'meta_run' (GridSearch) handelt. Wenn ja, verwenden wir den Pfad aus
    #    'results_base_dir_from_cfg' DIREKT, um den Fehler der doppelten Ordnerstruktur
    #    zu vermeiden, den der Runner erwartet.
    if current_run_results_dir:
        log.debug(f"Verwende 'current_run_actual_results_dir' (explizit): {current_run_results_dir}")
    elif "meta_run" in results_base_dir_from_cfg:
        # Runner-Lauf erkannt (z.B. .../rep_1). Wir verwenden diesen Pfad als endgültiges Ziel.
        log.info(f"Runner-Lauf ('meta_run') erkannt. Verwende '{results_base_dir_from_cfg}' als finalen Pfad.")
        current_run_results_dir = results_base_dir_from_cfg
    else:
        # Standard-Fallback für eigenständige Läufe (ursprüngliches Verhalten).
        log.info("Kein Runner-Lauf erkannt. Erstelle Zeitstempel-Ordner.")
        current_run_results_dir = os.path.join(
            results_base_dir_from_cfg, f"{run_timestamp}_{base_run_identifier}"
        )
        
    try:
        os.makedirs(current_run_results_dir, exist_ok=True)
    except OSError as e:
        # Frühes Logging, falls Hauptlogger noch nicht initialisiert ist oder FileHandler nicht gesetzt werden kann
        print(f"KRITISCHER FEHLER: Konnte Ergebnisordner '{current_run_results_dir}' nicht erstellen: {e}. Pipeline-Abbruch.")
        if log: log.critical(f"FEHLER beim Erstellen des Ergebnisordners '{current_run_results_dir}': {e}. Pipeline-Abbruch.")
        return None


    detailed_log_file_handler: Optional[logging.FileHandler] = None
    detailed_log_path = "n/a"
    
    try:
        detailed_log_filename = f"{run_timestamp}_{base_run_identifier}_pipeline_full.log"
        detailed_log_path = os.path.join(current_run_results_dir, detailed_log_filename)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S,%03d'
        )
        
        for handler in list(log.handlers):
            if isinstance(handler, logging.FileHandler) and handler.name == "detailed_run_log_handler":
                log.removeHandler(handler)
                handler.close()

        detailed_log_file_handler = logging.FileHandler(detailed_log_path, mode='w', encoding='utf-8')
        detailed_log_file_handler.setFormatter(file_formatter)
        detailed_log_file_handler.setLevel(log.level) 
        detailed_log_file_handler.name = "detailed_run_log_handler"
        log.addHandler(detailed_log_file_handler)
        
        log.info(f"Detailliertes Text-Log wird nach '{detailed_log_path}' geschrieben.")
        log.info(f"Alle Ergebnisse für diesen spezifischen Run werden in '{current_run_results_dir}' gespeichert.")

        device_str = run_cfg.get("device_str", "auto")
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)
        log.info(f"Verwende Device: {device}")

        wandb_mode = wandb_cfg.get("mode", "online")
        if wandb_mode != "disabled":
            try:
                wandb_run_dir = os.path.join(current_run_results_dir, "wandb")
                os.makedirs(wandb_run_dir, exist_ok=True)
                # Verwende base_run_identifier für den wandb-Laufnamen, um Konsistenz mit Ordnernamen zu haben
                # Der Zeitstempel ist bereits Teil des current_run_results_dir
                run_name_for_wandb = f"{wandb_cfg.get('run_name_prefix', 'run')}_{os.path.basename(current_run_results_dir)}"
                
                wandb.init(
                    project=wandb_cfg.get("project_name", "FHE_QAT_Project"),
                    entity=wandb_cfg.get("entity"),
                    name=run_name_for_wandb,
                    config=config, 
                    dir=wandb_run_dir, 
                    mode=wandb_mode,
                    reinit=True, 
                    save_code=True 
                )
                log.info(
                    f"Weights & Biases initialisiert. Projekt: {wandb.config.wandb_settings.get('project_name', 'default')}, "
                    f"Run: {wandb.run.name if wandb.run else 'FEHLER BEI RUN NAME'}"
                )
            except Exception as e_wandb_init:
                log.error(f"Fehler bei der Initialisierung von Weights & Biases: {e_wandb_init}")
                log.warning("Weights & Biases Logging wird für diesen Lauf deaktiviert.")
                wandb_mode = "disabled"
        else:
            log.info("Weights & Biases Logging ist deaktiviert (mode='disabled' in Config).")

        npz_file_path = data_cfg.get("npz_file_path")
        if not npz_file_path:
            log.error("FEHLER: 'npz_file_path' nicht in data_params der Config. Abbruch.")
            raise ValueError("'npz_file_path' nicht in data_params der Config.")

        log.info(f"Lade Daten für Dataset: {dataset_name} von {npz_file_path}")
        dataloader_batch_size = data_cfg.get("dataloader_batch_size", 256)
        train_loader, val_loader, test_loader, raw_data = load_edgeiiot_data(
            npz_path=npz_file_path, batch_size=dataloader_batch_size,
            refit_scaler=data_cfg.get("refit_scaler", False),
            refit_encoder=data_cfg.get("refit_encoder", False),
            return_raw_data=True, logger=log
        )
        if raw_data is None:
            log.error("Fehler beim Laden der Daten. Abbruch der Pipeline.")
            raise ValueError("Fehler beim Laden der Rohdaten.")
        input_size = raw_data["num_features"]
        num_classes = raw_data["num_classes"]
        label_encoder = raw_data["label_encoder"]
        X_train_np_all, y_train_np_all = raw_data["X_train"], raw_data["y_train"]
        if data_cfg.get("use_subset_training", False):
            subset_fraction = data_cfg.get("subset_fraction", 0.1)
            if not (0 < subset_fraction <= 1.0):
                log.warning(f"Ungültige subset_fraction ({subset_fraction}). Verwende 0.1.")
                subset_fraction = 0.1
            subset_size = int(len(X_train_np_all) * subset_fraction)
            X_train_np, y_train_np = X_train_np_all[:subset_size], y_train_np_all[:subset_size]
            log.info(f"Verwende Subset von {len(X_train_np)} Trainingssamples ({subset_fraction*100:.1f}%).")
        else:
            X_train_np, y_train_np = X_train_np_all, y_train_np_all
            log.info(f"Verwende vollen Trainingsdatensatz ({len(X_train_np)} Samples).")

        n_hidden = model_cfg.get("n_hidden", 100)
        quantization_bits = model_cfg.get("quantization_bits", 3)
        unpruned_neurons = model_cfg.get("unpruned_neurons", 16)
        dropout_config = model_cfg.get("dropout", {})
        dropout_rate1 = dropout_config.get("rate1", 0.0)
        dropout_rate2 = dropout_config.get("rate2", 0.0)
        log.info(
            f"Modellparameter (aus Config): n_hidden={n_hidden}, qbits={quantization_bits}, "
            f"unpruned={unpruned_neurons}, dropout1={dropout_rate1}, "
            f"dropout2={dropout_rate2}"
        )
        qlinear_args = {"weight_bit_width": quantization_bits, "bias": model_cfg.get("qlinear_bias", True)}
        qidentity_args = {"bit_width": quantization_bits, "return_quant_tensor": model_cfg.get("qidentity_return_quant_tensor", True)}
        qrelu_args = {"bit_width": quantization_bits, "return_quant_tensor": model_cfg.get("qrelu_return_quant_tensor", True)}
        torch_model = QATPrunedSimpleNet(
            input_size, num_classes, n_hidden,
            qlinear_args_config=qlinear_args,
            qidentity_args_config=qidentity_args,
            qrelu_args_config=qrelu_args,
            dropout_rate1=dropout_rate1,
            dropout_rate2=dropout_rate2
        ).to(device)
        if unpruned_neurons > 0 and input_size > unpruned_neurons:
            torch_model.prune(max_non_zero_per_neuron=unpruned_neurons)
        initial_pruning_summary = get_pruning_summary(torch_model)
        log.info(f"Initiales Pruning Summary: {initial_pruning_summary}")
        layer_structure = []
        for name, m in torch_model.named_modules():
            is_relevant = False; layer_type = type(m).__name__
            in_f, out_f, has_b = None, None, None
            if isinstance(m, qnn.QuantLinear): is_relevant = True
            elif isinstance(m, torch.nn.Linear): is_relevant = True; layer_type = "torch.nn.Linear"
            elif hasattr(torch.nn, 'qat') and isinstance(m, torch.nn.qat.Linear): is_relevant = True; layer_type = "torch.nn.qat.Linear"
            if is_relevant:
                if hasattr(m, 'in_features'): in_f = m.in_features
                if hasattr(m, 'out_features'): out_f = m.out_features
                if hasattr(m, 'bias'): has_b = m.bias is not None
                layer_structure.append({"name": name, "type": layer_type, "in": in_f, "out": out_f, "bias": has_b})
        log.info(f"Modell Layer Struktur: {layer_structure}")

        learning_rate = train_cfg.get("learning_rate", 0.001)
        num_epochs = train_cfg.get("num_epochs", 10)
        early_stopping_cfg = train_cfg.get("early_stopping", {})
        patience = early_stopping_cfg.get("patience", 15)
        min_delta = early_stopping_cfg.get("min_delta", 1e-5)
        use_train_loader = train_cfg.get("use_train_loader_for_batches", False)
        manual_batch_size = train_cfg.get("manual_batch_size", 256)
        criterion_cfg = train_cfg.get("criterion", {})
        criterion_name = criterion_cfg.get("name", "CrossEntropyLoss").lower()
        log.debug(f"DEBUG: Konfigurierter Criterion-Name: '{criterion_cfg.get('name')}'")
        weights_tensor = None
        class_weights_list = criterion_cfg.get("class_weights")
        if class_weights_list is not None and isinstance(class_weights_list, list):
            try:
                if len(class_weights_list) == num_classes:
                    weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)
                    log.info(f"Verwende manuell konfigurierte Klassengewichte: {weights_tensor.tolist()}")
                else:
                    log.warning(f"Länge der manuellen class_weights ({len(class_weights_list)}) entspricht nicht num_classes ({num_classes}). Ignoriere manuelle Gewichte.")
            except Exception as e_cw:
                log.warning(f"Konnte manuell konfigurierte class_weights nicht als Tensor konvertieren: {e_cw}. Fahre ohne manuelle Gewichte fort.")
                weights_tensor = None
        if weights_tensor is None and criterion_cfg.get("calculate_class_weights", False):
            method = criterion_cfg.get("class_weight_calculation_method", "inverse_frequency")
            log.info(f"Berechne Klassengewichte (Methode: {method})...")
            if y_train_np is not None and len(y_train_np) > 0:
                if num_classes > 0:
                    counts = np.bincount(y_train_np.astype(int), minlength=num_classes)
                    calc_weights_list = []
                    total_samples_for_w = len(y_train_np)
                    if method == "effective_number":
                        beta = criterion_cfg.get("class_weight_beta", 0.999)
                        if not (0 <= beta < 1.0): log.warning(f"Ungültiger Beta-Wert ({beta}). Verwende 0.999."); beta = 0.999
                        for i in range(num_classes):
                            count = counts[i] if i < len(counts) else 0
                            if count > 0: eff_num = (1.0 - beta**count) / (1.0 - beta)
                            else: eff_num = (1.0 - beta**1) / (1.0 - beta) if beta < 1.0 else 1.0
                            calc_weights_list.append(1.0 / eff_num if eff_num > 1e-9 else 1.0)
                        temp_w = torch.tensor(calc_weights_list, dtype=torch.float32)
                        if temp_w.sum() > 1e-9: weights_tensor = (temp_w / temp_w.sum() * num_classes).to(device)
                        else: weights_tensor = torch.ones(num_classes, dtype=torch.float32).to(device); log.warning("Summe der 'Effective Number' Gewichte war 0. Verwende neutrale Gewichte.")
                        log.info(f"Automatisch berechnete 'Effective Number' Klassengewichte (beta={beta}, normalisiert): {weights_tensor.tolist()}")
                    elif method == "inverse_frequency":
                        for i in range(num_classes):
                            count = counts[i] if i < len(counts) else 0
                            calc_weights_list.append(total_samples_for_w / (num_classes * count) if count > 0 else 1.0)
                        weights_tensor = torch.tensor(calc_weights_list, dtype=torch.float32).to(device)
                        log.info(f"Automatisch berechnete 'Inverse Frequency' Klassengewichte: {weights_tensor.tolist()}")
                    else: log.warning(f"Unbekannte class_weight_calculation_method: {method}. Keine Gewichte berechnet.")
                else: log.error("num_classes ist <= 0, kann Klassengewichte nicht berechnen.")
            else: log.warning("Keine Trainingslabels (y_train_np) zur Berechnung der Klassengewichte verfügbar.")
        if (weights_tensor is not None and criterion_cfg.get("calculate_class_weights", False)):
            scale = criterion_cfg.get("auto_weight_overall_scale", 1.0)
            if scale != 1.0: weights_tensor *= scale; log.info(f"Automatische Klassengewichte skaliert mit Faktor {scale}: {weights_tensor.tolist()}")
        if weights_tensor is None: log.info("Keine Klassengewichte werden für die Verlustfunktion verwendet.")
        reduction = criterion_cfg.get("reduction", "mean")
        if criterion_name == "focalloss":
            alpha = criterion_cfg.get("focal_loss_alpha", 0.25); gamma = criterion_cfg.get("focal_loss_gamma", 2.0)
            try:
                criterion = FocalLoss(alpha=alpha, gamma=gamma, weight=weights_tensor, reduction=reduction)
                log.info(f"Verwende FocalLoss (alpha={alpha}, gamma={gamma}, reduction='{reduction}', mit Gewichten: {weights_tensor is not None})")
            except Exception as e_crit:
                log.error(f"FEHLER bei Instanziierung von FocalLoss: {e_crit}. Fallback zu CrossEntropyLoss."); criterion = nn.CrossEntropyLoss(weight=weights_tensor, reduction=reduction)
                log.info(f"Verwende CrossEntropyLoss als Fallback (reduction='{reduction}', mit Gewichten: {weights_tensor is not None}).")
        else:
            criterion = nn.CrossEntropyLoss(weight=weights_tensor, reduction=reduction)
            log.info(f"Verwende CrossEntropyLoss (Config war: '{criterion_cfg.get('name', 'CrossEntropyLoss')}'), reduction='{reduction}', mit Gewichten: {weights_tensor is not None}")
        optimizer_cfg = train_cfg.get("optimizer", {}); optimizer_name = optimizer_cfg.get("name", "AdamW").lower(); opt_weight_decay = optimizer_cfg.get("weight_decay", 0.01); opt_momentum = optimizer_cfg.get("momentum", 0.9)
        if optimizer_name == "adamw": optimizer = optim.AdamW(torch_model.parameters(), lr=learning_rate, weight_decay=opt_weight_decay)
        elif optimizer_name == "adam": optimizer = optim.Adam(torch_model.parameters(), lr=learning_rate, weight_decay=opt_weight_decay)
        elif optimizer_name == "sgd": optimizer = optim.SGD(torch_model.parameters(), lr=learning_rate, momentum=opt_momentum, weight_decay=opt_weight_decay)
        else: log.warning(f"Unbekannter Optimizer '{optimizer_name}'. Verwende AdamW."); optimizer = optim.AdamW(torch_model.parameters(), lr=learning_rate, weight_decay=opt_weight_decay)
        log.info(f"Verwende Optimizer: {type(optimizer).__name__} mit initialer lr={learning_rate}")
        
        scheduler_cfg = train_cfg.get("scheduler", {})
        scheduler_name = scheduler_cfg.get("name")
        scheduler = None
        if (scheduler_name and isinstance(scheduler_name, str) and
                scheduler_name.lower() not in ["null", "none", ""]):
            s_name_lower = scheduler_name.lower()
            log.debug(f"DEBUG: Verarbeite Scheduler-Namen: '{s_name_lower}'")
            
            # Parameter werden aus dem 'params' Unter-Dictionary gelesen, falls vorhanden
            params_from_config = scheduler_cfg.get("params", {}) 

            if s_name_lower == "reducelronplateau":
                actual_params = {
                    "factor": params_from_config.get("factor", scheduler_cfg.get("reduce_lr_factor", 0.1)),
                    "patience": params_from_config.get("patience", scheduler_cfg.get("reduce_lr_patience", 10)),
                    "min_lr": params_from_config.get("min_lr", scheduler_cfg.get("reduce_lr_min_lr", 1e-6))
                }
                # verbose wird nicht an den Konstruktor übergeben
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', **actual_params
                )
                log.info(f"Verwende ReduceLROnPlateau Scheduler mit Parametern: {actual_params}")
            elif s_name_lower == "steplr":
                actual_params = {
                    "step_size": params_from_config.get("step_size", scheduler_cfg.get("step_lr_step_size", 10)),
                    "gamma": params_from_config.get("gamma", scheduler_cfg.get("step_lr_gamma", 0.1))
                }
                scheduler = optim.lr_scheduler.StepLR(optimizer, **actual_params)
                log.info(f"Verwende StepLR Scheduler mit Parametern: {actual_params}")
            else:
                log.warning(
                    f"Unbekannter Scheduler-Name '{scheduler_name}'. "
                    "Kein Scheduler verwendet."
                )
        else:
            log.info("Kein Scheduler konfiguriert.")

        # --- 5. Trainingsschleife ---
        train_losses, val_losses, lr_history = [], [], []
        best_val_loss = float('inf'); patience_counter_early_stop = 0
        best_f1_w = 0.0; best_model_state = None
        best_true_val, best_probs_val = None, None
        best_f1_m_at_best_f1_w, best_roc_auc_at_best_f1_w = "N/A", "N/A"
        best_epoch = -1
        X_train_t = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_np, dtype=torch.long).to(device)
        X_val_t = torch.tensor(raw_data["X_val"], dtype=torch.float32).to(device)
        y_val_t = torch.tensor(raw_data["y_val"], dtype=torch.long).to(device)
        log.info(f"Starte Training für {num_epochs} Epochen...")
        actual_epochs = 0
        for epoch_idx in range(num_epochs):
            actual_epochs = epoch_idx + 1; epoch_start_time = time.time()
            current_lr = optimizer.param_groups[0]['lr']; lr_history.append(current_lr)
            train_data_source = train_loader if use_train_loader else X_train_t
            train_labels_source = y_train_t if not use_train_loader else None
            current_manual_batch_size = manual_batch_size if not use_train_loader else None
            avg_train_loss = _train_epoch_step(torch_model, train_data_source, train_labels_source, current_manual_batch_size, criterion, optimizer, device, epoch_idx, num_epochs, log, use_loader=use_train_loader)
            train_losses.append(avg_train_loss)
            avg_val_loss, f1_w, f1_m, roc_auc_m, true_val, probs_v = _validate_epoch_step(torch_model, X_val_t, y_val_t, criterion, device, label_encoder, epoch_idx, log)
            val_losses.append(avg_val_loss)
            log_training_epoch(log, epoch_idx, avg_train_loss, avg_val_loss, f1_w, f1_m, roc_auc_m, epoch_start_time)
            if wandb_mode != "disabled" and wandb.run:
                wandb.log({"epoch": epoch_idx + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_f1_weighted": f1_w, "val_f1_macro": f1_m, "val_roc_auc": roc_auc_m if isinstance(roc_auc_m, float) else -1, "learning_rate": current_lr}, step=epoch_idx + 1)
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(avg_val_loss)
                else: scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != current_lr: log.info(f"LR durch Scheduler in Epoche {epoch_idx + 1} angepasst auf: {new_lr:.7f}")
            if f1_w > best_f1_w:
                best_f1_w = f1_w; best_model_state = torch_model.state_dict(); best_true_val, best_probs_val = true_val, probs_v; best_f1_m_at_best_f1_w = f1_m; best_roc_auc_at_best_f1_w = roc_auc_m; best_epoch = epoch_idx
                log.info(f"Neues bestes Modell in Epoche {epoch_idx + 1} gefunden (F1-Weighted: {f1_w:.4f}, Val Loss: {avg_val_loss:.4f})")
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss; patience_counter_early_stop = 0
            else:
                patience_counter_early_stop += 1; log.debug(f"EarlyStopping: Val loss nicht verbessert. Patience: {patience_counter_early_stop}/{patience}")
                if patience_counter_early_stop >= patience: log.info(f"Early Stopping ausgelöst in Epoche {epoch_idx + 1}."); break
        final_pruning_summary = get_pruning_summary(torch_model)
        log.info(f"Finales Pruning Summary: {final_pruning_summary}")
        if wandb_mode != "disabled" and wandb.run: wandb.summary["final_pruning_summary"] = final_pruning_summary

        # --- 6. Artefakt-Management ---
        artifact_file_prefix = "no_model_saved"
        if best_model_state:
            f1_str = f"{best_f1_w:.4f}".replace("0.", "").replace(".", "")
            artifact_file_prefix = f"{base_run_identifier}_ep{actual_epochs}_f1w{f1_str}"
        else:
            artifact_file_prefix = f"{base_run_identifier}_ep{actual_epochs}_no_best_model"
        model_path, loss_plot_path, roc_plot_path = "n/a", "n/a", "n/a"
        if best_model_state:
            model_path = os.path.join(current_run_results_dir, f"{artifact_file_prefix}.pth"); torch.save(best_model_state, model_path); log.info(f"Bestes Modell gespeichert: {model_path}")
            loss_plot_path = os.path.join(current_run_results_dir, f"{artifact_file_prefix}_loss.png")
            plt.figure(figsize=(12, 5)); plt.subplot(1,2,1); plt.plot(range(1, actual_epochs + 1), train_losses,label='Train Loss', marker='.'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title("Trainingsverlauf"); plt.grid(True); plt.legend()
            plt.subplot(1,2,2); plt.plot(range(1, actual_epochs + 1), val_losses,label='Val Loss', marker='.'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title("Validierungsverlauf"); plt.grid(True); plt.legend()
            plt.tight_layout(); plt.savefig(loss_plot_path); plt.close(); log.info(f"Loss-Plot gespeichert: {loss_plot_path}")
            roc_plot_path = "n/a (init_roc_plot)"
            can_plot_roc = (best_true_val is not None and best_probs_val is not None and len(np.unique(best_true_val)) >= 2 and best_probs_val.shape[0] > 0 and best_probs_val.shape[1] >=2)
            if can_plot_roc:
                try:
                    roc_plot_path = os.path.join(current_run_results_dir, f"{artifact_file_prefix}_roc_curves.png"); fig_roc, ax_roc = plt.subplots(figsize=(10, 8)); num_plot_cls = best_probs_val.shape[1]
                    plot_cls_names = [f"Class {i}" for i in range(num_plot_cls)]
                    if (hasattr(label_encoder, 'classes_') and label_encoder.classes_ is not None and len(label_encoder.classes_) == num_plot_cls): plot_cls_names = [str(cn) for cn in label_encoder.classes_]
                    else: log.warning(f"Klassennamen vom LabelEncoder nicht für ROC-Plot verwendbar (erwartet {num_plot_cls}, hat {len(getattr(label_encoder, 'classes_', [])) }). Generische Namen.")
                    for i in range(num_plot_cls):
                        y_true_ovr = (best_true_val == i).astype(int)
                        if len(np.unique(y_true_ovr)) < 2: log.info(f"ROC für Klasse '{str(plot_cls_names[i])}' (Idx {i}) nicht geplottet: Benötigt pos/neg Samples in Val-Daten."); continue
                        RocCurveDisplay.from_predictions(y_true_ovr, best_probs_val[:, i], name=f"ROC Klasse '{str(plot_cls_names[i])}' (idx {i})", ax=ax_roc)
                    ax_roc.plot([0, 1], [0, 1], 'k--', label='Chance (AUC=0.5)'); ax_roc.axis('square'); ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate'); ax_roc.set_title(f"ROC Curves (Best Model F1-w: {best_f1_w:.4f})"); ax_roc.legend(loc="lower right", fontsize='small'); fig_roc.tight_layout(); fig_roc.savefig(roc_plot_path); plt.close(fig_roc); log.info(f"ROC-Plot gespeichert: {roc_plot_path}")
                except Exception as e_roc_plot: log.error(f"ROC Plot Erstellung fehlgeschlagen: {e_roc_plot}\n{traceback.format_exc()}"); roc_plot_path = "n/a (PlotError)"
            else: log.warning(f"Bedingungen für ROC-Plot nicht erfüllt. Validierungsdaten: True Labels unique {len(np.unique(best_true_val)) if best_true_val is not None else 'None'}, Probs shape {best_probs_val.shape if best_probs_val is not None else 'None'}."); roc_plot_path = "n/a (NoValDataForROC)"
        else: log.warning("Kein bestes Modell während des Trainings gespeichert. Keine Modell-Artefakte erstellt.")

        # --- 7. Evaluierung des besten PyTorch-Modells ---
        torch_eval_res = {}
        pytorch_eval_cfg_params = eval_cfg.get("pytorch_model_eval", {})
        run_pytorch_eval = pytorch_eval_cfg_params.get("run_eval", True)
        if run_pytorch_eval and best_model_state:
            pytorch_eval_n_samples = pytorch_eval_cfg_params.get("n_samples", None);
            eval_model = QATPrunedSimpleNet(input_size, num_classes, n_hidden, qlinear_args_config=qlinear_args, qidentity_args_config=qidentity_args, qrelu_args_config=qrelu_args, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2).to(device)
            if unpruned_neurons > 0 and input_size > unpruned_neurons: eval_model.prune(max_non_zero_per_neuron=unpruned_neurons)
            eval_model.load_state_dict(best_model_state)
            torch_eval_res = evaluate_torch_model(eval_model, test_loader, device, label_encoder, model_name=f"{artifact_file_prefix}_pytorch_eval", results_dir=current_run_results_dir, n_samples=pytorch_eval_n_samples, logger=log)
            if wandb_mode != "disabled" and wandb.run and torch_eval_res:
                wandb.summary["pytorch_eval_accuracy"] = torch_eval_res.get("accuracy")
                wandb.summary["pytorch_eval_f1_weighted"] = torch_eval_res.get("f1_weighted")
                wandb.summary["pytorch_eval_f1_macro"] = torch_eval_res.get("f1_macro")
                wandb.summary["pytorch_eval_roc_auc"] = torch_eval_res.get("roc_auc_macro_ovr")
                wandb.summary["pytorch_eval_inference_time_s"] = torch_eval_res.get("total_inference_time_s")
                wandb.summary["pytorch_eval_time_per_1000_samples_s"] = torch_eval_res.get("time_per_1000_samples_s")
        elif not run_pytorch_eval: log.info("PyTorch Modell-Evaluierung übersprungen (run_eval=false).")
        else: log.warning("Kein bestes PyTorch-Modell zum Evaluieren vorhanden.")

        # --- 8. FHE-Modell Kompilierung und Evaluierung ---
        quant_numpy_module = None
        fhe_eval_cfg_params = eval_cfg.get("fhe_model_eval", {})
        all_fhe_eval_results: Dict[str, Any] = {}
        fhe_compile_time_log: Union[str, float] = "N/A (Skipped or Not Run)"
        run_fhe_pipeline = fhe_eval_cfg_params.get("run_fhe_pipeline", True)
        run_fhe_eval_overall_flag = fhe_eval_cfg_params.get("run_fhe_eval", True)

        if run_fhe_pipeline and best_model_state:
            log.info("Vorbereitung des besten Modells für FHE Kompilierung...")
            fhe_model = QATPrunedSimpleNet(input_size, num_classes, n_hidden, qlinear_args_config=qlinear_args, qidentity_args_config=qidentity_args, qrelu_args_config=qrelu_args, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2)
            if unpruned_neurons > 0 and input_size > unpruned_neurons: fhe_model.prune(max_non_zero_per_neuron=unpruned_neurons)
            fhe_model.load_state_dict(best_model_state); fhe_model.cpu()
            if unpruned_neurons > 0 and input_size > unpruned_neurons: log.info("Unpruning FHE model instance before compilation."); fhe_model.unprune()
            compile_sample_size = min(fhe_eval_cfg_params.get("compilation_sample_size", 128), len(X_train_np))
            
            if compile_sample_size == 0 and len(X_train_np) > 0: compile_sample_size = 1
            elif len(X_train_np) == 0:
                log.error("Keine Trainingsdaten für FHE-Kompilierungs-Sample. Überspringe FHE.")
                compile_sample_size = 0; run_fhe_eval_overall_flag = False; fhe_compile_time_log = "N/A (No Compile Sample)"; quant_numpy_module = None
            
            if compile_sample_size > 0:
                compile_data = torch.tensor(X_train_np[:compile_sample_size], dtype=torch.float32).cpu(); log.info(f"Starte FHE Kompilierung mit {compile_sample_size} Samples...")
                start_compile_t = time.time()
                try:
                    quant_numpy_module = compile_brevitas_qat_model(fhe_model, compile_data)
                    fhe_compile_time_log = round(time.time() - start_compile_t, 2)
                    log.info(f"FHE Kompilierung erfolgreich. Dauer: {fhe_compile_time_log}s")
                    if wandb_mode != "disabled" and wandb.run: wandb.summary["fhe_compilation_time_s"] = fhe_compile_time_log
                except Exception as e_fhe_compile:
                    fhe_compile_time_log = "N/A (Failed)"; log.error(f"Fehler bei FHE Kompilierung: {e_fhe_compile}\n{traceback.format_exc()}"); quant_numpy_module = None
                    if wandb_mode != "disabled" and wandb.run: wandb.summary["fhe_compilation_status"] = "Failed"
            
            if quant_numpy_module and run_fhe_eval_overall_flag:
                fhe_modes_to_evaluate = ["simulate", "execute"]
                for fhe_mode_to_run in fhe_modes_to_evaluate:
                    n_samples_key = f"n_samples_{fhe_mode_to_run}"
                    fhe_n_samples_to_run = fhe_eval_cfg_params.get(n_samples_key, 0) 
                    if fhe_n_samples_to_run is None or fhe_n_samples_to_run <= 0:
                        log.info(f"FHE Modell-Evaluierung für Modus '{fhe_mode_to_run}' übersprungen (n_samples nicht > 0).")
                        all_fhe_eval_results[fhe_mode_to_run] = {"status": "skipped", "reason": "n_samples_not_positive"}
                        continue
                    log.info(f"Starte FHE Modell-Evaluierung (Modus '{fhe_mode_to_run}', {fhe_n_samples_to_run} Samples)...")
                    current_mode_results = evaluate_fhe_model(quant_numpy_module, raw_data["X_test"], raw_data["y_test"],label_encoder, model_name=f"{artifact_file_prefix}_fhe_eval_{fhe_mode_to_run}", results_dir=current_run_results_dir, fhe_mode=fhe_mode_to_run, n_samples=fhe_n_samples_to_run, logger=log, fhe_compilation_time_s=fhe_compile_time_log)
                    all_fhe_eval_results[fhe_mode_to_run] = current_mode_results
                    if wandb_mode != "disabled" and wandb.run and current_mode_results:
                        wandb.summary[f"fhe_eval_{fhe_mode_to_run}_accuracy"] = current_mode_results.get("accuracy")
                        wandb.summary[f"fhe_eval_{fhe_mode_to_run}_f1_weighted"] = current_mode_results.get("f1_weighted")
                        wandb.summary[f"fhe_eval_{fhe_mode_to_run}_inference_time_s"] = current_mode_results.get("total_inference_time_s")
                        wandb.summary[f"fhe_eval_{fhe_mode_to_run}_time_per_1000_samples_s"] = current_mode_results.get("time_per_1000_samples_s")
            elif not run_fhe_eval_overall_flag and run_fhe_pipeline: log.info("FHE Modell-Evaluierung übersprungen (run_fhe_eval=false in config).")
            elif not quant_numpy_module and compile_sample_size > 0 : log.warning("FHE Kompilierung nicht erfolgreich. FHE Evaluierung wird nicht durchgeführt.")
        elif not run_fhe_pipeline: log.info("FHE Kompilierung und Evaluierung übersprungen (run_fhe_pipeline=false).")
        else: log.warning("Kein bestes Modell für FHE Kompilierung und Evaluierung vorhanden.")

        # --- 9. Zusammenfassendes Logging (JSON) ---
        eff_batch_size = dataloader_batch_size if use_train_loader else manual_batch_size
        batch_src_param = ("data_params.dataloader_batch_size" if use_train_loader else "training_params.manual_batch_size")
        log_train_params = train_cfg.copy()
        if use_train_loader:
            log_train_params['INFO_eff_dataloader_batch_size'] = dataloader_batch_size
            if 'manual_batch_size' in log_train_params: log_train_params['INFO_manual_batch_size_ignored'] = log_train_params['manual_batch_size']
        crit_weights_log = "N/A"
        if hasattr(criterion, 'weight') and criterion.weight is not None:
            try: crit_weights_log = criterion.weight.cpu().tolist()
            except Exception: crit_weights_log = "Error converting weights to list"
        val_loss_at_best_f1w = "N/A"
        if best_epoch != -1 and best_epoch < len(val_losses): val_loss_at_best_f1w = round(val_losses[best_epoch], 5)

        log_data = {
            "run_overview": {"run_timestamp": run_timestamp, "dataset_configured": run_cfg.get("dataset_name"), "model_type_configured": model_cfg.get("type"), "device_actually_used": str(device), "config_file_path_source": config_path},
            "input_configuration_from_yaml": {"run_settings": run_cfg, "data_params": data_cfg, "model_params": model_cfg, "training_params": log_train_params, "evaluation_params": eval_cfg},
            "data_summary_runtime": {"npz_file_path_used": npz_file_path, "input_features_detected": input_size, "output_classes_detected": num_classes, "train_samples_processed": len(X_train_t), "val_samples_processed": len(X_val_t), "test_samples_in_dataset": len(raw_data["X_test"])},
            "model_details_runtime": {"layer_structure_generated": layer_structure, "dropout_rate1_applied": dropout_rate1, "dropout_rate2_applied": dropout_rate2, "quantization_bits_applied": cfg_qb_early, "unpruned_neurons_applied": cfg_up_early, "n_hidden_applied": cfg_h_early},
            "training_execution_summary": {
                "epochs_run_actual": actual_epochs,
                "early_stopping_details_runtime": {"patience_used": patience, "min_delta_used": min_delta, "triggered": (actual_epochs < num_epochs and patience_counter_early_stop >= patience), "stopped_at_epoch_if_triggered": (actual_epochs if (actual_epochs < num_epochs and patience_counter_early_stop >= patience) else None)},
                "criterion_details_runtime": {"name_runtime": type(criterion).__name__, "reduction_runtime": getattr(criterion, "reduction", "N/A"), "focal_gamma_runtime": (getattr(criterion, "gamma", "N/A") if isinstance(criterion, FocalLoss) else "N/A"), "focal_alpha_runtime": (getattr(criterion, "alpha", "N/A") if isinstance(criterion, FocalLoss) else "N/A"), "class_weights_applied_runtime": ("yes" if isinstance(crit_weights_log, list) else "no"), "class_weights_tensor_values": crit_weights_log},
                "optimizer_details_runtime": {"name_runtime": type(optimizer).__name__, "initial_learning_rate_runtime": learning_rate, "weight_decay_runtime": optimizer.param_groups[0].get('weight_decay',"N/A"), "momentum_runtime": optimizer.param_groups[0].get('momentum',"N/A")},
                "scheduler_details_runtime": {"name_runtime": (type(scheduler).__name__ if scheduler is not None else "None"), "factor_runtime": (getattr(scheduler,'factor',"N/A") if hasattr(scheduler,'factor') else "N/A"), "patience_runtime": (getattr(scheduler,'patience',"N/A") if hasattr(scheduler,'patience') else "N/A"), "min_lr_runtime": (getattr(scheduler,'min_lrs',[{"N/A"}])[0] if hasattr(scheduler,'min_lrs') and scheduler.min_lrs else "N/A"), "step_size_runtime": (getattr(scheduler,'step_size',"N/A") if hasattr(scheduler,'step_size') else "N/A"), "gamma_runtime": (getattr(scheduler,'gamma',"N/A") if hasattr(scheduler,'gamma') else "N/A"), "last_lr_runtime": lr_history[-1] if lr_history else learning_rate},
                "batching_details_runtime": {"mode_used_for_training": ("DataLoader" if use_train_loader else "ManualTensorSlicing"), "batch_size_parameter_source_in_config": batch_src_param, "effective_batch_size_applied": eff_batch_size}
            },
            "evaluation_execution_details": {
                "pytorch_eval_run": (pytorch_eval_cfg_params.get("run_eval", True) and bool(best_model_state)),
                "pytorch_eval_summary": torch_eval_res,
                "fhe_pipeline_run": (fhe_eval_cfg_params.get("run_fhe_pipeline", True) and bool(best_model_state)),
                "fhe_compilation_time_s": fhe_compile_time_log,
                "fhe_compilation_status": ("Success" if isinstance(fhe_compile_time_log, (int, float)) else fhe_compile_time_log),
                "fhe_evaluations": all_fhe_eval_results
            },
            "best_model_metrics_achieved_val": {
                "best_f1_weighted_val": (round(best_f1_w, 5) if isinstance(best_f1_w, float) else best_f1_w),
                "f1_macro_at_best_f1w_val": (round(best_f1_m_at_best_f1_w, 5) if isinstance(best_f1_m_at_best_f1_w, float) else best_f1_m_at_best_f1_w),
                "roc_auc_at_best_f1w_val": (round(best_roc_auc_at_best_f1_w, 5) if isinstance(best_roc_auc_at_best_f1_w, float) else best_roc_auc_at_best_f1_w),
                "val_loss_at_best_f1w_epoch": val_loss_at_best_f1w,
                "epoch_of_best_f1w": best_epoch + 1 if best_epoch != -1 else "N/A"
            },
            "output_artifact_paths": {"results_run_directory": current_run_results_dir, "model_file_saved": model_path, "loss_plot_file_saved": loss_plot_path, "roc_plot_file_saved": roc_plot_path, "detailed_text_log": detailed_log_path},
            "pruning_log_details": {"initial_sparsity": initial_pruning_summary, "final_sparsity": final_pruning_summary, "pruned_layers_list": (list(torch_model.pruned_layers) if hasattr(torch_model, 'pruned_layers') and torch_model.pruned_layers else [])},
            "epoch_history_data": {"train_loss_per_epoch": [round(l_val, 5) for l_val in train_losses], "val_loss_per_epoch": [round(l_val, 5) for l_val in val_losses], "lr_per_epoch": lr_history}
        }

        json_log_name = f"{run_timestamp}_{artifact_file_prefix}_full_run_log.json"
        json_log_path = os.path.join(current_run_results_dir, json_log_name)
        log_data["output_artifact_paths"]["json_log_file_self"] = json_log_path

        def default_json_converter(obj: Any) -> Any:
            """Konvertiert nicht-standard JSON-Typen für json.dump."""
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.bool_): return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, torch.Tensor): return obj.cpu().detach().tolist()
            if isinstance(obj, set): return list(obj)
            if isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
            if hasattr(obj, '__dict__'):
                try: return obj.__dict__
                except TypeError: return str(obj)
            if isinstance(obj, (optim.lr_scheduler.ReduceLROnPlateau, optim.AdamW, FocalLoss, nn.CrossEntropyLoss)): return f"Object<{obj.__class__.__name__}>"
            if callable(obj): return f"Callable<{obj.__name__ if hasattr(obj, '__name__') else str(obj)}>"
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    f"Objekt vom Typ {type(obj).__name__} nicht direkt JSON "
                    f"serialisierbar. Fallback zu str(). Wert: {str(obj)[:100]}"
                )
            return str(obj)

        try:
            with open(json_log_path, "w") as f_json:
                json.dump(log_data, f_json, indent=4, default=default_json_converter)
            log.info(f"JSON-Log gespeichert unter: {json_log_path}")
        except Exception as e_json_save:
            log.error(
                f"Fehler beim Speichern des JSON-Logs: {e_json_save}\n"
                f"{traceback.format_exc()}"
            )

        log_training_summary(
            log, model_path, best_f1_w, loss_plot_path, roc_plot_path,
            json_log_path
        )
        log.info(f"QAT Pipeline für '{artifact_file_prefix}' abgeschlossen.")

        # *** WANDB ARTEFAKTE LOGGEN (optional) ***
        if wandb_mode != "disabled" and wandb.run and wandb_cfg.get("log_artifacts", True):
            log.info("Logging Artefakte zu Weights & Biases...")
            try:
                if os.path.exists(model_path) and model_path != "n/a":
                    model_artifact = wandb.Artifact(f"{artifact_file_prefix}_model", type="model")
                    model_artifact.add_file(model_path)
                    wandb.log_artifact(model_artifact); log.info(f"Modell-Artefakt '{model_artifact.name}' geloggt.")
                if os.path.exists(loss_plot_path) and loss_plot_path != "n/a": wandb.log({"loss_plot": wandb.Image(loss_plot_path)})
                if os.path.exists(roc_plot_path) and roc_plot_path != "n/a" and not roc_plot_path.startswith("n/a ("): wandb.log({"roc_plot": wandb.Image(roc_plot_path)})
                
                log_files_artifact = wandb.Artifact(f"{artifact_file_prefix}_run_files", type="run_files")
                if os.path.exists(json_log_path): log_files_artifact.add_file(json_log_path, name="run_summary.json")
                if os.path.exists(detailed_log_path) and detailed_log_path != "n/a": log_files_artifact.add_file(detailed_log_path, name="pipeline_full_output.log")
                
                pytorch_eval_txt_filename = f"eval_torch_{artifact_file_prefix}_pytorch_eval.txt"
                pytorch_eval_txt_path = os.path.join(current_run_results_dir, pytorch_eval_txt_filename)
                if os.path.exists(pytorch_eval_txt_path):
                    log_files_artifact.add_file(pytorch_eval_txt_path, name="pytorch_eval_report.txt")
                
                for mode, fhe_res_data in all_fhe_eval_results.items():
                    if fhe_res_data.get("status") != "skipped" and "total_inference_time_s" in fhe_res_data :
                        fhe_eval_txt_filename = f"eval_fhe_{artifact_file_prefix}_fhe_eval_{mode}.txt"
                        fhe_eval_txt_path = os.path.join(current_run_results_dir, fhe_eval_txt_filename)
                        if os.path.exists(fhe_eval_txt_path):
                             log_files_artifact.add_file(fhe_eval_txt_path, name=f"fhe_{mode}_eval_report.txt")
                
                if len(log_files_artifact.files) > 0 : wandb.log_artifact(log_files_artifact); log.info(f"Run-Files-Artefakt '{log_files_artifact.name}' geloggt.")

            except Exception as e_wandb_artifact:
                log.error(f"Fehler beim Loggen von Artefakten zu Weights & Biases: {e_wandb_artifact}")
        # *** ENDE WANDB ARTEFAKTE LOGGEN ***

    finally: # Sicherstellen, dass Handler und wandb-Lauf geschlossen werden
        if detailed_log_file_handler:
            log.info(f"Schließe detailliertes Text-Log: {detailed_log_path}")
            log.removeHandler(detailed_log_file_handler)
            detailed_log_file_handler.close()
            detailed_log_file_handler = None
        
        if wandb_mode != "disabled" and wandb.run:
            exit_code = 0 if best_model_state else 1
            log.info(f"Beende Weights & Biases Lauf (Exit-Code: {exit_code}).")
            wandb.finish(exit_code=exit_code)

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
    project_root = os.path.dirname(os.path.dirname(script_dir))

    default_cfg_name = "config.yaml"
    config_path_for_cli = os.path.join(project_root, default_cfg_name)

    default_npz_filename = "edgeiiot_dataset_all.npz"
    npz_path_for_dummy_config_default = os.path.join(
        project_root, "data", "processed", default_npz_filename
    )
    npz_path_for_dummy_config = npz_path_for_dummy_config_default

    main_loaded_config: Dict[str, Any] = {}
    if os.path.exists(config_path_for_cli):
        try:
            with open(config_path_for_cli, "r") as f_base_config:
                main_loaded_config = yaml.safe_load(f_base_config) or {}
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
                    f"Kein NPZ-Pfad in Config '{config_path_for_cli}'. "
                    f"Verwende Default: '{npz_path_for_dummy_config_default}'"
                )
        except Exception as e_load_main_cfg:
            cli_logger.error(
                f"Fehler beim Laden der Basis-Config '{config_path_for_cli}': {e_load_main_cfg}. "
                "Verwende Default NPZ-Pfad."
            )
            main_loaded_config = {}
    else:
        cli_logger.info(
            f"Haupt-Config '{config_path_for_cli}' nicht gefunden. "
            "Verwende Default NPZ-Pfad für Dummy-Config."
        )
        main_loaded_config = {}

    config_to_run_with = main_loaded_config
    if not os.path.exists(config_path_for_cli) or not main_loaded_config:
        cli_logger.warning(
            f"Erstelle temporäre Test-Konfigurationsdatei, da "
            f"'{config_path_for_cli}' nicht gefunden oder leer war."
        )
        temp_config_filename = "temp_config_for_cli.yaml"
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

        dummy_config_content = {
            "run_settings": {
                "dataset_name": "EdgeIIoT_CLI_Temp",
                "results_base_dir": dummy_results_base_dir,
                "device_str": "auto",
                "logger_instance_name": "QAT_Pipeline_TempCfg",
                "logger_level": "DEBUG"
            },
            "data_params": {
                "npz_file_path": npz_path_for_dummy_config,
                "dataloader_batch_size": 64,
                "use_subset_training": True, "subset_fraction": 0.002,
                "refit_scaler": False, "refit_encoder": False
            },
            "model_params": {
                "type": "QATPrunedSimpleNet", "n_hidden": 16,
                "quantization_bits": 2, "unpruned_neurons": 4,
                "dropout": {"rate1": 0.0, "rate2": 0.0},
                "qlinear_bias": True,
                "qidentity_return_quant_tensor": True,
                "qrelu_return_quant_tensor": True
            },
            "training_params": {
                "num_epochs": 1, "manual_batch_size": 16,
                "use_train_loader_for_batches": True,
                "learning_rate": 0.001,
                "criterion": {
                    "name": "FocalLoss", "calculate_class_weights": True,
                    "class_weight_calculation_method": "inverse_frequency",
                    "focal_loss_alpha": 0.25, "focal_loss_gamma": 2.0,
                    "reduction": "mean"
                },
                "optimizer": {"name": "AdamW", "weight_decay": 0.01},
                "scheduler": {"name": "null", "params": {}},
                "early_stopping": {"patience": 1, "min_delta": 0.01}
            },
            "evaluation_params": {
                "pytorch_model_eval": {"run_eval": True, "n_samples": 20},
                "fhe_model_eval": {
                    "run_fhe_pipeline": False,
                    "compilation_sample_size": 16,
                    "run_fhe_eval": False, "mode": "simulate",
                    "n_samples_simulate": 10, "n_samples_execute": 1
                }
            },
            "wandb_settings": {
                "project_name": "CLI_Dummy_Test_Project",
                "entity": None,
                "run_name_prefix": "cli_dummy_run",
                "log_artifacts": False,
                "mode": "disabled"
            }
        }
        try:
            os.makedirs(dummy_results_base_dir, exist_ok=True)
            with open(config_path_for_cli, 'w') as cf_temp_write:
                yaml.dump(dummy_cfg_content, cf_temp_write, indent=2, sort_keys=False)
            cli_logger.info(
                f"Temporäre Konfigurationsdatei '{config_path_for_cli}' erstellt."
            )
            config_to_run_with = dummy_cfg_content
        except Exception as e_yaml_write_cli:
            cli_logger.error(
                "Konnte temporäre Konfigurationsdatei nicht schreiben: "
                f"{e_yaml_write_cli}"
            )
            exit(1)
    else:
        config_to_run_with = main_loaded_config

    final_npz_path_check = config_to_run_with.get("data_params", {}).get(
        "npz_file_path"
    )
    if final_npz_path_check:
        path_to_verify = final_npz_path_check
        if not os.path.isabs(path_to_verify):
            path_to_verify = os.path.join(project_root, path_to_verify)
        if not os.path.exists(path_to_verify):
             cli_logger.error(
                f"KRITISCH: Die endgültig konfigurierte NPZ-Datendatei "
                f"'{path_to_verify}' (abgeleitet von "
                f"'{final_npz_path_check}') wurde nicht gefunden."
            )
    elif not final_npz_path_check :
            cli_logger.error(
                "KRITISCH: Kein NPZ-Pfad ('npz_file_path') in der endgültigen "
                "Konfiguration gefunden."
            )

    cli_logger.info(
        f"Starte QAT Trainingspipeline als Skript mit Config: {config_path_for_cli}"
    )
    try:
        run_qat_training_pipeline(config_path=config_path_for_cli)
    except FileNotFoundError as e_fnf_in_main:
        cli_logger.error(
            f"FileNotFoundError: {e_fnf_in_main}. Sicherstellen, dass Pfade "
            "(insb. NPZ) korrekt sind."
        )
        cli_logger.error(traceback.format_exc())
    except Exception as e_main_exc:
        cli_logger.error(
            "Unerwarteter Fehler während der Pipeline-Ausführung: "
            f"{e_main_exc}"
        )
        cli_logger.error(traceback.format_exc())
    cli_logger.info("QAT Trainingspipeline Skriptlauf beendet.")

