#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_fhe_concrete_models.py # Beachten Sie, dass der Dateiname im Kommentar nicht mit Ihrem Aufruf übereinstimmt.
                           # Ich gehe davon aus, dass Ihr tatsächlicher Dateiname train_fhe_transformer.py ist.

Dieses Skript trainiert und evaluiert ein FHETabularTransformer-Modell
unter Verwendung von Concrete ML für Fully Homomorphic Encryption (FHE).
Es beinhaltet das Laden von Daten, Training, Validierung, FHE-Kompilierung und -Evaluierung.
"""

import os
import yaml
import json
import time
import traceback
import logging
from datetime import datetime
from typing import Callable, Dict, Tuple, Any, Optional, List 
import argparse

import numpy as np
import pandas as pd 
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder 
from torch.utils.data import DataLoader, TensorDataset 
from tqdm import tqdm 

# Eigene Modul-Importe aus Ihrem Projekt
from src.data.edge_iiot_dataset import load_edgeiiot_data 
from src.models.fhe_tabular_transformer import FHETabularTransformer 
from src.evaluation.concrete_evaluate import (
    _calculate_robust_roc_auc, 
    evaluate_fhe_model,      
    evaluate_torch_model,    
)
from src.training.custom_losses import FocalLoss 
from src.utils.config_loader import load_config 
from src.utils.logger import (
    setup_logger,          
    log_training_epoch,    
    log_training_summary   
)

# Brevitas und Concrete ML für Kompilierung
import brevitas.nn as qnn 
from concrete.ml.torch.compile import compile_brevitas_qat_model 
from concrete.ml.torch.hybrid_model import HybridFHEModel 


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__) 

FHE_FRIENDLY_ACTIVATION = qnn.QuantReLU 


def _train_transformer_epoch(
    model: FHETabularTransformer, 
    data_loader: DataLoader, 
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_num: int,
    total_epochs: int,
    logger: logging.Logger 
) -> float:
    """Führt eine Trainingsepoche für das Transformer-Modell durch."""
    model.train()
    running_loss = 0.0
    num_batches_processed = 0

    progress_bar = tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=f"Epoch {epoch_num + 1}/{total_epochs} [T]",
        leave=False,
    )

    for _, batch_data in progress_bar:
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
    
    if num_batches_processed > 0:
        return running_loss / num_batches_processed
    else:
        logger.warning("Keine Batches in Trainingsepoche verarbeitet.")
        return 0.0


def _validate_transformer_epoch(
    model: FHETabularTransformer, 
    data_loader: DataLoader, 
    criterion: nn.Module,
    device: torch.device,
    epoch_num: int, 
    logger: logging.Logger, 
    label_encoder: Optional[LabelEncoder] = None 
) -> Dict[str, Any]:
    """Führt eine Validierungsepoche für das Transformer-Modell durch."""
    model.eval()
    running_val_loss = 0.0
    all_labels_val = []
    all_outputs_val = [] 

    with torch.no_grad():
        for batch_data in data_loader:
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * inputs.size(0) 

            all_labels_val.append(labels.cpu())
            all_outputs_val.append(outputs.cpu())
    
    dataset_len = len(data_loader.dataset) if data_loader.dataset is not None else 0 # type: ignore
    avg_val_loss = running_val_loss / dataset_len if dataset_len > 0 else 0

    true_labels_np = torch.cat(all_labels_val).numpy()
    outputs_np = torch.cat(all_outputs_val).numpy()
    
    probs_np = torch.softmax(torch.tensor(outputs_np), dim=1).numpy()
    preds_np = np.argmax(probs_np, axis=1)
    
    from sklearn.metrics import f1_score as sk_f1_score 

    metric_labels_unique = np.unique(np.concatenate((true_labels_np, preds_np)))
    f1_w, f1_m = 0.0, 0.0 
    if len(metric_labels_unique) == 0: 
        logger.warning(f"Epoche {epoch_num+1} Val: Keine Labels für Metrikberechnung.")
    elif len(metric_labels_unique) == 1: 
        f1_w = sk_f1_score(true_labels_np, preds_np, labels=metric_labels_unique, average="weighted", zero_division=0)
        f1_m = sk_f1_score(true_labels_np, preds_np, labels=metric_labels_unique, average="macro", zero_division=0)
    else: 
        f1_w = sk_f1_score(true_labels_np, preds_np, labels=metric_labels_unique, average="weighted", zero_division=0)
        f1_m = sk_f1_score(true_labels_np, preds_np, labels=metric_labels_unique, average="macro", zero_division=0)

    roc_auc_m = "N/A"
    if label_encoder is not None and true_labels_np.size > 0 and probs_np.size > 0 : 
         roc_auc_m = _calculate_robust_roc_auc(true_labels_np, probs_np, logger, 
                                             context_msg=f"Ep{epoch_num+1} Val")
    else:
        logger.warning(f"Epoche {epoch_num+1} Val: LabelEncoder oder Daten nicht für ROC AUC Berechnung verfügbar.")

    return {
        "avg_val_loss": avg_val_loss, "f1_weighted": f1_w, "f1_macro": f1_m,
        "roc_auc_macro": roc_auc_m, "true_labels": true_labels_np, 
        "probabilities": probs_np     
    }

def run_transformer_training_pipeline(config_path: str) -> Optional[str]:
    """
    Führt die gesamte Trainingspipeline für das FHETabularTransformer-Modell aus.
    """
    config = load_config(config_path)
    if not config:
        print(f"FEHLER: Konnte Config von '{config_path}' nicht laden. Pipeline-Abbruch.")
        return None

    run_cfg = config.get("run_settings", {})
    data_cfg = config.get("data_params", {})
    model_cfg = config.get("transformer_model_params", {}) 
    train_cfg = config.get("training_params", {})
    eval_cfg = config.get("evaluation_params", {}) 

    logger_name = run_cfg.get("logger_instance_name", "transformer_pipeline_run")
    log_level = run_cfg.get("logger_level", "INFO").upper()
    log = setup_logger(name=logger_name, level=log_level) 

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = run_cfg.get("dataset_name", "UnknownDataset")
    
    results_base_dir_from_runner = run_cfg.get("results_base_dir")
    if not results_base_dir_from_runner:
        log.error("FEHLER: 'results_base_dir' wurde nicht vom Experiment Runner in der Config gesetzt! Abbruch.")
        return None
    try:
        os.makedirs(results_base_dir_from_runner, exist_ok=True)
    except OSError as e:
        log.error(f"FEHLER beim Erstellen des Ergebnisordners '{results_base_dir_from_runner}': {e}. Abbruch.")
        return None
        
    d_model = model_cfg.get("d_model", 64)
    n_heads = model_cfg.get("n_heads", 4)
    num_enc_layers = model_cfg.get("num_encoder_layers", 2)
    base_artifact_name = f"FHETabularTransformer_{dataset_name}_dm{d_model}_h{n_heads}_l{num_enc_layers}"

    detailed_log_filename = f"{run_timestamp}_{base_artifact_name}_pipeline_full.log"
    detailed_log_path = os.path.join(results_base_dir_from_runner, detailed_log_filename)
    detailed_log_file_handler: Optional[logging.FileHandler] = None # Für korrekten Scope
    try:
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S,%03d")
        
        for handler in list(log.handlers):
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == detailed_log_path: # Spezifischer entfernen
                log.removeHandler(handler)
                handler.close()
                
        detailed_log_file_handler = logging.FileHandler(detailed_log_path, mode="w")
        detailed_log_file_handler.setFormatter(file_formatter)
        detailed_log_file_handler.setLevel(log.getEffectiveLevel()) 
        log.addHandler(detailed_log_file_handler)
        log.info(f"Detailliertes Text-Log wird nach '{detailed_log_path}' geschrieben.")
    except Exception as e_log_setup:
        log.error(f"Fehler beim Setup des detaillierten File-Loggers: {e_log_setup}")
        # Fahre fort, aber ohne File-Logging für dieses Detail-Log

    device_str = run_cfg.get("device_str", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() and device_str != "cpu" else "cpu")
    log.info(f"Verwende Device: {device}")

    npz_file_path = data_cfg.get("npz_file_path")
    if not npz_file_path:
        log.error("FEHLER: 'npz_file_path' nicht in data_params. Abbruch.")
        if detailed_log_file_handler: detailed_log_file_handler.close(); log.removeHandler(detailed_log_file_handler)
        return None

    log.info(f"Lade Daten für Dataset: {dataset_name} von {npz_file_path}")
    try:
        loaded_data_output = load_edgeiiot_data( 
            npz_path=npz_file_path,
            batch_size=train_cfg.get("dataloader_batch_size", 64),
            return_raw_data=True, 
            logger=log
        )
        if loaded_data_output is None or not isinstance(loaded_data_output, tuple) or len(loaded_data_output) != 4 or loaded_data_output[3] is None:
             raise ValueError("Fehler beim Laden der Daten: load_edgeiiot_data gab ungültige Daten zurück.")
        
        train_loader, val_loader, test_loader, raw_data_map = loaded_data_output

        num_features = raw_data_map["num_features"] # JETZT DEFINIERT
        num_classes = raw_data_map["num_classes"]   # JETZT DEFINIERT
        label_encoder = raw_data_map["label_encoder"] 
        
        x_train_for_sample = raw_data_map["X_train"]
        if x_train_for_sample is None or len(x_train_for_sample) == 0:
            log.error("X_train ist leer oder None. FHE Kompilierung nicht möglich.")
            if detailed_log_file_handler: detailed_log_file_handler.close(); log.removeHandler(detailed_log_file_handler)
            return None
        X_train_sample_for_fhe = torch.tensor(x_train_for_sample[:min(100, len(x_train_for_sample))], dtype=torch.float32)

    except Exception as e_data:
        log.error(f"Fehler beim Laden oder Vorbereiten der Daten: {e_data}")
        log.error(traceback.format_exc())
        if detailed_log_file_handler: detailed_log_file_handler.close(); log.removeHandler(detailed_log_file_handler)
        return None

    log.info(f"Initialisiere FHETabularTransformer Modell mit Parametern: {model_cfg}")
    quant_linear_module_name = model_cfg.get("quant_linear_type", "QuantLinear")
    quant_linear_module = getattr(qnn, quant_linear_module_name, getattr(nn, quant_linear_module_name, nn.Linear))
    
    activation_str = model_cfg.get("activation_type_torch", "ReLU")
    activation_module_resolved = getattr(nn, activation_str, getattr(qnn, activation_str, qnn.QuantReLU))

    model = FHETabularTransformer(
        num_features=num_features,  
        num_classes=num_classes,     
        d_model=model_cfg.get("d_model", 128),
        n_heads=model_cfg.get("n_heads", 4),
        num_encoder_layers=model_cfg.get("num_encoder_layers", 3),
        ffn_dim=model_cfg.get("ffn_dim", 256),
        dropout_rate=model_cfg.get("dropout_rate", 0.1),
        quant_linear_layer_name=quant_linear_module_name, 
        activation_module_name=activation_str,          
        weight_bit_width=model_cfg.get("weight_bit_width"), 
        activation_bit_width=model_cfg.get("activation_bit_width")
    ).to(device)

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.get("learning_rate", 0.001))
    log.info(f"Verwende Criterion: {type(criterion).__name__}, Optimizer: {type(optimizer).__name__}")
    scheduler = None 
    scheduler_name_cfg = train_cfg.get("scheduler", {}).get("name")
    if scheduler_name_cfg and scheduler_name_cfg.lower() == "reducelronplateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', 
            factor=train_cfg.get("scheduler",{}).get("reduce_lr_factor", 0.1),
            patience=train_cfg.get("scheduler",{}).get("reduce_lr_patience", 10),
            min_lr=train_cfg.get("scheduler",{}).get("reduce_lr_min_lr", 1e-6),
            verbose=train_cfg.get("scheduler",{}).get("reduce_lr_verbose",True) # verbose aus config
        )
        log.info("ReduceLROnPlateau Scheduler aktiviert.")

    num_epochs_total = train_cfg.get("num_epochs", 50)
    best_f1_w_val = 0.0
    actual_epochs_run_count = 0 
    best_model_state = None
    val_metrics_at_best_f1w: Dict[str, Any] = {}

    log.info(f"Starte Transformer Training für {num_epochs_total} Epochen...")
    for epoch in range(num_epochs_total):
        actual_epochs_run_count = epoch + 1
        epoch_start_time = time.time()
        avg_train_loss = _train_transformer_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs_total, log
        )
        val_results = _validate_transformer_epoch(
            model, val_loader, criterion, device, epoch, log, label_encoder
        )
        
        log_training_epoch(log, epoch, avg_train_loss, val_results["avg_val_loss"],
                           val_results["f1_weighted"], val_results["f1_macro"],
                           val_results["roc_auc_macro"], epoch_start_time)

        if val_results["f1_weighted"] > best_f1_w_val:
            best_f1_w_val = val_results["f1_weighted"]
            best_model_state = model.state_dict()
            val_metrics_at_best_f1w = val_results.copy() 
            val_metrics_at_best_f1w["epoch_of_best_f1w"] = epoch + 1 
            log.info(f"Neues bestes Modell in Epoche {epoch+1} gefunden (F1-Weighted Val: {best_f1_w_val:.4f})")
        
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_results["avg_val_loss"])
            else:
                scheduler.step()
    
    # Stelle sicher, dass final_artifact_name auch definiert ist, wenn keine Verbesserung erfolgte
    epoch_for_name = val_metrics_at_best_f1w.get('epoch_of_best_f1w', actual_epochs_run_count)
    f1_for_name = best_f1_w_val if best_model_state else 0.0
    final_artifact_name = f"{base_artifact_name}_ep{epoch_for_name}_f1w{f1_for_name:.4f}".replace('.', '')
    
    model_save_path = "n/a"
    if best_model_state:
        model_save_path = os.path.join(results_base_dir_from_runner, f"{final_artifact_name}.pth")
        torch.save(best_model_state, model_save_path)
        log.info(f"Bestes PyTorch Modell gespeichert: {model_save_path}")
    else:
        log.warning("Kein bestes Modell zum Speichern gefunden (basierend auf F1-Weighted Val).")

    pytorch_eval_summary = {}
    model_for_eval_and_fhe = FHETabularTransformer( 
        num_features=num_features, num_classes=num_classes,
        d_model=model_cfg.get("d_model", 128), n_heads=model_cfg.get("n_heads", 4),
        num_encoder_layers=model_cfg.get("num_encoder_layers", 3),
        ffn_dim=model_cfg.get("ffn_dim", 256), dropout_rate=model_cfg.get("dropout_rate", 0.1),
        quant_linear_layer_name=quant_linear_module_name,
        activation_module_name=activation_str,
        weight_bit_width=model_cfg.get("weight_bit_width"),
        activation_bit_width=model_cfg.get("activation_bit_width")
    ).to(device) # Frische Instanz für FHE

    if best_model_state:
        model_for_eval_and_fhe.load_state_dict(best_model_state)
        log.info("Bestes Modell für PyTorch-Evaluierung und FHE-Kompilierung geladen.")
        pytorch_eval_summary = evaluate_torch_model(
            model_for_eval_and_fhe, test_loader, device, label_encoder,
            model_name=f"{final_artifact_name}_pytorch_eval", 
            results_dir=results_base_dir_from_runner, logger=log
        )
    else:
        log.warning("Kein bestes PyTorch-Modell vorhanden für Evaluierung und FHE-Kompilierung.")

    # src/training/train_fhe_transformer.py
# ... (Anfang der Funktion run_transformer_training_pipeline und PyTorch-Eval-Teil bleiben gleich) ...

    # --- NEU: FHE Kompilierung und Evaluierung mit HybridFHEModel ---
    fhe_simulation_summary = {} 
    fhe_execute_summary = {}    
    fhe_compilation_time_val = "N/A (Skipped or No Model)"
    
    fhe_model_eval_cfg = eval_cfg.get("fhe_model_eval", {})
    if best_model_state and fhe_model_eval_cfg.get("run_fhe_pipeline", False):
        log.info("Starte FHE Hybrid-Modell Kompilierung des besten Modells...")
        try:
            # Das Modell muss für die Kompilierung im eval-Modus sein
            # KORREKTUR: Verwende model_for_eval_and_fhe
            model_for_eval_and_fhe.eval() 
            model_for_eval_and_fhe.cpu() # Wichtig für HybridFHEModel Kompilierung mit CPU-Samples
            
            remote_names = []
            # KORREKTUR: Verwende model_for_eval_and_fhe
            for name, module in model_for_eval_and_fhe.named_modules():
                if isinstance(module, (qnn.QuantLinear, nn.Linear)): 
                    remote_names.append(name)
            
            if not remote_names:
                log.warning("Keine 'remote_names' im Modell für HybridFHEModel gefunden. Überspringe FHE-Teil.")
            else:
                log.info(f"Folgende {len(remote_names)} Module werden für FHE (remote) deklariert: {remote_names}")
                # KORREKTUR: Verwende model_for_eval_and_fhe
                hybrid_model = HybridFHEModel(model_for_eval_and_fhe, module_names=remote_names)
                log.info("HybridFHEModel Instanz erstellt.")

                fhe_quant_bits = fhe_model_eval_cfg.get("fhe_quantization_bits", 8) 
                use_dyn_quant = fhe_model_eval_cfg.get("fhe_use_dynamic_quantization", True)
                
                compilation_start_time = time.time()
                log.info(f"Starte hybrid_model.compile_model mit n_bits={fhe_quant_bits}, use_dynamic_quantization={use_dyn_quant}...")
                hybrid_model.compile_model(
                    X_train_sample_for_fhe.cpu(), 
                    n_bits=fhe_quant_bits,
                    use_dynamic_quantization=use_dyn_quant
                )
                fhe_compilation_time_val = round(time.time() - compilation_start_time, 2)
                log.info(f"HybridFHEModel Kompilierung erfolgreich. Dauer: {fhe_compilation_time_val}s")

                if fhe_model_eval_cfg.get("run_fhe_eval", False):
                    # ... (Rest der FHE-Evaluierung mit hybrid_model.model bleibt gleich) ...
                    log.info("Starte FHE-Evaluierung mit hybrid_model...")
                    
                    n_samples_simulate = fhe_model_eval_cfg.get("n_samples_simulate", 100)
                    if n_samples_simulate > 0:
                        log.info(f"Setze FHE-Modus auf 'simulate' für {n_samples_simulate} Samples.")
                        hybrid_model.set_fhe_mode("simulate")
                        fhe_simulation_summary = evaluate_torch_model(
                            hybrid_model.model, 
                            test_loader,        
                            device,             
                            label_encoder,
                            model_name=f"{final_artifact_name}_hybrid_fhe_simulate",
                            results_dir=results_base_dir_from_runner,
                            n_samples=n_samples_simulate, 
                            logger=log,
                            is_fhe_hybrid_eval=True 
                        )
                        if isinstance(fhe_simulation_summary, dict):
                            fhe_simulation_summary["fhe_compilation_time_s"] = fhe_compilation_time_val

                    n_samples_execute = fhe_model_eval_cfg.get("n_samples_execute", 0)
                    if n_samples_execute > 0:
                        log.info(f"Setze FHE-Modus auf 'execute' für {n_samples_execute} Samples.")
                        hybrid_model.set_fhe_mode("execute")
                        fhe_execute_summary = evaluate_torch_model(
                            hybrid_model.model,
                            test_loader,
                            device, 
                            label_encoder,
                            model_name=f"{final_artifact_name}_hybrid_fhe_execute",
                            results_dir=results_base_dir_from_runner,
                            n_samples=n_samples_execute,
                            logger=log,
                            is_fhe_hybrid_eval=True 
                        )
                        if isinstance(fhe_execute_summary, dict):
                             fhe_execute_summary["fhe_compilation_time_s"] = fhe_compilation_time_val
                    else:
                        log.info("FHE Execute-Modus Evaluierung übersprungen (n_samples_execute <= 0).")
                else:
                    log.info("FHE Evaluierung übersprungen (run_fhe_eval=false).")
        except Exception as e_fhe_hybrid:
            log.error(f"Fehler bei HybridFHEModel Kompilierung/Evaluierung: {e_fhe_hybrid}")
            log.error(traceback.format_exc())
            fhe_compilation_time_val = "Error"
    
    elif not best_model_state: # Diese Bedingung war vorher auch schon da
        log.warning("Kein bestes Modell vorhanden, FHE-Pipeline wird übersprungen.")
    else: 
        log.info("FHE Kompilierung und Evaluierung übersprungen (run_fhe_pipeline=false in config).")


    full_log_data = {
        "run_overview": {
            "run_timestamp": run_timestamp, "dataset_configured": dataset_name,
            "model_type_configured": model_cfg.get("type", "FHETabularTransformer"),
            "device_actually_used": str(device), "config_file_path_source": config_path
        },
        "input_configuration_from_yaml": config, 
        "data_summary_runtime": {"num_features": num_features, "num_classes": num_classes },
        "model_details_runtime": model_cfg, 
        "training_execution_summary": {
             "epochs_run_actual": val_metrics_at_best_f1w.get("epoch_of_best_f1w", actual_epochs_run_count),
             "best_val_f1_weighted_at_epoch": best_f1_w_val 
        },
        "best_model_metrics_achieved_val": val_metrics_at_best_f1w, 
        "evaluation_execution_details": {
            "pytorch_eval_summary": pytorch_eval_summary,
            "fhe_compilation_time_s": fhe_compilation_time_val, 
            "fhe_evaluations": { 
                "simulate": fhe_simulation_summary if fhe_simulation_summary else {"status": "skipped_or_error"},
                "execute": fhe_execute_summary if fhe_execute_summary else {"status": "skipped_or_error"}
            }
        },
        "output_artifact_paths": {
            "results_run_directory": results_base_dir_from_runner,
            "model_file_saved": model_save_path, 
            "detailed_text_log": detailed_log_path,
        }
    }
    json_log_filename = f"{run_timestamp}_{final_artifact_name}_full_run_log.json" 
    json_log_path = os.path.join(results_base_dir_from_runner, json_log_filename)
    try:
        with open(json_log_path, 'w') as f_json:
            def complex_handler(obj):
                if isinstance(obj, (np.ndarray, torch.Tensor)): return obj.tolist()
                if isinstance(obj, (np.integer, np.floating)): return obj.item()
                if isinstance(obj, Path): return str(obj)
                return f"Unserializable_{type(obj).__name__}"
            json.dump(full_log_data, f_json, indent=4, default=complex_handler) 
        log.info(f"JSON-Log gespeichert: {json_log_path}")
    except Exception as e_json:
        log.error(f"Fehler beim Speichern des JSON-Logs: {e_json}")
        log.error(traceback.format_exc())

    log_training_summary(log, model_save_path, best_f1_w_val, 
                           "N/A_loss_plot", "N/A_roc_plot", 
                           json_log_path)
    log.info(f"Transformer Pipeline für '{base_artifact_name}' abgeschlossen.") 
    
    if detailed_log_file_handler: 
        log.removeHandler(detailed_log_file_handler)
        detailed_log_file_handler.close()
        
    return results_base_dir_from_runner

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainiert ein FHE-Transformer Modell.")
    parser.add_argument(
        "--config",
        type=str,
        required=True, 
        help="Pfad zur YAML-Konfigurationsdatei für das Experiment."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"FEHLER: Konfigurationsdatei nicht gefunden: {args.config}")
        exit(1)
        
    try:
        run_transformer_training_pipeline(config_path=args.config)
    except Exception as e_main_run:
        logger.error(f"Ein unerwarteter Fehler ist in run_transformer_training_pipeline aufgetreten: {e_main_run}")
        logger.error(traceback.format_exc())