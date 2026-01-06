"""
concrete_evaluate_execute.py
Dediziertes Modul für die ausführliche FHE-Execution-Analyse.
Enthält System-Monitoring (CPU/RAM) und spezifische Formatierung.
"""

import os
import time
import logging
from datetime import datetime
from typing import Union, Optional, Dict, Any

import numpy as np
import psutil  # Für System-Metriken
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax as scipy_softmax

def format_confusion_matrix_brackets(cm):
    """
    Formatiert die CM exakt im [[ 1 0 ] [ 0 1 ]] Stil.
    """
    cm_str = "[\n"
    for row in cm:
        # Formatiert jede Zahl mit Breite 4 für Ausrichtung
        row_str = " [" + " ".join(f"{val:4d}" for val in row) + "]"
        cm_str += row_str + "\n"
    cm_str += "]"
    return cm_str

def _calculate_robust_roc_auc(y_true, y_prob):
    """Berechnet ROC-AUC, fängt Fehler bei fehlenden Klassen ab."""
    try:
        if len(np.unique(y_true)) < 2: 
            return "N/A (Only 1 Class in Subset)"
        return f"{roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'):.4f}"
    except Exception as e: 
        return f"N/A ({e})"

def evaluate_fhe_model(
    quantized_numpy_module: Any,
    X_test_data: np.ndarray,
    y_test_data: np.ndarray,
    label_encoder_global: LabelEncoder,
    model_name: str = "FHEModel",
    results_dir: str = "results/eval",
    fhe_mode: str = "execute",
    n_samples: int = 10,
    logger: Optional[logging.Logger] = None,
    footprint_metrics: Optional[Dict[str, Any]] = None 
) -> Dict[str, Any]:
    
    log_print = logger.info if logger else print
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Setup System Monitoring
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None) # Initial call (blockt nicht)
    initial_ram_mb = process.memory_info().rss / (1024 * 1024)
    peak_ram_mb = initial_ram_mb
    
    # 2. Daten Subset wählen
    if n_samples is None or n_samples > len(X_test_data):
        n_samples = len(X_test_data)
    
    X_subset = X_test_data[:n_samples]
    y_subset = y_test_data[:n_samples]

    log_print(f" -> Starte {fhe_mode.upper()} auf {n_samples} Samples...")

    start_time = time.time()
    y_pred_list = []
    y_prob_list = []
    cpu_measurements = []

    # 3. Inferenz Loop
    for i in tqdm(range(len(X_subset)), desc=f"FHE {fhe_mode}", leave=False):
        # CPU Snapshot vor der Berechnung
        cpu_measurements.append(process.cpu_percent(interval=None))
        
        sample = X_subset[i].reshape(1, -1)
        try:
            # Actual FHE Inference
            output = quantized_numpy_module.forward(sample, fhe=fhe_mode)
            
            # Ergebnisse sammeln
            logits = output[0]
            y_pred_list.append(np.argmax(logits))
            y_prob_list.append(logits)
            
            # RAM Check
            current_ram = process.memory_info().rss / (1024 * 1024)
            if current_ram > peak_ram_mb:
                peak_ram_mb = current_ram
                
        except Exception as e:
            log_print(f"Fehler bei Sample {i}: {e}")
            break

    total_duration = time.time() - start_time
    num_processed = len(y_pred_list)

    if num_processed == 0:
        log_print("ABBRUCH: Keine Samples erfolgreich verarbeitet.")
        return {}

    y_pred = np.array(y_pred_list)
    y_true = y_subset[:num_processed]

    # 4. Statistiken & Metriken
    avg_cpu = sum(cpu_measurements) / len(cpu_measurements) if cpu_measurements else 0.0
    time_per_sample = total_duration / num_processed
    time_per_1000 = time_per_sample * 1000

    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Softmax für ROC
    try:
        y_prob_arr = np.array(y_prob_list)
        y_prob_soft = scipy_softmax(y_prob_arr, axis=1)
        roc_str = _calculate_robust_roc_auc(y_true, y_prob_soft)
    except: roc_str = "N/A"

    # Confusion Matrix formatieren
    cm = confusion_matrix(y_true, y_pred)
    cm_str = format_confusion_matrix_brackets(cm)

    # Classification Report
    target_names = None
    if hasattr(label_encoder_global, 'classes_'):
        unique = sorted(list(set(y_true) | set(y_pred)))
        if max(unique) < len(label_encoder_global.classes_):
             target_names = [str(label_encoder_global.classes_[i]) for i in unique]

    report = classification_report(y_true, y_pred, digits=4, zero_division=0, target_names=target_names)

    # Footprint String bauen
    footprint_str = ""
    if footprint_metrics:
        footprint_str += "\n=== Model Footprint & Complexity ===\n"
        for k, v in footprint_metrics.items():
            footprint_str += f"{k:<30}: {v}\n"

    # System Stats String
    sys_stats = f"""
Avg CPU Usage:          {avg_cpu:.2f}%
Initial RAM:            {initial_ram_mb:.2f} MB
Peak RAM:               {peak_ram_mb:.2f} MB
Time per sample:        {time_per_sample:.4f}s
Time per 1000 samples:  {time_per_1000:.2f}s
"""

    # 5. Log schreiben
    log_content = f"""
=== FHE Model Evaluation ({fhe_mode}) ===
Timestamp:              {datetime.now()}
Model path:             {model_name}
Samples evaluated:      {num_processed}
Total execution time:   {total_duration:.2f}s
{sys_stats}
Accuracy:               {acc:.4f}
F1-Score (weighted):    {f1_w:.4f}
MCC:                    {mcc:.4f}
ROC-AUC:                {roc_str}
{footprint_str}
Confusion Matrix:
{cm_str}

Classification Report:
{report}
================================================================================
"""
    filename = f"eval_fhe_{model_name}_{fhe_mode}.txt"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(log_content)
    
    log_print(f" -> [LOG] Gespeichert: {filename}")
    
    return {"accuracy": acc, "f1_weighted": f1_w}