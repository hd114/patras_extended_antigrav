#!/usr/bin/env python
# coding: utf-8

"""
Analyse-Skript zur Visualisierung und Überprüfung der Pruning-Struktur.

Dieses Skript durchsucht Ergebnisordner von Experiment-Läufen, lädt die
gespeicherten Modelle (.pth) und die zugehörigen JSON-Logs. Es analysiert
die Gewichtsmatrizen der trainierten Modelle, um festzustellen, welches
strukturelle Pruning (z.B. Entfernung ganzer Neuronen/Zeilen) angewendet wurde.

Es generiert Heatmaps der binären Gewichtsmasken (aktiv vs. geprunt) und
einen Konnektivitäts-Report, der problematische Muster (z.B. "tote" Neuronen,
die Inputs empfangen, aber keine Outputs senden) identifiziert.

Ausführung (vom Projektstammverzeichnis FHE_athens):
    python -m src.utils.analyze_pruning_structure
"""

import os
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from typing import Dict, Any, Optional, List
import traceback
import logging
from datetime import datetime
import re
import sys

# --- Pfad-Konfiguration und Projekt-Root-Bestimmung ---
# Notwendig, damit das Skript sowohl direkt als auch als Modul funktioniert
# und die 'src'-Importe korrekt auflöst.
try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    UTILS_DIR = CURRENT_SCRIPT_DIR
    SRC_DIR = os.path.dirname(UTILS_DIR)
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
except NameError:
    # Fallback, falls __file__ nicht verfügbar ist (z.B. in interaktiven Umgebungen)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
    print(f"WARNUNG: __file__ nicht definiert. PROJECT_ROOT als '{PROJECT_ROOT}' angenommen.")

# Füge das Projektstammverzeichnis zum sys.path hinzu
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.utils.logger import setup_logger
    from src.models.qat_model import QATPrunedSimpleNet
    import brevitas.nn as qnn
except ImportError as e:
    print(f"FEHLER: Kritische Module konnten nicht importiert werden: {e}")
    print("Stellen Sie sicher, dass Sie das Skript vom Projektstammverzeichnis aus ausführen:")
    print("python -m src.utils.analyze_pruning_structure")
    sys.exit(1)


# --- Konfiguration für dieses Skript ---
DEFAULT_BASE_RESULTS_DIR_NAME = "qat_runs"
META_RUN_PATTERN: str = "meta_run_*"
REP_FOLDER_PATTERN: str = "rep_*"
JSON_LOG_SUFFIX: str = "_full_run_log.json"
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Schwellenwert, um ein Gewicht als "Null" zu betrachten (für strukturelle Analyse)
ZERO_THRESHOLD: float = 1e-9

# --- Hilfsfunktionen ---

def get_nested_val(data_dict: Dict, path: List[str], default: Any = None) -> Any:
    """
    Greift sicher auf einen verschachtelten Wert in einem Dictionary zu.
    
    Args:
        data_dict (Dict): Das Dictionary, das durchsucht wird.
        path (List[str]): Eine Liste von Schlüsseln, die den Pfad definieren.
        default (Any, optional): Der Standardwert, der zurückgegeben wird, 
                                 falls der Pfad nicht existiert.

    Returns:
        Any: Der gefundene Wert oder der Standardwert.
    """
    current = data_dict
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def load_model_for_pruning_analysis(
    model_class: Any,
    model_config_params: Dict,
    raw_data_summary: Dict,
    pth_file_path: str,
    logger_instance: logging.Logger
) -> Optional[torch.nn.Module]:
    """
    Lädt ein Modell (potenziell strukturell gepruned) zur Analyse.
    
    Diese Version ist an das neue strukturierte Pruning angepasst und ruft
    NICHT mehr die alte (fehlerhafte) .prune()-Methode auf.
    
    Args:
        model_class (Any): Die Modellklasse (z.B. QATPrunedSimpleNet).
        model_config_params (Dict): Das 'model_params' Dict aus dem JSON-Log.
        raw_data_summary (Dict): Das 'data_summary_runtime' Dict aus dem JSON-Log.
        pth_file_path (str): Der Pfad zur .pth-Datei.
        logger_instance (logging.Logger): Die Logger-Instanz.

    Returns:
        Optional[torch.nn.Module]: Das geladene Modell oder None bei einem Fehler.
    """
    try:
        # --- Parameter extrahieren (wie im Original) ---
        input_size = raw_data_summary.get("input_features_detected")
        num_classes = raw_data_summary.get("output_classes_detected")
        if input_size is None or num_classes is None:
            logger_instance.error("  Konnte input_size oder num_classes nicht aus raw_data_summary extrahieren.")
            return None
            
        cfg_h = model_config_params.get("n_hidden", 100)
        cfg_qb = model_config_params.get("quantization_bits", 3)
        dropout_cfg = model_config_params.get("dropout", {}); dr1 = dropout_cfg.get("rate1", 0.0); dr2 = dropout_cfg.get("rate2", 0.0)
        
        # Rekonstruiere die Brevitas-Argumente
        qlin_args = {"weight_bit_width": cfg_qb, "bias": model_config_params.get("qlinear_bias", True)}
        qident_args = {"bit_width": cfg_qb, "return_quant_tensor": model_config_params.get("qidentity_return_quant_tensor", True)}
        qrelu_args = {"bit_width": cfg_qb, "return_quant_tensor": model_config_params.get("qrelu_return_quant_tensor", True)}
        
        # --- Modell instanziieren ---
        model = model_class(
            input_size, num_classes, cfg_h,
            qlinear_args_config=qlin_args,
            qidentity_args_config=qident_args,
            qrelu_args_config=qrelu_args,
            dropout_rate1=dr1,
            dropout_rate2=dr2
        )

        # --- KORREKTUR: Die alte, FHE-ineffiziente Pruning-Logik wurde entfernt ---
        # Das Modell wird so instanziiert, wie es vor dem Pruning (Epoche 5) war.
        # Das State Dict enthält die permanent geprunten (entfernten) Gewichte.
        
        model.load_state_dict(torch.load(pth_file_path, map_location='cpu'))
        model.to(DEVICE)
        model.eval()
        
        logger_instance.info(f"  Modell erfolgreich von {pth_file_path} geladen.")
        return model
    
    except RuntimeError as e_load_state:
        logger_instance.error(f"  RuntimeError beim Laden des State Dict von {pth_file_path}: {e_load_state}")
        if "Missing key(s)" in str(e_load_state) or "Unexpected key(s)" in str(e_load_state):
            logger_instance.error("  Mismatch in Modell-Architektur oder State Dict (Pruning-Struktur?).")
        logger_instance.debug(traceback.format_exc())
        return None
    except Exception as e:
        logger_instance.error(f"  Allg. Fehler beim Inst/Laden des Modells {pth_file_path}: {e}")
        logger_instance.debug(traceback.format_exc())
        return None


def plot_weight_matrix_heatmap(weights_or_mask: np.ndarray, title: str, output_path: str, logger_instance: logging.Logger):
    """
    Erstellt eine Heatmap einer Gewichts- oder Maskenmatrix.
    
    Args:
        weights_or_mask (np.ndarray): Die 2D-Matrix, die geplottet werden soll.
        title (str): Der Titel für den Plot.
        output_path (str): Der Pfad zum Speichern des PNG-Bildes.
        logger_instance (logging.Logger): Die Logger-Instanz.
    """
    if weights_or_mask is None or weights_or_mask.size == 0:
        logger_instance.warning(f"  Keine Daten für Heatmap '{title}'.")
        return
        
    try:
        # Dynamische Anpassung der Figurengröße basierend auf der Matrixgröße
        plt.figure(figsize=(max(10, weights_or_mask.shape[1] * 0.15), max(8, weights_or_mask.shape[0] * 0.15)))
        
        # Ticks intelligent setzen, um Überlappung zu vermeiden
        max_ticks = 30
        y_ticks_pos = np.linspace(0, weights_or_mask.shape[0] - 1, min(weights_or_mask.shape[0], max_ticks), dtype=int) if weights_or_mask.shape[0] > 0 else []
        x_ticks_pos = np.linspace(0, weights_or_mask.shape[1] - 1, min(weights_or_mask.shape[1], max_ticks), dtype=int) if weights_or_mask.shape[1] > 0 else []
        y_tick_labels = [str(int(i)) for i in y_ticks_pos]
        x_tick_labels = [str(int(i)) for i in x_ticks_pos]
        
        sns.heatmap(
            weights_or_mask,
            cmap="viridis",  # 'viridis' (0=dunkel, 1=hell) eignet sich gut für binäre Masken
            cbar=True,
            xticklabels=x_tick_labels if weights_or_mask.shape[1] > 0 else False,
            yticklabels=y_tick_labels if weights_or_mask.shape[0] > 0 else False,
            square=False,    # Rechteckige Zellen
            linewidths=.1,
            linecolor='grey'
        )
        
        if weights_or_mask.shape[1] > max_ticks:
            plt.xticks(x_ticks_pos, x_tick_labels, rotation=45, ha="right", fontsize=8)
        else:
            plt.xticks(fontsize=8, rotation=45, ha="right")
            
        if weights_or_mask.shape[0] > max_ticks:
            plt.yticks(y_ticks_pos, y_tick_labels, fontsize=8)
        else:
            plt.yticks(fontsize=8)
            
        plt.title(title, fontsize=14)
        plt.xlabel("Input Neurons / Features Index", fontsize=12)
        plt.ylabel("Output Neurons Index", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger_instance.info(f"  Heatmap gespeichert: {output_path}")
    except Exception as e:
        logger_instance.error(f"  Fehler beim Erstellen der Heatmap '{title}': {e}")
        logger_instance.debug(traceback.format_exc())


def analyze_neuron_connectivity(
    model: QATPrunedSimpleNet,
    model_name_prefix: str,
    output_dir: str,
    logger_instance: logging.Logger
) -> Dict[str, List[int]]:
    """
    Analysiert die Konnektivität von strukturell geprunten Modellen.

    Sucht nach Zeilen (Neuronen) und Spalten (Inputs), die komplett Null sind,
    indem die L2-Norm (oder Absolutwert-Summe) der Gewichte analysiert wird.
    
    Args:
        model (QATPrunedSimpleNet): Das geladene PyTorch-Modell.
        model_name_prefix (str): Ein Präfix für die Dateinamen der Plots.
        output_dir (str): Der Ordner zum Speichern der Heatmaps/Reports.
        logger_instance (logging.Logger): Die Logger-Instanz.

    Returns:
        Dict[str, List[int]]: Ein Report über Konnektivitätsprobleme.
    """
    logger_instance.info(f"  Starte Konnektivitäts- und Strukturanalyse für {model_name_prefix}...")
    
    connectivity_issues_report: Dict[str, List[int]] = {
        "h1_active_in_all_out_pruned": [],
        "h1_all_in_pruned_active_out": [],
        "h2_active_in_all_out_pruned": [],
        "h2_all_in_pruned_active_out": []
    }

    layer_fc_names = []
    if not qnn:
        logger_instance.error("Brevitas (qnn) konnte nicht importiert werden. Analyse nicht möglich.")
        return connectivity_issues_report

    # Finde alle relevanten Layer im Modell
    if hasattr(model, 'fc1') and isinstance(model.fc1, (torch.nn.Linear, qnn.QuantLinear)): layer_fc_names.append('fc1')
    if hasattr(model, 'fc2') and isinstance(model.fc2, (torch.nn.Linear, qnn.QuantLinear)): layer_fc_names.append('fc2')
    if hasattr(model, 'fc3') and isinstance(model.fc3, (torch.nn.Linear, qnn.QuantLinear)): layer_fc_names.append('fc3')

    if not layer_fc_names:
        logger_instance.warning("  Keine erwarteten fc-Layer (fc1, fc2, fc3) im Modell gefunden.")
        return connectivity_issues_report

    # Speichert die binären (0/1) Gewichtsmatrizen für die Konnektivitätsanalyse
    weights_binary_masks: Dict[str, np.ndarray] = {}

    for name in layer_fc_names:
        layer = getattr(model, name)
        
        if not (hasattr(layer, 'weight') and layer.weight is not None):
            logger_instance.warning(f"  Layer {name} hat keine Gewichte. Überspringe Analyse für diesen Layer.")
            continue
            
        # Hole die Gewichte (diese enthalten die Nullen vom strukturierten Pruning)
        current_weights = layer.weight.cpu().detach().numpy()
        
        # Erstelle eine binäre Maske (1 = Aktiv, 0 = Inaktiv/Geprunt)
        current_mask_data = (np.abs(current_weights) > ZERO_THRESHOLD).astype(int)
        weights_binary_masks[name] = current_mask_data

        # --- Visualisierung der binären Maske ---
        plot_title = f"Binäre Gewichtsmaske (Strukturiert) - {name} ({model_name_prefix})"
        plot_path = os.path.join(output_dir, f"{model_name_prefix}_{name}_structured_mask_heatmap.png")
        plot_weight_matrix_heatmap(current_mask_data, plot_title, plot_path, logger_instance)

        # --- Analyse der Masken-Statistiken ---
        if current_mask_data is not None:
            num_output_neurons = current_mask_data.shape[0]
            num_input_features = current_mask_data.shape[1]
            
            # Zähle Neuronen (Ausgänge/Zeilen), die komplett Null sind
            output_neuron_activity = np.sum(current_mask_data, axis=1) # Summe der Inputs pro Neuron
            dead_output_neurons = np.sum(output_neuron_activity == 0)
            
            # Zähle Inputs (Spalten), die komplett Null sind
            input_feature_activity = np.sum(current_mask_data, axis=0) # Summe der Outputs pro Input
            dead_input_features = np.sum(input_feature_activity == 0)

            logger_instance.info(f"  Analyse für Layer '{name}':")
            logger_instance.info(f"    Form der Gewichte: {current_mask_data.shape}")
            logger_instance.info(f"    Output-Neuronen (Zeilen) in diesem Layer: {num_output_neurons}")
            logger_instance.info(f"    Davon komplett geprunt (alle Eingänge 0): {dead_output_neurons} ({dead_output_neurons/num_output_neurons:.1%})")
            logger_instance.info(f"    Input-Features (Spalten) in diesem Layer: {num_input_features}")
            logger_instance.info(f"    Davon komplett geprunt (von keinem Neuron genutzt): {dead_input_features} ({dead_input_features/num_input_features:.1%})")

    # --- Konnektivitätsanalyse (Prüft auf "tote Stränge") ---
    
    # Prüfe fc1 -> fc2
    if 'fc1' in weights_binary_masks and 'fc2' in weights_binary_masks:
        fc1_mask_data = weights_binary_masks['fc1']
        fc2_mask_data = weights_binary_masks['fc2']
        num_neurons_h1 = fc1_mask_data.shape[0] # Anzahl Neuronen in fc1
        
        # Iteriere über die Neuronen von H1 (Ausgänge von fc1 / Eingänge von fc2)
        for j in range(num_neurons_h1):
            # Prüfe, ob fc1-Neuron 'j' überhaupt aktiviert wird (hat es Eingänge?)
            # Wir nehmen an, wenn die Zeile > 0 ist, ist es aktiv.
            inputs_to_j_active = np.sum(fc1_mask_data[j, :]) > 0
            
            # Prüfe, ob fc1-Neuron 'j' irgendwelche Ausgänge zu fc2 hat (Spalte j in fc2)
            outputs_from_j_active = np.sum(fc2_mask_data[:, j]) > 0
            
            if inputs_to_j_active and not outputs_from_j_active:
                connectivity_issues_report["h1_active_in_all_out_pruned"].append(j)
            if not inputs_to_j_active and outputs_from_j_active:
                connectivity_issues_report["h1_all_in_pruned_active_out"].append(j)
        
        # Logge die Ergebnisse für H1
        if connectivity_issues_report["h1_active_in_all_out_pruned"]:
            logger_instance.warning(f"  WARNUNG H1 (fc1->fc2, Indizes 0-{num_neurons_h1-1}): {len(connectivity_issues_report['h1_active_in_all_out_pruned'])} Neuronen sind aktiv, aber ihre Ausgänge zu fc2 sind alle geprunt (Verschwendung).")
        if connectivity_issues_report["h1_all_in_pruned_active_out"]:
            logger_instance.warning(f"  WARNUNG H1 (fc1->fc2, Indizes 0-{num_neurons_h1-1}): {len(connectivity_issues_report['h1_all_in_pruned_active_out'])} Neuronen haben keine Inputs, aber Inputs zu fc2? (Sollte nicht passieren).")
        if not connectivity_issues_report["h1_active_in_all_out_pruned"] and not connectivity_issues_report["h1_all_in_pruned_active_out"]:
            logger_instance.info("  Keine problematischen Konnektivitätsmuster für Hidden Layer 1 (fc1 -> fc2) gefunden.")
    else:
        logger_instance.warning("  Masken für fc1 oder fc2 nicht verfügbar für detaillierte H1-Neuronenanalyse.")
    
    # Prüfe fc2 -> fc3
    if 'fc2' in weights_binary_masks and 'fc3' in weights_binary_masks:
        fc2_mask_data = weights_binary_masks['fc2']
        fc3_mask_data = weights_binary_masks['fc3']
        num_neurons_h2 = fc2_mask_data.shape[0] # Anzahl Neuronen in fc2
        
        for k in range(num_neurons_h2):
            inputs_to_k_active = np.sum(fc2_mask_data[k, :]) > 0
            outputs_from_k_active = np.sum(fc3_mask_data[:, k]) > 0
            
            if inputs_to_k_active and not outputs_from_k_active:
                connectivity_issues_report["h2_active_in_all_out_pruned"].append(k)
            if not inputs_to_k_active and outputs_from_k_active:
                connectivity_issues_report["h2_all_in_pruned_active_out"].append(k)
        
        # Logge die Ergebnisse für H2
        if connectivity_issues_report["h2_active_in_all_out_pruned"]:
            logger_instance.warning(f"  WARNUNG H2 (fc2->fc3, Indizes 0-{num_neurons_h2-1}): {len(connectivity_issues_report['h2_active_in_all_out_pruned'])} Neuronen sind aktiv, aber ihre Ausgänge zu fc3 sind alle geprunt (Verschwendung).")
        if connectivity_issues_report["h2_all_in_pruned_active_out"]:
            logger_instance.warning(f"  WARNUNG H2 (fc2->fc3, Indizes 0-{num_neurons_h2-1}): {len(connectivity_issues_report['h2_all_in_pruned_active_out'])} Neuronen haben keine Inputs, aber Inputs zu fc3? (Sollte nicht passieren).")
        if not connectivity_issues_report["h2_active_in_all_out_pruned"] and not connectivity_issues_report["h2_all_in_pruned_active_out"]:
            logger_instance.info("  Keine problematischen Konnektivitätsmuster für Hidden Layer 2 (fc2 -> fc3) gefunden.")
            
    return connectivity_issues_report

# --- Hauptausführungspunkt des Skripts ---
if __name__ == "__main__":
    log_main = setup_logger("Pruning_Structure_Analysis", level="INFO")
    log_main.info("--- Starte Analyse der Pruning-Struktur für gespeicherte Modelle ---")

    # Setze den Projekt-Root relativ zum Skript-Standort
    # (Bereits oben global definiert, hier zur Sicherheit)
    if 'PROJECT_ROOT' not in locals():
        try:
            CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            UTILS_DIR = CURRENT_SCRIPT_DIR
            SRC_DIR = os.path.dirname(UTILS_DIR)
            PROJECT_ROOT = os.path.dirname(SRC_DIR)
        except NameError:
            PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
            log_main.warning(f"__file__ nicht definiert, PROJECT_ROOT als '{PROJECT_ROOT}' angenommen.")
    
    log_main.info(f"Projekt-Stammverzeichnis: {PROJECT_ROOT}")

    default_base_dir = os.path.join(PROJECT_ROOT, "results", DEFAULT_BASE_RESULTS_DIR_NAME)

    # --- Eingabeaufforderung für den zu analysierenden Ordner ---
    user_input_base_dir = input(
        f"Pfad zum Ordner '{DEFAULT_BASE_RESULTS_DIR_NAME}', spezifischen 'meta_run_*' Ordner, oder Condition-Ordner (z.B. 'qb5_p32') angeben.\n"
        f"(Enter für Standard: '{default_base_dir}'): "
    ).strip()
    base_dir_to_search = user_input_base_dir if user_input_base_dir else default_base_dir

    if not os.path.isdir(base_dir_to_search):
        log_main.error(f"FEHLER: Pfad '{base_dir_to_search}' ist kein gültiges Verzeichnis.")
        sys.exit(1)
    
    log_main.info(f"Analysiere Ergebnisse unter Basispfad: {base_dir_to_search}")

    # --- Sammle alle zu analysierenden 'leaf'-Ordner (die Logs und Modelle enthalten) ---
    leaf_log_dirs_to_process = []
    
    # Szenario 1: Der angegebene Pfad ist bereits ein 'rep_X'-Ordner oder ein Zeitstempel-Ordner
    basename_of_search_dir = os.path.basename(base_dir_to_search)
    if basename_of_search_dir.startswith("rep_") or re.match(r"^\d{8}_\d{6}_", basename_of_search_dir):
        leaf_log_dirs_to_process.append(base_dir_to_search)
        log_main.info(f"Einzelner Lauf-Ordner wird analysiert: {base_dir_to_search}")
    else:
        # Szenario 2: Durchsuche 'meta_run_*' / 'condition_*' / 'rep_*'
        search_pattern = os.path.join(base_dir_to_search, META_RUN_PATTERN, "*", REP_FOLDER_PATTERN)
        rep_folders_found = glob.glob(search_pattern)
        
        if not rep_folders_found:
            # Fallback: Vielleicht ist es ein 'meta_run' Ordner und die Struktur ist flacher
            search_pattern_flat = os.path.join(base_dir_to_search, "*", REP_FOLDER_PATTERN)
            rep_folders_found = glob.glob(search_pattern_flat)

        log_main.info(f"{len(rep_folders_found)} Wiederholungs-Ordner ('rep_X') gefunden, die analysiert werden...")
        
        for rep_path in rep_folders_found:
            # Finde den Ordner mit Zeitstempel *innerhalb* des rep_X Ordners
            # (basierend auf der Korrektur für den experiment_runner)
            # Da wir jetzt direkt in rep_X speichern, ist rep_path der leaf_log_dir
            if os.path.isdir(rep_path):
                leaf_log_dirs_to_process.append(rep_path)
            
            # Alte Logik für den Fall, dass ein Zeitstempel-Ordner *in* rep_X ist
            # potential_ts_subdirs = [d for d in os.listdir(rep_path) if os.path.isdir(os.path.join(rep_path, d)) and re.match(r"^\d{8}_\d{6}_", d)]
            # if potential_ts_subdirs:
            #     leaf_log_dirs_to_process.append(os.path.join(rep_path, potential_ts_subdirs[0]))
            # else:
            #     leaf_log_dirs_to_process.append(rep_path)
                 
    if not leaf_log_dirs_to_process:
        log_main.warning(f"Keine gültigen Ergebnisordner (rep_X) unter {base_dir_to_search} gefunden. Prüfe, ob der Pfad korrekt ist.")
        sys.exit(0)

    # --- Verarbeitung der gefundenen Ordner ---
    for leaf_log_dir in leaf_log_dirs_to_process:
        try:
            # Kontext extrahieren
            rep_name = os.path.basename(leaf_log_dir)
            condition_name = os.path.basename(os.path.dirname(leaf_log_dir))
            meta_run_name = os.path.basename(os.path.dirname(os.path.dirname(leaf_log_dir)))
            
            log_main.info(f"\n--- Analysiere Lauf --- \n  Meta-Run: {meta_run_name}\n  Condition: {condition_name}\n  Repetition: {rep_name}")

            # Finde die Log- und Modelldateien
            json_log_file = None
            for fname in os.listdir(leaf_log_dir):
                if fname.endswith(JSON_LOG_SUFFIX):
                    json_log_file = os.path.join(leaf_log_dir, fname)
                    break
            if not json_log_file:
                log_main.warning(f"  Keine JSON-Logdatei in {leaf_log_dir} gefunden. Überspringe.")
                continue

            # Lade JSON-Inhalt
            with open(json_log_file, 'r', encoding='utf-8') as f:
                log_content = json.load(f)

            # Extrahiere Konfigurationen
            input_config = get_nested_val(log_content, ["input_configuration_from_yaml"], {})
            model_params_from_log = get_nested_val(input_config, ["model_params"], {})
            raw_data_summary_from_log = get_nested_val(log_content, ["data_summary_runtime"], {})
            
            if not model_params_from_log or not raw_data_summary_from_log:
                log_main.warning(f"  Unvollständige Daten in JSON: {json_log_file}. Überspringe.")
                continue

            # Finde den .pth-Pfad
            pth_file_path_from_log = get_nested_val(log_content, ["output_artifact_paths", "model_file_saved"])
            if not pth_file_path_from_log:
                log_main.warning(f"  Kein 'model_file_saved' Pfad im JSON-Log {json_log_file}. Überspringe.")
                continue
                
            # Stelle den Pfad relativ zum Projektstamm wieder her (falls nötig)
            if not os.path.isabs(pth_file_path_from_log):
                actual_pth_path = os.path.join(PROJECT_ROOT, pth_file_path_from_log)
            else:
                actual_pth_path = pth_file_path_from_log
                
            if not os.path.isfile(actual_pth_path):
                 # Fallback: Suche es im selben Ordner wie das JSON
                actual_pth_path = os.path.join(leaf_log_dir, os.path.basename(pth_file_path_from_log))
                if not os.path.isfile(actual_pth_path):
                    log_main.warning(f"  Modelldatei (.pth) nicht gefunden unter: {pth_file_path_from_log} ODER {actual_pth_path}. Überspringe.")
                    continue

            # Lade das Modell
            model_instance = load_model_for_pruning_analysis(
                QATPrunedSimpleNet,
                model_params_from_log,
                raw_data_summary_from_log,
                actual_pth_path,
                log_main
            )
            
            # Analysiere das Modell
            if model_instance:
                model_file_basename_prefix = os.path.splitext(os.path.basename(actual_pth_path))[0]
                
                # Führe die Analyse durch (erstellt Heatmaps)
                connectivity_report = analyze_neuron_connectivity(
                    model_instance, 
                    model_file_basename_prefix, 
                    leaf_log_dir, 
                    log_main
                )
                
                # Speichere den Konnektivitäts-Textreport
                report_filename = f"{model_file_basename_prefix}_connectivity_issues.txt"
                report_filepath = os.path.join(leaf_log_dir, report_filename)
                
                try:
                    with open(report_filepath, 'w', encoding='utf-8') as f_report:
                        f_report.write(f"Konnektivitätsanalyse für: {model_file_basename_prefix}\n")
                        f_report.write(f"Meta-Run: {meta_run_name}\nCondition: {condition_name}\nRepetition: {rep_name}\n")
                        f_report.write("="*50 + "\n\n")
                        has_issues = False
                        for key_issue, neuron_indices_list in connectivity_report.items():
                            if neuron_indices_list:
                                has_issues = True
                                f_report.write(f"{key_issue.replace('_', ' ').capitalize()}:\n  Indizes: {neuron_indices_list}\n  Anzahl: {len(neuron_indices_list)}\n\n")
                        if not has_issues:
                            f_report.write("Keine der geprüften problematischen Konnektivitätsmuster gefunden.\n")
                    log_main.info(f"  Konnektivitäts-Report gespeichert: {report_filepath}")
                except Exception as e_report_save:
                    log_main.error(f"  Fehler beim Speichern des Konnektivitäts-Reports: {e_report_save}")

        except Exception as e_leaf:
            log_main.error(f"  FEHLER bei der Verarbeitung des Ordners {leaf_log_dir}: {e_leaf}")
            log_main.debug(traceback.format_exc())

    log_main.info("--- Analyse der Pruning-Struktur beendet ---")