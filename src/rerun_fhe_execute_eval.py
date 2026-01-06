# rerun_fhe_execute_eval.py

import os
import glob
import json
import pandas as pd
import torch
import numpy as np
import yaml
from datetime import datetime
import traceback
import logging
from typing import Dict, Any, Optional, List
import re
import multiprocessing as mp # Ist hier nicht aktiv, aber der Import schadet nicht
import copy
import time


# --- Notwendige Importe aus deinem Projekt ---
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.data.edge_iiot_dataset import load_edgeiiot_data
from src.models.qat_model import QATPrunedSimpleNet
from src.evaluation.concrete_evaluate import evaluate_fhe_model # Stelle sicher, dass diese CPU/RAM misst
from concrete.ml.torch.compile import compile_brevitas_qat_model

# --- Konfiguration für dieses Skript ---
DEFAULT_N_SAMPLES_EXECUTE = 1000 
OUTPUT_CSV_FILENAME_PREFIX = "fhe_execute_reevaluation_summary"
META_RUN_PATTERN: str = "meta_run_*" 
JSON_LOG_SUFFIX: str = "_full_run_log.json"
REP_FOLDER_PATTERN: str = "rep_*"

# --- Hilfsfunktionen ---
def get_nested_val(data_dict: Dict, path: List[str], default: Any = None) -> Any:
    current = data_dict
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def load_model_from_pth(model_architecture: torch.nn.Module, 
                        pth_file_path: str, 
                        device: torch.device, 
                        logger_instance: logging.Logger) -> Optional[torch.nn.Module]:
    try:
        # Diese Funktion wird hier nicht mehr direkt für das FHE-Modell verwendet,
        # da wir die Instanziierung und das Laden des State Dicts direkt handhaben.
        # Sie könnte aber nützlich sein, wenn man das PyTorch-Modell separat laden wollte.
        model_architecture.load_state_dict(torch.load(pth_file_path, map_location=device))
        model_architecture.to(device)
        model_architecture.eval()
        return model_architecture
    except Exception as e:
        logger_instance.error(f"Fehler beim Laden des Modells von {pth_file_path}: {e}")
        logger_instance.debug(traceback.format_exc())
        return None

# --- Hauptlogik des Skripts ---
if __name__ == "__main__":
    log = setup_logger("FHE_Execute_Reevaluation", level="INFO")
    log.info("--- Starte FHE 'execute' Re-Evaluierung für gespeicherte Modelle ---")

    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_main = current_script_dir 
    except NameError: 
        project_root_main = os.getcwd()
        log.warning(f"__file__ nicht definiert, verwende aktuelles Arbeitsverzeichnis als Projekt-Root: {project_root_main}")

    default_qat_runs_dir = os.path.join(project_root_main, "results", "qat_runs")

    user_input_base_dir = input(
        f"Bitte den Pfad zum 'qat_runs'-Ordner eingeben, der die 'meta_run_*' Ordner enthält.\n"
        f"(Enter für Standard: '{default_qat_runs_dir}'): "
    ).strip()
    qat_runs_base_dir = user_input_base_dir if user_input_base_dir else default_qat_runs_dir

    if not os.path.isdir(qat_runs_base_dir):
        log.error(f"FEHLER: Der angegebene Pfad '{qat_runs_base_dir}' ist kein gültiges Verzeichnis.")
        exit(1) 
    log.info(f"Durchsuche '{qat_runs_base_dir}' nach '{META_RUN_PATTERN}' Ordnern.")

    user_input_samples = input(
        f"Anzahl der Samples für FHE 'execute' Evaluierung (Enter für {DEFAULT_N_SAMPLES_EXECUTE}): "
    ).strip()
    n_samples_for_execute = int(user_input_samples) if user_input_samples.isdigit() and int(user_input_samples) > 0 else DEFAULT_N_SAMPLES_EXECUTE
    log.info(f"FHE 'execute' Evaluierung wird mit {n_samples_for_execute} Samples durchgeführt.")

    all_reevaluation_results = []
    processed_run_dirs_for_json_search = set()

    meta_run_folders = glob.glob(os.path.join(qat_runs_base_dir, META_RUN_PATTERN))

    if not meta_run_folders:
        log.warning(f"Keine Ordner gefunden, die dem Muster '{META_RUN_PATTERN}' in '{qat_runs_base_dir}' entsprechen.")
        exit() 
    
    log.info(f"{len(meta_run_folders)} Ordner gefunden, die mit '{META_RUN_PATTERN}' beginnen und verarbeitet werden.")

    for meta_run_path in meta_run_folders:
        if not os.path.isdir(meta_run_path):
            continue
        meta_run_name = os.path.basename(meta_run_path)
        log.info(f"\nVerarbeite Meta Run: {meta_run_name}")

        for condition_folder_name in os.listdir(meta_run_path):
            condition_folder_path = os.path.join(meta_run_path, condition_folder_name)
            if not os.path.isdir(condition_folder_path) or condition_folder_name == "run_configs":
                continue
            log.info(f"  Prüfe Condition-Ordner: {condition_folder_name}")

            rep_folders_paths = glob.glob(os.path.join(condition_folder_path, REP_FOLDER_PATTERN))
            if not rep_folders_paths:
                log.info(f"    Keine '{REP_FOLDER_PATTERN}' in {condition_folder_path}. Prüfe Ordner direkt als einzelnen Lauf.")
                rep_folders_paths = [condition_folder_path] 
            
            for rep_folder_path in rep_folders_paths:
                repetition_name = os.path.basename(rep_folder_path) if rep_folder_path != condition_folder_path else "direct_in_condition"
                
                actual_log_containing_dir = rep_folder_path
                potential_run_subdirs = [
                    d for d in os.listdir(rep_folder_path)
                    if os.path.isdir(os.path.join(rep_folder_path, d)) and re.match(r"^\d{8}_\d{6}_", d)
                ]
                if potential_run_subdirs:
                    actual_log_containing_dir = os.path.join(rep_folder_path, potential_run_subdirs[0])
                    log.info(f"    Verarbeite Wiederholung: {repetition_name} (Lauf-Ordner: {os.path.basename(actual_log_containing_dir)})")
                else:
                    log.info(f"    Verarbeite Wiederholung/Lauf-Ordner direkt: {repetition_name} (Pfad: {actual_log_containing_dir})")

                if actual_log_containing_dir in processed_run_dirs_for_json_search:
                    log.debug(f"      Ordner {actual_log_containing_dir} bereits verarbeitet, überspringe.")
                    continue

                json_log_file = None
                for fname in os.listdir(actual_log_containing_dir):
                    if fname.endswith(JSON_LOG_SUFFIX):
                        json_log_file = os.path.join(actual_log_containing_dir, fname)
                        break
                
                if not json_log_file:
                    log.warning(f"      Keine JSON-Logdatei in {actual_log_containing_dir} gefunden.")
                    processed_run_dirs_for_json_search.add(actual_log_containing_dir)
                    continue
                
                log.debug(f"      Verarbeite JSON-Log: {json_log_file}")

                try:
                    with open(json_log_file, 'r', encoding='utf-8') as f:
                        log_content = json.load(f)
                except Exception as e:
                    log.error(f"      Konnte JSON nicht laden {json_log_file}: {e}")
                    continue

                input_config_from_json = get_nested_val(log_content, ["input_configuration_from_yaml"], {})
                if not input_config_from_json:
                    log.warning(f"      Keine 'input_configuration_from_yaml' im JSON: {json_log_file}")
                    continue
                
                model_params_orig = get_nested_val(input_config_from_json, ["model_params"], {})
                data_params_orig = get_nested_val(input_config_from_json, ["data_params"], {})

                pth_file_path_from_log = get_nested_val(log_content, ["output_artifact_paths", "model_file_saved"])
                if not pth_file_path_from_log: 
                    pth_file_path_from_log = get_nested_val(log_content, ["output_artifact_paths", "model_file"])

                pth_file_path = None
                if pth_file_path_from_log:
                    if os.path.isabs(pth_file_path_from_log) and os.path.isfile(pth_file_path_from_log):
                        pth_file_path = pth_file_path_from_log
                    else: 
                        pth_file_path_sibling = os.path.join(actual_log_containing_dir, os.path.basename(pth_file_path_from_log))
                        if os.path.isfile(pth_file_path_sibling):
                            pth_file_path = pth_file_path_sibling
                        else:
                            pth_file_path_rel_project = os.path.join(project_root_main, pth_file_path_from_log)
                            if os.path.isfile(pth_file_path_rel_project):
                                pth_file_path = pth_file_path_rel_project
                
                if not pth_file_path:
                    log.warning(f"      Modelldatei nicht gefunden für JSON {json_log_file} (gesuchter Pfad: {pth_file_path_from_log}). Überspringe.")
                    continue
                
                log.info(f"      Verwende Modelldatei: {pth_file_path}")

                npz_path_orig = data_params_orig.get("npz_file_path")
                if not npz_path_orig: log.warning(f"      Kein NPZ-Pfad in Config für {json_log_file}. Überspringe."); continue
                if not os.path.isabs(npz_path_orig): npz_path_orig = os.path.join(project_root_main, npz_path_orig)
                if not os.path.isfile(npz_path_orig): log.warning(f"      NPZ-Datei '{npz_path_orig}' nicht gefunden. Überspringe."); continue
                
                log.info(f"      Lade Daten von: {npz_path_orig}")
                loaded_data = load_edgeiiot_data(npz_path=npz_path_orig, batch_size=32, return_raw_data=True, logger=log)
                if loaded_data is None or loaded_data[3] is None: log.error(f"      Fehler beim Laden der Daten von {npz_path_orig}."); continue
                _, _, _, raw_data = loaded_data
                X_train_np_for_compile = raw_data["X_train"]
                X_test_np_for_eval, y_test_np_for_eval = raw_data["X_test"], raw_data["y_test"]
                label_encoder_loaded = raw_data["label_encoder"]
                input_size_loaded = raw_data["num_features"]
                num_classes_loaded = raw_data["num_classes"]

                device = torch.device("cpu") # FHE Kompilierung und Inferenz laufen auf CPU
                
                cfg_h = model_params_orig.get("n_hidden",100)
                cfg_qb = model_params_orig.get("quantization_bits",3)
                cfg_up_orig = model_params_orig.get("unpruned_neurons",0) 
                dr1_cfg = get_nested_val(model_params_orig,["dropout","rate1"],0.0)
                dr2_cfg = get_nested_val(model_params_orig,["dropout","rate2"],0.0)
                qlin_args = {"weight_bit_width":cfg_qb, "bias":model_params_orig.get("qlinear_bias",True)}
                qident_args = {"bit_width":cfg_qb, "return_quant_tensor":model_params_orig.get("qidentity_return_quant_tensor",True)}
                qrelu_args = {"bit_width":cfg_qb, "return_quant_tensor":model_params_orig.get("qrelu_return_quant_tensor",True)}

                # 1. Erstelle die Modellinstanz für die FHE-Kompilierung
                fhe_model_to_compile = QATPrunedSimpleNet(
                    input_size_loaded, num_classes_loaded, cfg_h, 
                    qlinear_args_config=qlin_args, qidentity_args_config=qident_args,
                    qrelu_args_config=qrelu_args, dropout_rate1=dr1_cfg, dropout_rate2=dr2_cfg
                ) # Wird auf CPU instanziiert

                # 2. Wende Pruning auf diese neue Instanz an, *bevor* das state_dict geladen wird,
                #    FALLS das ursprüngliche Modell (dessen state_dict wir laden) gepruned war.
                if cfg_up_orig > 0 and input_size_loaded > cfg_up_orig:
                    log.info(f"        Wende Pruning (max_non_zero_per_neuron={cfg_up_orig}) auf neue Modellinstanz an.")
                    fhe_model_to_compile.prune(max_non_zero_per_neuron=cfg_up_orig)
                
                # 3. Lade das State Dictionary
                try:
                    fhe_model_to_compile.load_state_dict(torch.load(pth_file_path, map_location='cpu')) 
                    fhe_model_to_compile.eval()
                    log.info(f"        State Dict erfolgreich in fhe_model_to_compile geladen von: {pth_file_path}")
                except RuntimeError as e_load_fhe: 
                    log.error(f"      Fehler beim Laden des State Dict in das FHE-Compile-Modell von {pth_file_path}: {e_load_fhe}")
                    log.debug(traceback.format_exc())
                    if "Missing key(s)" in str(e_load_fhe) or "Unexpected key(s)" in str(e_load_fhe):
                        log.error("        Dies deutet auf ein Mismatch in der Pruning-Struktur hin. "
                                  "Überprüfe, ob das Modell vor dem Laden des State Dicts korrekt gepruned wurde, "
                                  "falls das gespeicherte Modell gepruned war (und die cfg_up_orig aus der JSON korrekt ist).")
                    continue 
                except Exception as e_load_fhe_other: 
                    log.error(f"      Allgemeiner Fehler beim Laden des State Dict: {e_load_fhe_other}")
                    log.debug(traceback.format_exc())
                    continue

                # 4. Für ConcreteML unprunen, falls es gepruned war (nachdem das state_dict geladen wurde)
                if cfg_up_orig > 0 and input_size_loaded > cfg_up_orig:
                    # Stelle sicher, dass das Modell auf CPU ist, bevor unprune gerufen wird, wenn es state manipuliert
                    fhe_model_to_compile.cpu() 
                    log.info("        Unpruning FHE model instance before ConcreteML compilation.")
                    fhe_model_to_compile.unprune() 
                
                fhe_model_to_compile.cpu() # Sicherstellen, dass es auf CPU ist
                
                fhe_eval_cfg_orig = get_nested_val(input_config_from_json, ["evaluation_params", "fhe_model_eval"], {})
                comp_sample_size_cfg = fhe_eval_cfg_orig.get("compilation_sample_size", 128)
                compile_sample_size_actual = min(comp_sample_size_cfg, len(X_train_np_for_compile))
                if compile_sample_size_actual == 0 and len(X_train_np_for_compile) > 0: compile_sample_size_actual = 1
                
                quantized_fhe_module = None; compilation_time_val = "N/A (Skipped)"
                if compile_sample_size_actual > 0:
                    compile_input = torch.tensor(X_train_np_for_compile[:compile_sample_size_actual], dtype=torch.float32).cpu()
                    log.info(f"        Starte FHE Kompilierung mit {compile_sample_size_actual} Samples...")
                    compile_start_time = time.time()
                    try:
                        quantized_fhe_module = compile_brevitas_qat_model(fhe_model_to_compile, compile_input)
                        compilation_time_val = round(time.time() - compile_start_time, 2)
                        log.info(f"        FHE Kompilierung erfolgreich. Dauer: {compilation_time_val}s")
                    except Exception as e_compile:
                        log.error(f"        Fehler bei FHE Kompilierung: {e_compile}")
                        log.debug(traceback.format_exc())
                        compilation_time_val = "N/A (Failed)"
                else: log.warning("        Keine Samples für FHE Kompilierung. Überspringe.")

                fhe_execute_eval_results = {}
                if quantized_fhe_module:
                    eval_model_name_prefix = os.path.splitext(os.path.basename(pth_file_path))[0]
                    eval_output_dir = actual_log_containing_dir 

                    log.info(f"        Starte FHE 'execute' Evaluierung für {eval_model_name_prefix} mit {n_samples_for_execute} Samples...")
                    fhe_execute_eval_results = evaluate_fhe_model(
                        quantized_numpy_module=quantized_fhe_module, X_test_data=X_test_np_for_eval,
                        y_test_data=y_test_np_for_eval, label_encoder_global=label_encoder_loaded,
                        model_name=f"{eval_model_name_prefix}_EXECUTE_REEVAL",
                        results_dir=eval_output_dir, fhe_mode="execute",
                        n_samples=n_samples_for_execute, logger=log,
                        fhe_compilation_time_s=compilation_time_val
                    )
                else:
                    log.warning("        FHE Kompilierung nicht erfolgreich. Keine 'execute' Evaluierung.")
                
                run_data_summary = {
                    "meta_run": meta_run_name, "condition": condition_folder_name, "repetition": repetition_name,
                    "original_json_log": json_log_file, "pth_model_path": pth_file_path,
                    "fhe_compilation_time_s": compilation_time_val,
                    "fhe_execute_n_samples": n_samples_for_execute if quantized_fhe_module else "N/A",
                }
                for key, value in fhe_execute_eval_results.items():
                    run_data_summary[f"fhe_execute_{key}"] = value 

                all_reevaluation_results.append(run_data_summary)
                processed_run_dirs_for_json_search.add(actual_log_containing_dir)

    if all_reevaluation_results:
        summary_df = pd.DataFrame(all_reevaluation_results)
        timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
        summary_csv_filename = f"{OUTPUT_CSV_FILENAME_PREFIX}_{timestamp_str}.csv"
        summary_csv_path = os.path.join(qat_runs_base_dir, summary_csv_filename)
        try:
            summary_df.to_csv(summary_csv_path, index=False, sep=';')
            log.info(f"\nZusammenfassung der FHE 'execute' Re-Evaluierung gespeichert in: {summary_csv_path}")
            print(f"\nZusammenfassung der FHE 'execute' Re-Evaluierung gespeichert in: {summary_csv_path}")
            pd.set_option('display.max_columns', None); pd.set_option('display.width', 200)
            print(summary_df.head().to_string())
        except Exception as e_csv: log.error(f"Fehler beim Speichern der Zusammenfassung: {e_csv}")
    else:
        log.info("Keine Modelle für FHE 'execute' Re-Evaluierung verarbeitet oder Ergebnisse gesammelt.")

    log.info("--- FHE 'execute' Re-Evaluierungsskript beendet ---")