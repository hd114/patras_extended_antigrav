# main_analyzer.py
import os
import glob
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import traceback # Für detailliertere Fehlermeldungen
import re
from datetime import datetime

# --- Konfiguration (Globale Konstanten) ---
META_RUN_PATTERN: str = "meta_run_*"
CONFIG_FOLDER_NAME_DEFAULT: str = "default_config" 
REP_FOLDER_PATTERN: str = "rep_*"
JSON_LOG_SUFFIX: str = "_full_run_log.json"

# --- HILFSFUNKTIONEN (wie in deiner letzten, funktionierenden Version) ---
def get_nested_val(data_dict: Dict, path: List[str], default: Any = None) -> Any:
    current = data_dict
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def parse_json_log(file_path: str,
                   meta_run_name: str,
                   condition_name: str,
                   repetition_name: str) -> Optional[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
    except Exception as e:
        print(f"      Fehler: Konnte JSON-Datei nicht laden oder parsen: {file_path} - {e}")
        return None
    data = {
        "meta_run": meta_run_name, "condition": condition_name, "repetition": repetition_name,
        "run_folder_leaf": os.path.basename(os.path.dirname(file_path)),
        "json_file_name": os.path.basename(file_path),
        "dataset_name": get_nested_val(content, ["run_overview", "dataset_configured"]),
        "config_source_path": get_nested_val(content, ["run_overview", "config_file_path_source"]),
    }
    config_yaml = content.get("input_configuration_from_yaml", {})
    model_cfg = get_nested_val(config_yaml, ["model_params"], {})
    train_cfg = get_nested_val(config_yaml, ["training_params"], {})
    data_params_cfg = get_nested_val(config_yaml, ["data_params"], {})
    data["model_type_cfg"] = model_cfg.get("type")
    data["cfg_n_hidden"] = model_cfg.get("n_hidden")
    data["cfg_quant_bits"] = model_cfg.get("quantization_bits")
    data["cfg_unpruned_neurons"] = model_cfg.get("unpruned_neurons")
    data["cfg_dropout_rate1"] = get_nested_val(model_cfg, ["dropout", "rate1"])
    data["cfg_dropout_rate2"] = get_nested_val(model_cfg, ["dropout", "rate2"])
    data["cfg_learning_rate"] = train_cfg.get("learning_rate")
    data["cfg_dataloader_batch_size"] = data_params_cfg.get("dataloader_batch_size")
    data["cfg_use_train_loader"] = train_cfg.get("use_train_loader_for_batches")
    data["cfg_manual_batch_size"] = train_cfg.get("manual_batch_size")
    data["cfg_use_subset_training"] = data_params_cfg.get("use_subset_training")
    data["cfg_subset_fraction"] = data_params_cfg.get("subset_fraction")
    criterion_cfg = get_nested_val(train_cfg, ["criterion"], {})
    data["cfg_criterion_name"] = criterion_cfg.get("name")
    data["cfg_focal_alpha"] = criterion_cfg.get("focal_loss_alpha")
    data["cfg_focal_gamma"] = criterion_cfg.get("focal_loss_gamma")
    data["cfg_calc_class_weights"] = criterion_cfg.get("calculate_class_weights")
    data["cfg_class_weight_method"] = criterion_cfg.get("class_weight_calculation_method")
    data["cfg_class_weight_beta"] = criterion_cfg.get("class_weight_beta")
    optimizer_cfg = get_nested_val(train_cfg, ["optimizer"], {})
    data["cfg_optimizer_name"] = optimizer_cfg.get("name")
    data["cfg_weight_decay"] = optimizer_cfg.get("weight_decay")
    scheduler_cfg = get_nested_val(train_cfg, ["scheduler"], {})
    data["cfg_scheduler_name"] = scheduler_cfg.get("name")
    if data["cfg_scheduler_name"] == "ReduceLROnPlateau":
        data["cfg_scheduler_factor"] = scheduler_cfg.get("reduce_lr_factor")
        data["cfg_scheduler_patience"] = scheduler_cfg.get("reduce_lr_patience")
        data["cfg_scheduler_min_lr"] = scheduler_cfg.get("reduce_lr_min_lr")
    elif data["cfg_scheduler_name"] == "StepLR":
        data["cfg_scheduler_step_size"] = scheduler_cfg.get("step_lr_step_size")
        data["cfg_scheduler_gamma"] = scheduler_cfg.get("step_lr_gamma")
    early_stopping_cfg = get_nested_val(train_cfg, ["early_stopping"], {})
    data["cfg_early_stop_patience"] = early_stopping_cfg.get("patience")
    data["cfg_num_epochs"] = train_cfg.get("num_epochs")
    train_exec = content.get("training_execution_summary", {})
    data["epochs_run_actual"] = train_exec.get("epochs_run_actual")
    early_stop_details = train_exec.get("early_stopping_details_runtime", {})
    data["early_stop_triggered"] = early_stop_details.get("triggered")
    best_val_metrics = content.get("best_model_metrics_achieved_val", {})
    data["best_val_f1_weighted"] = best_val_metrics.get("best_f1_weighted_val")
    data["f1_macro_at_best_f1w_val"] = best_val_metrics.get("f1_macro_at_best_f1w_val")
    data["roc_auc_at_best_f1w_val"] = best_val_metrics.get("roc_auc_at_best_f1w_val")
    data["epoch_of_best_f1w"] = best_val_metrics.get("epoch_of_best_f1w")
    eval_details = content.get("evaluation_execution_details", {})
    pytorch_summary = get_nested_val(eval_details, ["pytorch_eval_summary"], {})
    data["pytorch_json_accuracy"] = pytorch_summary.get("accuracy")
    data["pytorch_json_f1_macro"] = pytorch_summary.get("f1_macro")
    data["pytorch_json_f1_weighted"] = pytorch_summary.get("f1_weighted")
    data["pytorch_json_roc_auc_macro"] = pytorch_summary.get("roc_auc_macro_ovr")
    data["pytorch_json_mcc"] = pytorch_summary.get("mcc")
    data["pytorch_json_infer_time_s"] = pytorch_summary.get("total_inference_time_s")
    data["pytorch_json_time_per_1000"] = pytorch_summary.get("time_per_1000_samples_s")
    data["pytorch_json_avg_cpu"] = pytorch_summary.get("avg_cpu_percent_process")
    data["pytorch_json_peak_ram_mb"] = pytorch_summary.get("peak_ram_rss_mb_process")
    data["fhe_compilation_time_s"] = eval_details.get("fhe_compilation_time_s")
    fhe_evaluations = get_nested_val(eval_details, ["fhe_evaluations"], {})
    for mode in ["simulate", "execute"]:
        fhe_mode_summary = fhe_evaluations.get(mode, {})
        if fhe_mode_summary and fhe_mode_summary.get("status") != "skipped":
            data[f"fhe_{mode}_json_accuracy"] = fhe_mode_summary.get("accuracy")
            data[f"fhe_{mode}_json_f1_macro"] = fhe_mode_summary.get("f1_macro")
            data[f"fhe_{mode}_json_f1_weighted"] = fhe_mode_summary.get("f1_weighted")
            data[f"fhe_{mode}_json_roc_auc_macro"] = fhe_mode_summary.get("roc_auc_macro_ovr")
            data[f"fhe_{mode}_json_mcc"] = fhe_mode_summary.get("mcc")
            data[f"fhe_{mode}_json_infer_time_s"] = fhe_mode_summary.get("total_inference_time_s")
            data[f"fhe_{mode}_json_time_per_1000"] = fhe_mode_summary.get("time_per_1000_samples_s")
            data[f"fhe_{mode}_json_avg_cpu"] = fhe_mode_summary.get("avg_cpu_percent_process")
            data[f"fhe_{mode}_json_peak_ram_mb"] = fhe_mode_summary.get("peak_ram_rss_mb_process")
    final_sparsity_details = get_nested_val(content, ["pruning_log_details", "final_sparsity"], {})
    if isinstance(final_sparsity_details, dict):
        sparsities = [layer.get("sparsity", 0) for layer_name, layer in final_sparsity_details.items() if isinstance(layer, dict) and "sparsity" in layer]
        data["avg_final_sparsity"] = sum(sparsities) / len(sparsities) if sparsities else None
    else: data["avg_final_sparsity"] = None
    return data

def parse_eval_text_file(file_path: str, prefix: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        patterns = {
            "Acc": r"Acc:\s*([0-9\.]+)", "Precm": r"Prec\(m\):\s*([0-9\.]+)",
            "Recm": r"Rec\(m\):\s*([0-9\.]+)", "F1m": r"F1\(m\):\s*([0-9\.]+)",
            "F1w": r"F1\(w\):\s*([0-9\.]+)", "MCC": r"MCC:\s*([0-9\.]+)",
            "ROC-AUCm": r"ROC-AUC\(m\):\s*([0-9\.]+|n/a\s*\(.*?\))"
        }
        for key_metric, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value_str = match.group(1)
                try:
                    if "n/a" in value_str.lower(): metrics[f'{prefix}_{key_metric}'] = value_str
                    else: metrics[f'{prefix}_{key_metric}'] = float(value_str)
                except ValueError: metrics[f'{prefix}_{key_metric}'] = value_str
        time_match = re.search(r"Total (?:PyTorch|FHE) Infer Time:\s*([0-9\.]+)s", content)
        if time_match: metrics[f'{prefix}_total_infer_time_s_txt'] = float(time_match.group(1))
        time_1k_match = re.search(r"(?:PyTorch|FHE) Time/1000 samples:\s*([0-9\.]+)s", content)
        if time_1k_match: metrics[f'{prefix}_time_per_1000_samples_s_txt'] = float(time_1k_match.group(1))
        if "FHE Model Eval" in content:
            comp_time_match = re.search(r"FHE Compilation Time:\s*([0-9\.]+|N/A[^s]*)s?", content)
            if comp_time_match:
                comp_time_val = comp_time_match.group(1)
                try: metrics[f'{prefix}_fhe_compilation_time_s_txt'] = float(comp_time_val)
                except ValueError: metrics[f'{prefix}_fhe_compilation_time_s_txt'] = comp_time_val
    except Exception as e: print(f"      Fehler beim Parsen der Textdatei {file_path}: {e}")
    return metrics

# --- FUNKTIONEN, DIE DIE OBIGEN AUFRUFEN ---
def process_repetition_folder(rep_folder_path: str,
                              meta_run_name: str,
                              condition_name: str,
                              repetition_name: str) -> List[Dict[str, Any]]:
    aggregated_data: List[Dict[str, Any]] = []
    potential_run_subdirs = [
        d for d in os.listdir(rep_folder_path)
        if os.path.isdir(os.path.join(rep_folder_path, d)) and
           re.match(r"^\d{8}_\d{6}_", d) # Sucht nach Zeitstempel-Format YYYYMMDD_HHMMSS_
    ]
    paths_to_process_logs_in = []
    if potential_run_subdirs:
        for subdir_name in potential_run_subdirs:
            paths_to_process_logs_in.append(os.path.join(rep_folder_path, subdir_name))
    else:
        paths_to_process_logs_in.append(rep_folder_path)

    for actual_log_containing_path in paths_to_process_logs_in:
        json_log_files = glob.glob(os.path.join(actual_log_containing_path, f"*{JSON_LOG_SUFFIX}"))
        if not json_log_files:
            # print(f"        WARNUNG: Keine JSON-Log-Datei ('*{JSON_LOG_SUFFIX}') in {actual_log_containing_path} gefunden.")
            continue
        for json_file_path in json_log_files:
            if os.path.isfile(json_file_path):
                data_item = parse_json_log(json_file_path, meta_run_name, condition_name, repetition_name)
                if data_item:
                    torch_eval_txt_path = None
                    for file_name in os.listdir(actual_log_containing_path):
                        if file_name.startswith("eval_torch_") and file_name.endswith(".txt"):
                            torch_eval_txt_path = os.path.join(actual_log_containing_path, file_name)
                            torch_txt_data = parse_eval_text_file(torch_eval_txt_path, "pytorch_txt")
                            data_item.update(torch_txt_data)
                            break
                    fhe_eval_files = [f for f in os.listdir(actual_log_containing_path) if f.startswith("eval_fhe_") and f.endswith(".txt")]
                    for fhe_file in fhe_eval_files:
                        fhe_eval_txt_path_nested = os.path.join(actual_log_containing_path, fhe_file)
                        mode_match = re.search(r"eval_fhe_.*_(simulate|execute)(?:_.*)?\.txt", fhe_file)
                        fhe_txt_prefix = f"fhe_{mode_match.group(1)}_txt" if mode_match else "fhe_unknown_txt"
                        fhe_txt_data = parse_eval_text_file(fhe_eval_txt_path_nested, fhe_txt_prefix)
                        data_item.update(fhe_txt_data)
                    aggregated_data.append(data_item)
    return aggregated_data

def find_and_process_repetitions(base_path_for_condition: str,
                                 meta_run_name: str,
                                 condition_name: str) -> List[Dict[str, Any]]:
    aggregated_data: List[Dict[str, Any]] = []
    rep_folders = glob.glob(os.path.join(base_path_for_condition, REP_FOLDER_PATTERN))
    if not rep_folders:
        # print(f"    INFO: Keine '{REP_FOLDER_PATTERN}' Ordner in '{base_path_for_condition}'. Behandle als einzelnen Lauf.")
        data_from_run = process_repetition_folder(base_path_for_condition, meta_run_name, condition_name, "rep_direct")
        aggregated_data.extend(data_from_run)
    else:
        for rep_path in rep_folders:
            if os.path.isdir(rep_path):
                repetition_name = os.path.basename(rep_path)
                data_from_rep = process_repetition_folder(rep_path, meta_run_name, condition_name, repetition_name)
                aggregated_data.extend(data_from_rep)
    return aggregated_data


def perform_overall_analysis(all_runs_data: List[Dict[str, Any]], base_results_dir_for_saving: str) -> Optional[pd.DataFrame]:
    """
    Führt eine Gesamtanalyse der gesammelten Laufdaten durch.
    Die Funktion konvertiert die Daten in einen DataFrame, bereinigt sie,
    gruppiert sie nach Konfigurationsparametern, aggregiert die Metriken (Mittelwert, Standardabweichung)
    und sortiert das Ergebnis.
    """
    print("\n--- Gesamtanalyse wird durchgeführt ---")
    if not all_runs_data:
        print("Keine Daten für Gesamtanalyse.")
        return None

    df = pd.DataFrame(all_runs_data)
    if df.empty:
        print("DataFrame ist leer.")
        return df

    print(f"Insgesamt {len(df)} Log-Einträge verarbeitet.")

    # Liste aller potenziell numerischen Spalten zur Konvertierung
    # ANPASSUNG: Precision (macro) und Recall (macro) für alle Modi hinzugefügt
    numeric_cols_to_convert = [
        'cfg_n_hidden', 'cfg_quant_bits', 'cfg_unpruned_neurons', 'cfg_dropout_rate1', 'cfg_dropout_rate2',
        'cfg_learning_rate', 'cfg_manual_batch_size', 'cfg_dataloader_batch_size', 'cfg_subset_fraction',
        'cfg_focal_alpha', 'cfg_focal_gamma', 'cfg_class_weight_beta', 'cfg_weight_decay',
        'cfg_scheduler_factor', 'cfg_scheduler_patience', 'cfg_scheduler_min_lr',
        'cfg_scheduler_step_size', 'cfg_scheduler_gamma',
        'cfg_early_stop_patience', 'cfg_num_epochs', 'epochs_run_actual', 'epoch_of_best_f1w',
        'best_f1_weighted_val', 'f1_macro_at_best_f1w_val', 'roc_auc_at_best_f1w_val',
        'pytorch_json_accuracy', 'pytorch_json_f1_macro', 'pytorch_json_f1_weighted',
        'pytorch_json_roc_auc_macro', 'pytorch_json_mcc', 'pytorch_json_infer_time_s', 'pytorch_json_time_per_1000',
        'pytorch_json_avg_cpu', 'pytorch_json_peak_ram_mb', 'fhe_compilation_time_s',
        'fhe_simulate_json_accuracy', 'fhe_simulate_json_f1_macro', 'fhe_simulate_json_f1_weighted',
        'fhe_simulate_json_roc_auc_macro', 'fhe_simulate_json_mcc', 'fhe_simulate_json_infer_time_s', 'fhe_simulate_json_time_per_1000',
        'fhe_simulate_json_avg_cpu', 'fhe_simulate_json_peak_ram_mb',
        'fhe_execute_json_accuracy', 'fhe_execute_json_f1_macro', 'fhe_execute_json_f1_weighted',
        'fhe_execute_json_roc_auc_macro', 'fhe_execute_json_mcc', 'fhe_execute_json_infer_time_s', 'fhe_execute_json_time_per_1000',
        'fhe_execute_json_avg_cpu', 'fhe_execute_json_peak_ram_mb', 'avg_final_sparsity',
        'pytorch_txt_Acc', 'pytorch_txt_Precm', 'pytorch_txt_Recm', 'pytorch_txt_F1m', 'pytorch_txt_F1w', 'pytorch_txt_MCC', 'pytorch_txt_ROC-AUCm',
        'pytorch_txt_total_infer_time_s_txt', 'pytorch_txt_time_per_1000_samples_s_txt',
        'fhe_simulate_txt_Acc', 'fhe_simulate_txt_Precm', 'fhe_simulate_txt_Recm', 'fhe_simulate_txt_F1m', 'fhe_simulate_txt_F1w', 'fhe_simulate_txt_MCC', 'fhe_simulate_txt_ROC-AUCm',
        'fhe_simulate_txt_total_infer_time_s_txt', 'fhe_simulate_txt_time_per_1000_samples_s_txt', 'fhe_simulate_txt_fhe_compilation_time_s_txt',
        'fhe_execute_txt_Acc', 'fhe_execute_txt_Precm', 'fhe_execute_txt_Recm', 'fhe_execute_txt_F1m', 'fhe_execute_txt_F1w', 'fhe_execute_txt_MCC', 'fhe_execute_txt_ROC-AUCm',
        'fhe_execute_txt_total_infer_time_s_txt', 'fhe_execute_txt_time_per_1000_samples_s_txt', 'fhe_execute_txt_fhe_compilation_time_s_txt',
    ]
    for col in numeric_cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parameter für die Gruppierung (Architektur, Konfiguration)
    grouping_params = [
        'dataset_name', 'model_type_cfg', 'cfg_quant_bits', 'cfg_unpruned_neurons', 'cfg_n_hidden',
        'cfg_dropout_rate1', 'cfg_dropout_rate2', 'cfg_learning_rate',
        'cfg_criterion_name', 'cfg_calc_class_weights', 'cfg_class_weight_method',
        'cfg_optimizer_name', 'cfg_scheduler_name', 'cfg_num_epochs'
    ]
    grouping_params = [col for col in grouping_params if col in df.columns and df[col].notna().any()]

    # Metriken für die Aggregation (Mittelwert und Standardabweichung)
    # ANPASSUNG: Precision (macro) und Recall (macro) für alle Modi hinzugefügt
    metrics_to_aggregate = [
        'best_f1_weighted_val', 'epochs_run_actual', 'f1_macro_at_best_f1w_val', 'roc_auc_at_best_f1w_val',
        'pytorch_json_f1_weighted', 'pytorch_json_roc_auc_macro', 'pytorch_json_mcc', 'pytorch_json_accuracy',
        'pytorch_txt_F1w', 'pytorch_txt_ROC-AUCm', 'pytorch_txt_MCC', 'pytorch_txt_Acc', 
        'pytorch_txt_Precm', 'pytorch_txt_Recm', # PyTorch Text Precision/Recall
        'fhe_simulate_json_f1_weighted', 'fhe_simulate_json_roc_auc_macro', 'fhe_simulate_json_mcc', 'fhe_simulate_json_accuracy',
        'fhe_simulate_txt_F1w', 'fhe_simulate_txt_ROC-AUCm', 'fhe_simulate_txt_MCC', 'fhe_simulate_txt_Acc',
        'fhe_simulate_txt_Precm', 'fhe_simulate_txt_Recm', # FHE Simulate Text Precision/Recall
        'fhe_execute_json_f1_weighted', 'fhe_execute_json_roc_auc_macro', 'fhe_execute_json_mcc', 'fhe_execute_json_accuracy',
        'fhe_execute_txt_F1w', 'fhe_execute_txt_ROC-AUCm', 'fhe_execute_txt_MCC', 'fhe_execute_txt_Acc',
        'fhe_execute_txt_Precm', 'fhe_execute_txt_Recm', # FHE Execute Text Precision/Recall
        'fhe_compilation_time_s', 'avg_final_sparsity'
    ]
    metrics_to_aggregate = [col for col in metrics_to_aggregate if col in df.columns and df[col].notna().any()]

    df_averaged_sorted = pd.DataFrame()
    if grouping_params and metrics_to_aggregate:
        print("\n\n--- Durchschnittliche Performance pro Konfiguration ---")
        df_grouped_analysis = df.copy()

        for col in grouping_params:
            if df_grouped_analysis[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_grouped_analysis[col].dtype):
                    df_grouped_analysis[col] = df_grouped_analysis[col].fillna(-99999.0)
                else:
                    df_grouped_analysis[col] = df_grouped_analysis[col].astype(str).fillna('N/A_cfg')

        try:
            agg_dict: Dict[str, Any] = {metric: ['mean', 'std'] for metric in metrics_to_aggregate}
            if metrics_to_aggregate: # Sicherstellen, dass die Liste nicht leer ist
                 agg_dict[metrics_to_aggregate[0]] = ['mean', 'std', 'size'] # 'size' für num_repetitions

            grouped_performance = df_grouped_analysis.groupby(grouping_params, dropna=False).agg(agg_dict)

            new_cols = []
            for col_tuple in grouped_performance.columns.values:
                if col_tuple[1] == 'size':
                    new_cols.append('num_repetitions')
                else:
                    new_cols.append(f"{col_tuple[0]}_{col_tuple[1]}")
            grouped_performance.columns = new_cols
            
            grouped_performance = grouped_performance.reset_index()
            
            sort_by_keys = [p for p in ['dataset_name', 'model_type_cfg', 'quant_bits', 'unpruned_neurons', 'n_hidden'] if p in grouped_performance.columns]
            
            # Spalten umbenennen (cfg_ entfernen) bevor sortiert wird, falls die sort_by_keys das Präfix noch erwarten
            # Es ist besser, die Sortierschlüssel auf die *ursprünglichen* Namen (mit cfg_) zu beziehen
            # und die Umbenennung danach durchzuführen.
            temp_sort_by_keys_with_cfg = [
                'dataset_name', 'model_type_cfg', 
                'cfg_quant_bits', 'cfg_unpruned_neurons', 'cfg_n_hidden'
            ]
            actual_sort_by_keys = [key for key in temp_sort_by_keys_with_cfg if key in grouped_performance.columns]


            if actual_sort_by_keys:
                 df_averaged_sorted = grouped_performance.sort_values(by=actual_sort_by_keys, ascending=True)
            else:
                 df_averaged_sorted = grouped_performance
            
            for col in df_averaged_sorted.select_dtypes(include=['float', 'float64']).columns:
                df_averaged_sorted[col] = df_averaged_sorted[col].round(4)

            # Spalten umbenennen, um "cfg_" zu entfernen
            df_averaged_sorted.columns = [c.replace('cfg_', '') for c in df_averaged_sorted.columns]

            print("\n** Durchschnittliche Metriken (sortiert & gerundet):**")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)
            pd.set_option('display.max_colwidth', 100)
            pd.set_option('display.precision', 4)
            print(df_averaged_sorted.to_string())
            pd.reset_option('all')

        except Exception as e:
            print(f"Fehler bei Gruppierung/Aggregation: {e}\n{traceback.format_exc()}")
            return df 
    else:
        print("WARNUNG: Keine gültigen Spalten für die Aggregation gefunden.")
        return df

    return df_averaged_sorted if not df_averaged_sorted.empty else df
    

def main_analysis_pipeline(base_results_dir: str) -> None:
    """
    Hauptpipeline zur Analyse der Experiment-Logs.
    Sucht nach meta_run_* Ordnern oder direkt nach Condition-Ordnern.
    """
    # KORREKTUR: Die Initialisierung der Liste ist hier entscheidend.
    all_aggregated_data: List[Dict[str, Any]] = []
    
    # Sucht nach meta_run_* Ordnern im base_results_dir
    meta_run_folders = glob.glob(os.path.join(base_results_dir, META_RUN_PATTERN))
    is_direct_conditions_search = False

    if not meta_run_folders:
        print(f"Keine Ordner gefunden, die dem Muster '{META_RUN_PATTERN}' in '{base_results_dir}' entsprechen.")
        # Prüfe, ob base_results_dir selbst Unterordner hat, die Conditions sein könnten
        if os.path.isdir(base_results_dir):
            # Liste nur Verzeichnisse auf, die nicht .ipynb_checkpoints oder run_configs sind oder CSVs
            potential_conditions = [
                d for d in os.listdir(base_results_dir)
                if os.path.isdir(os.path.join(base_results_dir, d)) and
                   d not in [".ipynb_checkpoints", "run_configs"] and
                   not d.endswith(".csv") and
                   not d.startswith(".") # Ignoriere versteckte Ordner
            ]
            if potential_conditions:
                print(f"Behandle '{base_results_dir}' als Hauptverzeichnis, das Condition-Ordner enthält.")
                meta_run_folders = [base_results_dir] # Behandle den Basisordner als einen "Meta-Lauf"
                is_direct_conditions_search = True
            else:
                print(f"Auch keine verarbeitbaren Condition-Unterordner direkt in '{base_results_dir}' gefunden.")
                return
        else:
            print(f"Pfad '{base_results_dir}' ist kein gültiges Verzeichnis.")
            return
    else:
        print(f"{len(meta_run_folders)} Ordner gefunden, die mit '{META_RUN_PATTERN}' beginnen.")

    # Verarbeite die gefundenen Ordner
    for top_level_path_for_conditions in meta_run_folders:
        meta_run_name = os.path.basename(top_level_path_for_conditions) if not is_direct_conditions_search else "direct_scan"
        print(f"\nVerarbeite Top-Level/Meta-Run Ordner: {meta_run_name}")
        
        for condition_folder_name in os.listdir(top_level_path_for_conditions):
            condition_folder_path = os.path.join(top_level_path_for_conditions, condition_folder_name)
            
            # Überspringe Dateien und bekannte Nicht-Condition-Ordner
            if not os.path.isdir(condition_folder_path) or \
               condition_folder_name == "run_configs" or \
               condition_folder_name.endswith(".csv") or \
               condition_folder_name.startswith("."):
                continue
            
            print(f"  Prüfe Condition-Ordner: {condition_folder_name}...")
            data_from_condition = find_and_process_repetitions(
                condition_folder_path, meta_run_name, condition_folder_name
            )
            all_aggregated_data.extend(data_from_condition)
            
    # --- Gesamtdaten speichern und analysieren ---
    if all_aggregated_data:
        original_df = pd.DataFrame(all_aggregated_data)
        if not original_df.empty:
            timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
            original_summary_filename = f"all_runs_detailed_data_{timestamp_str}.csv"
            
            save_dir_detailed = os.path.dirname(base_results_dir) if os.path.basename(base_results_dir) == "parameter_analysis_runs" else base_results_dir
            
            original_summary_save_path = os.path.join(save_dir_detailed, original_summary_filename)
            try:
                # Nutze Komma als Trennzeichen
                original_df.to_csv(original_summary_save_path, index=False, sep=',')
                print(f"\nDetaillierte Daten aller Läufe als '{original_summary_filename}' gespeichert in: {save_dir_detailed}")
            except Exception as e:
                print(f"\nFehler beim Speichern der detaillierten Daten als CSV: {e}")
    else:
        print("Keine Daten gesammelt, daher keine detaillierte CSV-Datei erstellt.")
        return 

    averaged_and_sorted_df = perform_overall_analysis(all_aggregated_data, base_results_dir)

    if averaged_and_sorted_df is not None and not averaged_and_sorted_df.empty:
        # Überprüfe, ob die Aggregation stattgefunden hat, indem nach der 'num_repetitions'-Spalte gesucht wird
        is_averaged_df = 'num_repetitions' in averaged_and_sorted_df.columns
        if is_averaged_df :
            averaged_summary_filename = f"averaged_runs_summary_sorted_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            save_dir_avg = os.path.dirname(base_results_dir) if os.path.basename(base_results_dir) == "parameter_analysis_runs" else base_results_dir
            averaged_summary_save_path = os.path.join(save_dir_avg, averaged_summary_filename)
            try:
                # Nutze Komma als Trennzeichen und speichere den Index nicht mit
                averaged_and_sorted_df.to_csv(averaged_summary_save_path, index=False, sep=',')
                print(f"\nDurchschnittliche und sortierte Analyse als '{averaged_summary_filename}' gespeichert in: {save_dir_avg}")
            except Exception as e:
                print(f"\nFehler beim Speichern der aggregierten CSV: {e}")
        elif averaged_and_sorted_df is original_df: 
            print("\nKeine separate Datei für Durchschnittswerte gespeichert (Aggregation fehlgeschlagen/nicht nötig oder keine Gruppen).")





if __name__ == "__main__":
    print("--- Start des Analyse-Skripts für Trainingsläufe ---")
    try:
        current_script_dir_main = os.path.dirname(os.path.abspath(__file__))
        # Wenn das Skript in src/utils liegt, ist project_root_main FHE_athens/
        project_root_main = os.path.dirname(os.path.dirname(current_script_dir_main))
        # Standard-Basispfad ist jetzt der 'qat_runs' Ordner, da 'parameter_analysis_runs' ein Unterordner davon ist.
        default_base_dir_calculated = os.path.join(project_root_main, "results", "qat_runs")
    except NameError: 
        project_root_main = os.getcwd() 
        default_base_dir_calculated = os.path.join(project_root_main, "results", "qat_runs")
        print(f"WARNUNG: __file__ nicht definiert, PWD als Basis für Default: {default_base_dir_calculated}")

    base_directory_to_analyze = default_base_dir_calculated
    print(f"INFO: Standard-Basispfad für die Suche nach 'meta_run_*' oder 'parameter_analysis_runs': {default_base_dir_calculated}")
    
    user_path_input = input(
        f"Bitte den Pfad zum Ordner eingeben, der die 'meta_run_*' Ordner ODER direkt "
        f"den 'parameter_analysis_runs' Ordner enthält.\n"
        f"(Beispiel: '{default_base_dir_calculated}' oder '{os.path.join(default_base_dir_calculated, 'parameter_analysis_runs')}')\n"
        f"(Enter für Standardpfad '{default_base_dir_calculated}', der dann nach 'meta_run_*' durchsucht wird ODER als Basis für 'parameter_analysis_runs' dient): "
    ).strip()

    if user_path_input:
        base_directory_to_analyze = user_path_input
    
    if not os.path.isdir(base_directory_to_analyze):
        print(f"FEHLER: Der angegebene Pfad '{base_directory_to_analyze}' ist kein gültiges Verzeichnis.")
    else:
        print(f"Verwende Analysepfad für diese Sitzung: {base_directory_to_analyze}")
        # Überprüfe, ob der Nutzer direkt den parameter_analysis_runs Ordner angegeben hat
        # oder ob wir in einem übergeordneten Ordner nach meta_run_* suchen sollen.
        if os.path.basename(base_directory_to_analyze) == "parameter_analysis_runs":
            print(f"Analysiere direkt den Ordner: {base_directory_to_analyze}")
            main_analysis_pipeline(base_directory_to_analyze)
        else:
            # Standardverhalten: Suche nach meta_run_* Ordnern oder behandle base_directory_to_analyze als direkten Container für Condition-Ordner
            main_analysis_pipeline(base_directory_to_analyze)
            
    print("\n--- Analyse-Skript beendet ---")