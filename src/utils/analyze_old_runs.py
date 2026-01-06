import os
import json
import pandas as pd
import re # Für das Parsen der Textdateien

# --- Konfiguration ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_script_dir)
PROJECT_ROOT = os.path.dirname(src_dir)
OLD_RUNS_BASE_DIR = os.path.join(PROJECT_ROOT, "results", "qat_runs", "old_runs")

# --- Hilfsfunktionen zum Parsen ---

def parse_json_log(file_path: str) -> dict:
    """Liest relevante Daten aus einer JSON-Logdatei."""
    data = {}
    try:
        with open(file_path, 'r') as f:
            log_content = json.load(f)

        # Wichtige Hyperparameter und Konfigurationen
        data['run_id'] = log_content.get("run_overview", {}).get("run_timestamp", os.path.basename(os.path.dirname(file_path))) # Fallback auf Ordnernamen
        data['timestamp'] = log_content.get("run_overview", {}).get("run_timestamp")
        
        model_params = log_content.get("model_details_runtime", {}) # Geändert von "input_configuration_from_yaml", "model_params"
        data['n_hidden'] = model_params.get("n_hidden_applied")
        data['quant_bits'] = model_params.get("quantization_bits_applied")
        data['unpruned_neurons'] = model_params.get("unpruned_neurons_applied")
        data['dropout_rate1'] = model_params.get("dropout_rate1_applied")
        data['dropout_rate2'] = model_params.get("dropout_rate2_applied")

        training_params_optimizer = log_content.get("training_execution_summary", {}).get("optimizer_details_runtime", {})
        data['learning_rate'] = training_params_optimizer.get("initial_learning_rate_runtime")
        data['optimizer'] = training_params_optimizer.get("name_runtime")
        
        training_params_criterion = log_content.get("training_execution_summary", {}).get("criterion_details_runtime", {})
        data['criterion'] = training_params_criterion.get("name_runtime")
        data['class_weights_applied'] = training_params_criterion.get("class_weights_applied_runtime")


        # Beste Validierungsmetriken
        best_val_metrics = log_content.get("best_model_metrics_achieved_val", {})
        data['best_val_f1_weighted'] = best_val_metrics.get("best_f1_weighted_val")
        data['best_val_f1_macro'] = best_val_metrics.get("f1_macro_at_best_f1w_val")
        data['best_val_roc_auc'] = best_val_metrics.get("roc_auc_at_best_f1w_val")
        data['epochs_run'] = log_content.get("training_execution_summary", {}).get("epochs_run_actual")

        # Zusammengefasste Evaluierungsergebnisse aus dem JSON
        eval_details = log_content.get("evaluation_execution_details", {})
        
        pytorch_summary = eval_details.get("pytorch_eval_summary", {})
        data['pytorch_test_accuracy'] = pytorch_summary.get("accuracy")
        data['pytorch_test_f1_weighted'] = pytorch_summary.get("f1_weighted")
        data['pytorch_test_f1_macro'] = pytorch_summary.get("f1_macro")
        data['pytorch_test_roc_auc'] = pytorch_summary.get("roc_auc_macro_ovr")
        data['pytorch_infer_time_ms_per_1k'] = pytorch_summary.get("time_per_1000_samples_s") # Ist in Sekunden, Umrechnung in ms später?

        fhe_summary = eval_details.get("fhe_eval_summary", {})
        data['fhe_test_accuracy'] = fhe_summary.get("accuracy")
        data['fhe_test_f1_weighted'] = fhe_summary.get("f1_weighted")
        data['fhe_test_f1_macro'] = fhe_summary.get("f1_macro")
        data['fhe_test_roc_auc'] = fhe_summary.get("roc_auc_macro_ovr")
        data['fhe_infer_time_ms_per_1k'] = fhe_summary.get("time_per_1000_samples_s") # Ist in Sekunden
        data['fhe_mode'] = fhe_summary.get("fhe_mode")
        data['fhe_compilation_time_s'] = eval_details.get("fhe_compilation_time_s")

    except Exception as e:
        print(f"Error parsing JSON file {file_path}: {e}")
    return data

def parse_eval_text_file(file_path: str) -> dict:
    """
    Liest spezifische Metriken aus einer Text-Evaluierungsdatei.
    Diese Funktion muss stark an das Format deiner Textdateien angepasst werden!
    """
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Beispiel-Extraktionen (bitte anpassen!):
        # Annahme: "Acc: 0.1234"
        acc_match = re.search(r"Acc:\s*([0-9\.]+)", content)
        if acc_match:
            metrics['text_accuracy'] = float(acc_match.group(1))

        # Annahme: "F1\(w\):\s*([0-9\.]+)" für gewichteten F1
        f1w_match = re.search(r"F1\(w\):\s*([0-9\.]+)", content)
        if f1w_match:
            metrics['text_f1_weighted'] = float(f1w_match.group(1))
            
        # Annahme: "F1\(m\):\s*([0-9\.]+)" für Macro F1
        f1m_match = re.search(r"F1\(m\):\s*([0-9\.]+)", content)
        if f1m_match:
            metrics['text_f1_macro'] = float(f1m_match.group(1))

        # Annahme: "ROC-AUC\(m\):\s*([0-9\.]+)"
        roc_match = re.search(r"ROC-AUC\(m\):\s*([0-9\.]+)", content)
        if roc_match:
            metrics['text_roc_auc'] = float(roc_match.group(1))
            
        # Annahme: "Total PyTorch Infer Time: 12.34s" oder "Total FHE Infer Time: 123.45s"
        time_match = re.search(r"Total (?:PyTorch|FHE) Infer Time:\s*([0-9\.]+)s", content)
        if time_match:
            metrics['text_total_infer_time_s'] = float(time_match.group(1))
            
        # Annahme: "PyTorch Time/1000 samples: 0.1234s" oder "FHE Time/1000 samples: 1.2345s"
        time_1k_match = re.search(r"(?:PyTorch|FHE) Time/1000 samples:\s*([0-9\.]+)s", content)
        if time_1k_match:
            metrics['text_time_per_1000_samples_s'] = float(time_1k_match.group(1))

        if "FHE Model Eval" in content:
            mode_match = re.search(r"\((\w+)\)\s*===", content) # z.B. (simulate) ===
            if mode_match:
                metrics['text_fhe_mode'] = mode_match.group(1)
            comp_time_match = re.search(r"FHE Compilation Time:\s*([0-9\.]+|N/A.*?)s", content)
            if comp_time_match:
                 metrics['text_fhe_compilation_time_s'] = comp_time_match.group(1)


    except Exception as e:
        print(f"Error parsing text file {file_path}: {e}")
    return metrics

# --- Hauptlogik ---
all_experiment_data = []

if not os.path.isdir(OLD_RUNS_BASE_DIR):
    print(f"FEHLER: Basisverzeichnis für alte Läufe nicht gefunden: {OLD_RUNS_BASE_DIR}")
else:
    print(f"Durchsuche Verzeichnis: {OLD_RUNS_BASE_DIR}")
    # Iteriere durch jeden Ordner im OLD_RUNS_BASE_DIR
    for run_folder_name in os.listdir(OLD_RUNS_BASE_DIR):
        run_folder_path = os.path.join(OLD_RUNS_BASE_DIR, run_folder_name)
        if not os.path.isdir(run_folder_path):
            continue

        print(f"\nVerarbeite Lauf: {run_folder_name}")
        current_run_data = {'run_folder': run_folder_name}

        # Finde und parse JSON-Logdatei
        json_log_path = None
        for file_name in os.listdir(run_folder_path):
            if file_name.endswith("_full_run_log.json"):
                json_log_path = os.path.join(run_folder_path, file_name)
                break
        
        if json_log_path:
            print(f"  Gefundenes JSON-Log: {json_log_path}")
            json_data = parse_json_log(json_log_path)
            current_run_data.update(json_data)
        else:
            print(f"  WARNUNG: Kein JSON-Log für {run_folder_name} gefunden.")

        # Finde und parse PyTorch Eval Textdatei
        torch_eval_txt_path = None
        for file_name in os.listdir(run_folder_path):
            if file_name.startswith("eval_torch_") and file_name.endswith(".txt"):
                torch_eval_txt_path = os.path.join(run_folder_path, file_name)
                break # Nimm die erste gefundene Datei
        
        if torch_eval_txt_path:
            print(f"  Gefundenes PyTorch Eval TXT: {torch_eval_txt_path}")
            torch_txt_data = parse_eval_text_file(torch_eval_txt_path)
            # Füge Präfix hinzu, um Überschreibungen mit JSON-Daten zu vermeiden, falls Schlüssel gleich sind
            current_run_data.update({f"pytorch_{k}": v for k, v in torch_txt_data.items()})
        else:
            print(f"  INFO: Kein PyTorch Eval TXT für {run_folder_name} gefunden.")

        # Finde und parse FHE Eval Textdatei
        fhe_eval_txt_path = None
        for file_name in os.listdir(run_folder_path):
            if file_name.startswith("eval_fhe_") and file_name.endswith(".txt"):
                fhe_eval_txt_path = os.path.join(run_folder_path, file_name)
                break # Nimm die erste gefundene Datei

        if fhe_eval_txt_path:
            print(f"  Gefundenes FHE Eval TXT: {fhe_eval_txt_path}")
            fhe_txt_data = parse_eval_text_file(fhe_eval_txt_path)
            current_run_data.update({f"fhe_{k}": v for k, v in fhe_txt_data.items()})
        else:
            print(f"  INFO: Kein FHE Eval TXT für {run_folder_name} gefunden.")
            
        all_experiment_data.append(current_run_data)

# Erstelle Pandas DataFrame
df_overview = pd.DataFrame(all_experiment_data)

# --- Datenbereinigung und -aufbereitung (Beispiele) ---
# Konvertiere numerische Spalten, behandle Fehler (setze auf NaN, falls nicht konvertierbar)
numeric_cols = [
    'n_hidden', 'quant_bits', 'unpruned_neurons', 'learning_rate', 'epochs_run',
    'best_val_f1_weighted', 'best_val_f1_macro', 'best_val_roc_auc',
    'pytorch_test_accuracy', 'pytorch_test_f1_weighted', 'pytorch_test_f1_macro',
    'pytorch_test_roc_auc', 'pytorch_infer_time_ms_per_1k',
    'pytorch_text_accuracy', 'pytorch_text_f1_weighted', 'pytorch_text_f1_macro',
    'pytorch_text_roc_auc', 'pytorch_text_total_infer_time_s', 'pytorch_text_time_per_1000_samples_s',
    'fhe_test_accuracy', 'fhe_test_f1_weighted', 'fhe_test_f1_macro',
    'fhe_test_roc_auc', 'fhe_infer_time_ms_per_1k', 'fhe_compilation_time_s',
    'fhe_text_accuracy', 'fhe_text_f1_weighted', 'fhe_text_f1_macro',
    'fhe_text_roc_auc', 'fhe_text_total_infer_time_s', 'fhe_text_time_per_1000_samples_s'
]
for col in numeric_cols:
    if col in df_overview.columns:
        df_overview[col] = pd.to_numeric(df_overview[col], errors='coerce')

# Umrechnung Inferenzzeit von Sekunden in Millisekunden, falls gewünscht (und in Sekunden gespeichert)
# time_cols_to_ms = ['pytorch_infer_time_ms_per_1k', 'fhe_infer_time_ms_per_1k', 
# 'pytorch_text_time_per_1000_samples_s', 'fhe_text_time_per_1000_samples_s']
# for col in time_cols_to_ms:
# if col in df_overview.columns:
# df_overview[col] = df_overview[col] * 1000 # Wenn die Quelle in s war

# --- Analyse und Anzeige (Beispiele) ---
if not df_overview.empty:
    print("\n\n--- Übersicht der extrahierten Experimentdaten ---")
    pd.set_option('display.max_columns', None) # Alle Spalten anzeigen
    pd.set_option('display.width', 200) # Breite der Anzeige
    print(df_overview.head())

    # Definiere die für dich wichtigsten Hyperparameter und Metriken für die Analyse
    key_hyperparameters = ['n_hidden', 'quant_bits', 'unpruned_neurons', 'learning_rate', 'criterion']
    key_metrics = [
        'best_val_f1_weighted', 
        'pytorch_test_f1_weighted', 'pytorch_test_roc_auc', 'pytorch_infer_time_ms_per_1k',
        'fhe_test_f1_weighted', 'fhe_test_roc_auc', 'fhe_infer_time_ms_per_1k', 'fhe_mode'
    ]
    
    # Filtere Spalten für eine fokussiertere Ansicht
    display_columns = ['run_folder'] + [col for col in key_hyperparameters if col in df_overview.columns] + \
                                     [col for col in key_metrics if col in df_overview.columns]
    df_display = df_overview[display_columns]

    print("\n\n--- Performance-Übersicht (sortiert nach bestem Validierungs-F1) ---")
    # Sortiere nach einer wichtigen Metrik (z.B. bester Validierungs-F1)
    if 'best_val_f1_weighted' in df_display.columns:
        print(df_display.sort_values(by='best_val_f1_weighted', ascending=False))
    else:
        print(df_display)

    # Beispiel: Gruppieren nach Hyperparametern und Mittelwert der Metriken anzeigen
    # Stelle sicher, dass die Hyperparameter-Spalten keine NaN-Werte für die Gruppierung haben
    # oder fülle sie ggf. auf (z.B. mit einem Platzhalter wie 'N/A_str')
    # relevant_hyperparams_for_grouping = [hp for hp in key_hyperparameters if hp in df_overview.columns]
    # if relevant_hyperparams_for_grouping:
    # print("\n\n--- Durchschnittliche Performance pro Hyperparameter-Kombination ---")
    # try:
    # # Fürs Gruppieren NaNs in Hyperparametern ggf. auffüllen, damit sie gruppiert werden
    # df_filled_hyperparams = df_overview.copy()
    # for hp_col in relevant_hyperparams_for_grouping:
    # if df_filled_hyperparams[hp_col].isnull().any():
    # # Konvertiere zu String, um verschiedene Typen mischen zu können + Platzhalter
    # df_filled_hyperparams[hp_col] = df_filled_hyperparams[hp_col].astype(str).fillna('N/A_val')
    #
    # grouped_performance = df_filled_hyperparams.groupby(relevant_hyperparams_for_grouping)[key_metrics].mean()
    # print(grouped_performance)
    # except Exception as e_group:
    # print(f"Fehler beim Gruppieren: {e_group}")
    # else:
    # print("Keine relevanten Hyperparameter zum Gruppieren gefunden oder DataFrame ist leer.")

    # Speichere den DataFrame als CSV für weitere Analysen (optional)
    try:
        output_csv_path = os.path.join(OLD_RUNS_BASE_DIR, "old_runs_analysis_summary.csv")
        df_overview.to_csv(output_csv_path, index=False)
        print(f"\n\nÜbersichts-DataFrame gespeichert als CSV: {output_csv_path}")
    except Exception as e_csv:
        print(f"Konnte DataFrame nicht als CSV speichern: {e_csv}")
else:
    print("Keine Daten zum Analysieren gefunden.")