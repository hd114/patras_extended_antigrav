# src/utils/dashboard_data_loader.py

import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional

def load_experiment_data(results_base_dir: str) -> pd.DataFrame:
    """
    Lädt Experimentdaten aus allen JSON-Logdateien in einem Basisverzeichnis.

    Args:
        results_base_dir: Der Pfad zum Basisverzeichnis, das die
                          Unterordner der einzelnen Experimentläufe enthält.
                          (z.B. "results/default_runs" oder "results/qat_runs")

    Returns:
        Ein Pandas DataFrame mit den aggregierten Daten aller Läufe.
        Jede Zeile repräsentiert einen Lauf.
    """
    all_runs_data: List[Dict[str, Any]] = []

    if not os.path.isdir(results_base_dir):
        print(f"FEHLER: Basisverzeichnis für Ergebnisse nicht gefunden: {results_base_dir}")
        return pd.DataFrame(all_runs_data)

    for run_dir_name in os.listdir(results_base_dir):
        run_dir_path = os.path.join(results_base_dir, run_dir_name)
        if not os.path.isdir(run_dir_path):
            continue

        json_log_file: Optional[str] = None
        # Finde die *_full_run_log.json Datei im Run-Verzeichnis
        for file_name in os.listdir(run_dir_path):
            if file_name.endswith("_full_run_log.json"):
                json_log_file = os.path.join(run_dir_path, file_name)
                break
        
        if json_log_file:
            try:
                with open(json_log_file, 'r') as f:
                    data = json.load(f)
                
                # Extrahiere relevante Informationen für die Übersichtstabelle
                # Die Pfade zu den Metriken wurden an die neue Struktur angepasst
                eval_details = data.get("evaluation_execution_details", {})
                pytorch_summary = eval_details.get("pytorch_eval_summary", {})
                fhe_summary = eval_details.get("fhe_eval_summary", {})

                run_summary = {
                    "run_id": run_dir_name,
                    "timestamp": data.get("run_overview", {}).get("run_timestamp"),
                    "dataset": data.get("run_overview", {}).get("dataset_configured"),
                    "n_hidden": data.get("model_details_runtime", {}).get("n_hidden_applied"),
                    "quant_bits": data.get("model_details_runtime", {}).get("quantization_bits_applied"),
                    "unpruned_neurons": data.get("model_details_runtime", {}).get("unpruned_neurons_applied"),
                    "learning_rate": data.get("training_execution_summary", {}).get("optimizer_details_runtime", {}).get("initial_learning_rate_runtime"),
                    "criterion": data.get("training_execution_summary", {}).get("criterion_details_runtime", {}).get("name_runtime"),
                    "epochs_run": data.get("training_execution_summary", {}).get("epochs_run_actual"),
                    "best_f1_weighted_val": data.get("best_model_metrics_achieved_val", {}).get("best_f1_weighted_val"),
                    
                    # Zugriff auf die detaillierten Test-Metriken
                    "pytorch_test_acc": pytorch_summary.get("accuracy"),
                    "pytorch_test_f1w": pytorch_summary.get("f1_weighted"),
                    "pytorch_test_f1m": pytorch_summary.get("f1_macro"),
                    "pytorch_test_roc_auc": pytorch_summary.get("roc_auc_macro_ovr"),
                    "pytorch_infer_time_1k": pytorch_summary.get("time_per_1000_samples_s"),
                    
                    "fhe_test_acc": fhe_summary.get("accuracy"),
                    "fhe_test_f1w": fhe_summary.get("f1_weighted"),
                    "fhe_test_f1m": fhe_summary.get("f1_macro"),
                    "fhe_test_roc_auc": fhe_summary.get("roc_auc_macro_ovr"),
                    "fhe_infer_time_1k": fhe_summary.get("time_per_1000_samples_s"),
                    "fhe_mode": fhe_summary.get("fhe_mode"),
                    
                    "json_log_path": data.get("output_artifact_paths", {}).get("json_log_file_self")
                }
                all_runs_data.append(run_summary)
            except Exception as e:
                print(f"Error loading or parsing {json_log_file}: {e}")
                # Optional: Logge den Traceback für Debugging
                # import traceback
                # print(traceback.format_exc())
                
    return pd.DataFrame(all_runs_data)

if __name__ == '__main__':
    # --- Testbereich für den Datenlader ---
    # Dieser Block wird nur ausgeführt, wenn das Skript direkt gestartet wird.
    
    # Bestimme das Projekt-Root-Verzeichnis relativ zum aktuellen Skriptpfad.
    # Annahme: Dieses Skript (dashboard_data_loader.py) ist in src/utils/
    current_script_path = os.path.abspath(__file__)
    # z.B. /pfad/zum/projekt/src/utils/dashboard_data_loader.py
    utils_dir = os.path.dirname(current_script_path)
    # z.B. /pfad/zum/projekt/src/utils/
    src_dir = os.path.dirname(utils_dir)
    # z.B. /pfad/zum/projekt/src/
    project_root_dir = os.path.dirname(src_dir)
    # z.B. /pfad/zum/projekt/

    # Definiere den Pfad zum Basisverzeichnis der Ergebnisse.
    # Passe 'default_runs' oder 'qat_runs' an, je nachdem, was in deiner
    # `config.yaml` unter `run_settings.results_base_dir` konfiguriert ist
    # und wo deine Trainingsskripte die Ergebnisse ablegen.
    # Beispiel: Wenn results_base_dir = "results/meine_laeufe" ist:
    # test_results_dir = os.path.join(project_root_dir, "results", "meine_laeufe")
    
    # Standardannahme basierend auf vorherigen Diskussionen
    configured_results_folder_name = "qat_runs" # Oder "qat_runs", etc.
    test_results_dir = os.path.join(project_root_dir, "results", configured_results_folder_name)

    print(f"INFO: Testweiser Datenladevorgang gestartet.")
    print(f"INFO: Versuche, Daten aus dem Verzeichnis zu laden: {test_results_dir}")

    if not os.path.isdir(test_results_dir):
        print(f"FEHLER: Das Verzeichnis '{test_results_dir}' wurde nicht gefunden.")
        print("       Bitte stelle sicher, dass der Pfad korrekt ist und dass bereits")
        print("       Trainingsläufe mit JSON-Logs in diesem Verzeichnis existieren.")
        print("       Passe ggf. 'configured_results_folder_name' im Testblock dieses Skripts an.")
    else:
        df_experiments = load_experiment_data(test_results_dir)
        if not df_experiments.empty:
            print(f"INFO: Erfolgreich {len(df_experiments)} Experiment(e) geladen.")
            print("INFO: Spalten im DataFrame:", list(df_experiments.columns))
            print("\nINFO: Erste 5 Zeilen der geladenen Daten:")
            print(df_experiments.head().to_string()) # .to_string() für bessere Konsolenausgabe
        else:
            print(f"WARNUNG: Keine Experimentdaten im Verzeichnis '{test_results_dir}' gefunden oder beim Parsen ist ein Fehler aufgetreten.")
            print("         Überprüfe, ob *_full_run_log.json Dateien vorhanden und korrekt formatiert sind.")