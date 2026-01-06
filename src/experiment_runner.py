# experiment_runner.py

import yaml
import os
import copy
import time
import json
import pandas as pd
from datetime import datetime
import multiprocessing as mp

#from src.training.run_qat_brevitas import run_qat_training_pipeline
from src.training.run_qat_brevitas_structured import run_qat_training_pipeline
from src.training.run_xgb_or_lr_pipeline import run_concrete_ml_pipeline as run_xgb_lr_pipeline

def deep_update(source_dict: dict, overrides: dict) -> dict:
    updated_dict = copy.deepcopy(source_dict)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in updated_dict and isinstance(updated_dict[key], dict):
            updated_dict[key] = deep_update(updated_dict[key], value)
        else:
            updated_dict[key] = value
    return updated_dict

def get_nested_val(data_dict: dict, path: list, default: any = None) -> any:
    current = data_dict
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def run_single_experiment_wrapper(args_tuple):
    config_set_name, i_rep, temp_config_path, num_total_reps, unique_run_output_dir = args_tuple # unique_run_output_dir hinzugefügt
    print(f"Starte Pipeline-Lauf für: {config_set_name}, Wiederholung {i_rep}/{num_total_reps} mit Config: {temp_config_path}")
    print(f"    Ergebnisse für diesen Lauf werden in '{unique_run_output_dir}' erwartet.")
    try:
        # run_qat_training_pipeline gibt jetzt den Pfad zurück, den es verwendet hat
        # (der sollte unique_run_output_dir sein)
        pipeline_result_dir = run_qat_training_pipeline(config_path=temp_config_path) 
        
        # Überprüfen, ob der zurückgegebene Pfad dem erwarteten Pfad entspricht
        if not pipeline_result_dir or not os.path.samefile(pipeline_result_dir, unique_run_output_dir):
             print(f"WARNUNG: Pipeline-Ergebnisordner '{pipeline_result_dir}' stimmt nicht mit dem erwarteten eindeutigen Ordner '{unique_run_output_dir}' überein.")
             # Wir verwenden trotzdem den unique_run_output_dir für die Suche nach dem Log,
             # da die Config diesen Pfad hatte.
             pipeline_result_dir = unique_run_output_dir # Korrigiere auf den erwarteten Pfad

        if pipeline_result_dir and os.path.isdir(pipeline_result_dir):
            print(f"Pipeline-Lauf abgeschlossen. Ergebnisordner geprüft: {pipeline_result_dir}")
            found_json_log = None
            for fname in os.listdir(pipeline_result_dir):
                if fname.endswith("_full_run_log.json"):
                    found_json_log = os.path.join(pipeline_result_dir, fname)
                    break
            if found_json_log:
                return {"config_set_name": config_set_name, 
                        "repetition": i_rep, 
                        "status": "success", 
                        "json_log_path": found_json_log,
                        "results_directory": pipeline_result_dir}
            else:
                print(f"WARNUNG: Kein JSON-Log im Ergebnisordner {pipeline_result_dir} für {config_set_name} Rep {i_rep} gefunden.")
                return {"config_set_name": config_set_name, "repetition": i_rep, "status": "error", "message": "No JSON log found", "results_directory": pipeline_result_dir}
        else:
            print(f"WARNUNG: Pipeline-Lauf für {config_set_name}, Rep {i_rep} hat keinen gültigen Ergebnisordner ({pipeline_result_dir}) zurückgegeben oder erstellt.")
            return {"config_set_name": config_set_name, "repetition": i_rep, "status": "error", "message": "Result directory not valid"}
    except Exception as e_run:
        print(f"FEHLER während der Ausführung von '{config_set_name}', Wiederholung {i_rep}: {e_run}")
        import traceback
        error_log_path = os.path.splitext(temp_config_path)[0] + "_error.txt"
        with open(error_log_path, "w") as f_err:
            f_err.write(f"Fehler bei Konfiguration: {config_set_name}, Wiederholung: {i_rep}\n")
            f_err.write(f"Verwendete Config: {temp_config_path}\n")
            f_err.write(f"Ziel-Ergebnisordner war: {unique_run_output_dir}\n") # Hinzugefügt
            f_err.write(str(e_run) + "\n")
            f_err.write(traceback.format_exc())
        print(f"Fehlerdetails gespeichert in: {error_log_path}")
        return {"config_set_name": config_set_name, "repetition": i_rep, "status": "error", "message": str(e_run)}

# ---- Konfiguration für den Experiment-Runner ----
BASE_CONFIG_PATH = "config.yaml"  # Diese wird als Basis verwendet
NUM_REPETITIONS_PER_CONFIG = 5   # Anzahl der Wiederholungen pro Parameterkombination
MAX_PARALLEL_PROCESSES = 6      # Anzahl der parallelen Prozesse

'''
# --- ISO-PARAMETER GRID: STRUCTURED PRUNING (CICIoT & EdgeIIoT, 2 & 4 Bits) ---

# Pfade zu den Datensätzen
CIC_PATH = "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
EDGE_PATH = "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz" # Pfad angepasst für EdgeIIoT

N_START = 100 

# --- KONFIGURATIONEN ---

# 1. Die neuen Bit-Raten
quant_bits_list = [2, 4]

# 2. Mapping für CICIoT (Wie von dir definiert)
k_map_ciciot = [
    (12, 4),   # k=12 (p4)
    (21, 8),   # k=21 (p8)
    (36, 16),  # k=36 (p16)
    (58, 32)   # k=58 (p32)
]

# 3. Mapping für EdgeIIoT (Wie von dir definiert)
k_map_edge = [
    (10, 4),   # k=10 (p4)
    (18, 8),   # k=18 (p8)
    (30, 16),  # k=30 (p16)
    (51, 32)   # k=51 (p32)
]

PARAMETER_GRID = []

# --- SCHLEIFE 1: CICIoT ---
for k_target, p_label in k_map_ciciot:
    for qbits in quant_bits_list:
        
        # Ratio berechnen (1 - k/100)
        ratio = 1.0 - (k_target / N_START)
        
        PARAMETER_GRID.append({
            "name_suffix": f"qb{qbits}_p{p_label}_structured_C", # C für CICIoT
            
            "run_settings": {
                "dataset_name": "CICIoT"
            },
            "data_params": {
                "npz_file_path": CIC_PATH
            },
            "model_params": {
                "quantization_bits": qbits,
                "n_hidden": N_START,
                "unpruned_neurons": 0,    # Structured -> kein PBT
                "pbt_layers": []
            },
            "training_params": {
                "num_epochs": 150,
                "neuron_pruning": {
                    "enable": True,
                    "pruning_ratio_fc1": ratio,
                    "pruning_ratio_fc2": ratio,
                    "run_at_epoch": 25,
                    "pruning_steps": 1,
                    "pruning_interval": 1
                }
            },
            "evaluation_params": {
                "pytorch_model_eval": {
                    "run_eval": True,
                    "n_samples": None
                },
                "fhe_model_eval": {
                    "run_fhe_pipeline": True,
                    "run_fhe_eval": True,
                    "mode": "simulate",
                    "n_samples_simulate": 1000,
                }
            }
        })

# --- SCHLEIFE 2: EdgeIIoT ---
for k_target, p_label in k_map_edge:
    for qbits in quant_bits_list:
        
        # Ratio berechnen
        ratio = 1.0 - (k_target / N_START)
        
        PARAMETER_GRID.append({
            "name_suffix": f"qb{qbits}_p{p_label}_structured_E", # E für EdgeIIoT
            
            "run_settings": {
                "dataset_name": "EdgeIIoT" # WICHTIG: Name angepasst
            },
            "data_params": {
                "npz_file_path": EDGE_PATH # WICHTIG: Pfad angepasst
            },
            "model_params": {
                "quantization_bits": qbits,
                "n_hidden": N_START,
                "unpruned_neurons": 0,
                "pbt_layers": []
            },
            "training_params": {
                "num_epochs": 150,
                "neuron_pruning": {
                    "enable": True,
                    "pruning_ratio_fc1": ratio,
                    "pruning_ratio_fc2": ratio,
                    "run_at_epoch": 25,
                    "pruning_steps": 1,
                    "pruning_interval": 1
                }
            },
            "evaluation_params": {
                "pytorch_model_eval": {
                    "run_eval": True,
                    "n_samples": None
                },
                "fhe_model_eval": {
                    "run_fhe_pipeline": True,
                    "run_fhe_eval": True,
                    "mode": "simulate",
                    "n_samples_simulate": 1000,
                }
            }
        })

print(f"GRID FERTIG: {len(PARAMETER_GRID)} Konfigurationen erstellt.")
print(f"Varianten: 2 Datensätze x 4 Pruning-Level x 2 Bit-Settings (2, 4)")



# --- ISO-PARAMETER GRID: STRUCTURED PRUNING (CICIoT, k adjusted) ---

CIC_PATH = "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
N_START = 100 

# Mapping: Angepasste k-Werte für CICIoT
# Da CICIoT weniger Input-Features/Klassen hat als EdgeIIoT, 
# müssen wir k erhöhen, um auf die gleiche Parameter-Summe zu kommen.
k_map = [
    (12, 4),   # k=12 (statt 10) entspricht Params von p4
    (21, 8),   # k=21 (statt 18) entspricht Params von p8
    (36, 16),  # k=34 (statt 30) entspricht Params von p16
    (58, 32)   # k=57 (statt 51) entspricht Params von p32
]

quant_bits_list = [3, 5, 7]

PARAMETER_GRID = []

for k_target, p_label in k_map:
    for qbits in quant_bits_list:
        
        # Berechnung der Ratio für Strukturiertes Pruning
        # Ziel: k Neuronen von 100 übrig behalten
        ratio = 1.0 - (k_target / N_START)
        
        PARAMETER_GRID.append({
            # Name: qb3_p4_structured_C (C für CICIoT)
            "name_suffix": f"qb{qbits}_p{p_label}_structured_C",
            
            "run_settings": {
                "dataset_name": "CICIoT" # WICHTIG: Name angepasst
            },
            "data_params": {
                "npz_file_path": CIC_PATH
            },
            "model_params": {
                "quantization_bits": qbits,
                "n_hidden": N_START,     # Start mit 100
                
                # Unstructured Pruning DEAKTIVIEREN
                "unpruned_neurons": 0,   
                "pbt_layers": []
            },
            "training_params": {
                "num_epochs": 150,
                "neuron_pruning": {
                    "enable": True,             # Structured AN
                    "pruning_ratio_fc1": ratio, 
                    "pruning_ratio_fc2": ratio,
                    "run_at_epoch": 25,         # Pruning nach 25 Epochen
                    "pruning_steps": 1,        
                    "pruning_interval": 1
                }
            },
            # Evaluierung AKTIVIEREN
            "evaluation_params": {
                "pytorch_model_eval": {
                    "run_eval": True,
                    "n_samples": None
                },
                "fhe_model_eval": {
                    "run_fhe_pipeline": True,
                    "run_fhe_eval": True,
                    "mode": "simulate",
                    "n_samples_simulate": 1000
                }
            }
        })

print(f"Structured Grid (CICIoT) erstellt: {len(PARAMETER_GRID)} Configs (k={k_map}).")

'''
# --- ISO-PARAMETER GRID: STRUCTURED PRUNING (EdgeIIoT, k=11,20,35,57) ---

EDGE_PATH = "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
N_START = 100 

# Mapping: Welches k entspricht welchem "p-Level" im Namen?
# (Damit die Sortierung p4 -> p8 -> p16 -> p32 erhalten bleibt)
k_map = [
    (10, 4),   # k=11 entspricht Params von p4
    (18, 8),   # k=18 entspricht Params von p8  (NICHT 20)
    (30, 16),  # k=31 entspricht Params von p16 (NICHT 35)
    (51, 32)   # k=51 entspricht Params von p32 (NICHT 57)
]

quant_bits_list = [3, 5, 7]

PARAMETER_GRID = []

for k_target, p_label in k_map:
    for qbits in quant_bits_list:
        
        # Berechnung der Ratio für Strukturiertes Pruning
        # Ziel: k Neuronen von 100 übrig behalten
        ratio = 1.0 - (k_target / N_START)
        
        PARAMETER_GRID.append({
            # Name: qb3_p4_structured_E
            "name_suffix": f"qb{qbits}_p{p_label}_structured_E",
            
            "run_settings": {
                "dataset_name": "EgeIIoT"
            },
            "data_params": {
                "npz_file_path": EDGE_PATH
            },
            "model_params": {
                "quantization_bits": qbits,
                "n_hidden": N_START,     # Start mit 100
                
                # Unstructured Pruning DEAKTIVIEREN
                "unpruned_neurons": 0,   
                "pbt_layers": []
            },
            "training_params": {
                "num_epochs": 150,
                "neuron_pruning": {
                    "enable": True,             # Structured AN
                    "pruning_ratio_fc1": ratio, 
                    "pruning_ratio_fc2": ratio,
                    "run_at_epoch": 25,         # Pruning nach 25 Epochen (wie Strategie B)
                    "pruning_steps": 1,        # Sanftes Pruning
                    "pruning_interval": 1
                }
            },
            # Evaluierung AKTIVIEREN (da "normales Training und Eval" gewünscht)
            "evaluation_params": {
                "pytorch_model_eval": {
                    "run_eval": True,
                    "n_samples": None
                },
                "fhe_model_eval": {
                    "run_fhe_pipeline": True,
                    "run_fhe_eval": True,
                    "mode": "simulate",
                    "n_samples_simulate": 1000
                }
            }
        })

print(f"Structured Grid erstellt: {len(PARAMETER_GRID)} Configs (k={k_map}).")

'''
# --- ISO-PARAMETER GEWICHTS-CHECK (Schnell) ---

CIC_PATH = "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
N_START = 100 

# Definition der Paare für CICIoT (39 In, 34 Out)
# Format: (Unstructured Inputs u, Äquivalente Strukturierte Neuronen k)
# Berechnungsgrundlage: k^2 + k*(39+34) - Params_Unstr = 0
iso_pairs_cic = [
    (2, 7),    # Level 2: ~468 Params  <-> k=6  (~474 Params) (0.43, 0.65, 0.80, 0.89, 0.94)
    (4, 12),   # Level 4: ~936 Params  <-> k=11 (~924 Params)
    (8, 21),   # Level 8: ~1872 Params <-> k=20 (~1860 Params)
    (16, 36),  # Level 16: ~3744 Params <-> k=35 (~3780 Params)
    (32, 58)   # Level 32: ~7488 Params <-> k=57 (~7410 Params)
]

PARAMETER_GRID = []
qbits = 4 # Fixe Bitbreite

for u, k_target in iso_pairs_cic:
    
    # Ratio berechnen: Wie viel von 100 muss weg, um auf k zu kommen?
    ratio = 1.0 - (k_target / N_START)

    # --- 1. UNSTRUCTURED CONFIG (PBT) ---
    PARAMETER_GRID.append({
        "name_suffix": f"IsoCheck_Unstr_p{u}_C",
        "run_settings": {"dataset_name": "CICIoT"},
        "data_params": {"npz_file_path": CIC_PATH},
        "model_params": {
            "quantization_bits": qbits,
            "n_hidden": N_START,             # Bleibt 100
            "unpruned_neurons": u,           # PBT Limit
            "pbt_layers": ["fc1", "fc2", "fc3"] # PBT auf alle Layer
        },
        "training_params": {
            "num_epochs": 1,                 # Kurztraining
            "neuron_pruning": {"enable": False}
        },
        # Evaluation deaktivieren
        "evaluation_params": {
            "pytorch_model_eval": {"run_eval": False},
            "fhe_model_eval": {"run_fhe_pipeline": False}
        }
    })

    # --- 2. STRUCTURED CONFIG (Neuron Pruning) ---
    PARAMETER_GRID.append({
        "name_suffix": f"IsoCheck_Struct_k{k_target}_C",
        "run_settings": {"dataset_name": "CICIoT"},
        "data_params": {"npz_file_path": CIC_PATH},
        "model_params": {
            "quantization_bits": qbits,
            "n_hidden": N_START,             # Startet auch mit 100
            "unpruned_neurons": 0,           # Kein PBT
            "pbt_layers": []
        },
        "training_params": {
            "num_epochs": 2,
            "neuron_pruning": {
                "enable": True,              # Pruning aktivieren
                "pruning_ratio_fc1": ratio,  # Berechnete Ratio
                "pruning_ratio_fc2": ratio,
                "run_at_epoch": 1,           # Sofort prunen (nach Init/Epoch 0)
                "pruning_steps": 1
            }
        },
        # Evaluation deaktivieren
        "evaluation_params": {
            "pytorch_model_eval": {"run_eval": False},
            "fhe_model_eval": {"run_fhe_pipeline": False}
        }
    })

print(f"CICIoT Iso-Grid erstellt: {len(PARAMETER_GRID)} Konfigurationen.")


EDGE_PATH = "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
N_START = 100 

iso_pairs = [
    (2, 6),    # Level 2: ~468 Params  <-> k=6  (~474 Params) (0.43, 0.65, 0.80, 0.89, 0.94)
    (4, 11),   # Level 4: ~936 Params  <-> k=11 (~924 Params)
    (8, 20),   # Level 8: ~1872 Params <-> k=20 (~1860 Params)
    (16, 35),  # Level 16: ~3744 Params <-> k=35 (~3780 Params)
    (32, 57)   # Level 32: ~7488 Params <-> k=57 (~7410 Params)
]

PARAMETER_GRID = []
qbits = 4

for u, k_target in iso_pairs:
    
    # Ratio berechnen
    ratio = 1.0 - (k_target / N_START)

    # --- UNSTRUCTURED CONFIG ---
    PARAMETER_GRID.append({
        "name_suffix": f"IsoCheck_Unstr_p{u}_E",
        "run_settings": {"dataset_name": "EgeIIoT"},
        "data_params": {"npz_file_path": EDGE_PATH},
        "model_params": {
            "quantization_bits": qbits,
            "n_hidden": N_START,
            "unpruned_neurons": u,
            "pbt_layers": ["fc1", "fc2", "fc3"]
        },
        "training_params": {
            "num_epochs": 1,                     # <--- Nur 1 Epoche reicht für Setup
            "neuron_pruning": {"enable": False}
        },
        # --- HIER DEAKTIVIEREN WIR ALLES UNNÖTIGE ---
        "evaluation_params": {
            "pytorch_model_eval": {"run_eval": False},    # Keine PyTorch Eval
            "fhe_model_eval": {"run_fhe_pipeline": False} # Keine FHE Eval
        }
    })

    # --- STRUCTURED CONFIG ---
    PARAMETER_GRID.append({
        "name_suffix": f"IsoCheck_Struct_k{k_target}_E",
        "run_settings": {"dataset_name": "EgeIIoT"},
        "data_params": {"npz_file_path": EDGE_PATH},
        "model_params": {
            "quantization_bits": qbits,
            "n_hidden": N_START,
            "unpruned_neurons": 0,
            "pbt_layers": []
        },
        "training_params": {
            "num_epochs": 2,                     # <--- Nur 1 Epoche
            "neuron_pruning": {
                "enable": True,
                "pruning_ratio_fc1": ratio,
                "pruning_ratio_fc2": ratio,
                "run_at_epoch": 1,               # Sofort prunen
                "pruning_steps": 1
            }
        },
        # --- HIER EBENFALLS DEAKTIVIEREN ---
        "evaluation_params": {
            "pytorch_model_eval": {"run_eval": False},
            "fhe_model_eval": {"run_fhe_pipeline": False}
        }
    })



# Definieren der zu testenden Quantisierungsbits und unpruned Neuronen
# 1. Konfigurationen definieren
quantization_levels = [2, 4, 8]
unpruned_neurons_levels = [4, 8, 16, 32]
datasets = [
    {"name": "CICIoT", "suffix": "C", "path": "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"},
    {"name": "EgeIIoT", "suffix": "E", "path": "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"}
]

PARAMETER_GRID = []

# 2. Grid generieren (Unstructured Pruning Focus)
for ds in datasets:
    for q_bits in quantization_levels:
        for unpruned in unpruned_neurons_levels:
            
            # Name Suffix generieren: z.B. "qb4_p16_unstructured_C"
            config_name = f"qb{q_bits}_p{unpruned}_unstructured_{ds['suffix']}"
            
            # Config Dictionary erstellen
            experiment_config = {
                "name_suffix": config_name,
                "run_settings": {
                    "dataset_name": ds["name"]
                },
                "data_params": {
                    "npz_file_path": ds["path"]
                },
                "model_params": {
                    "quantization_bits": q_bits,
                    "unpruned_neurons": unpruned,
                    # WICHTIG: PBT aktivieren, Structured Pruning deaktivieren
                    "pbt_layers":  ["fc1", "fc2", "fc3"],  # PBT auf Layer 1 anwenden
                },
                "training_params": {
                    "neuron_pruning": {
                        "enable": False     # Structured Pruning AUS
                    }
                }
            }
            
            PARAMETER_GRID.append(experiment_config)

# Optional: Kontrollausgabe der generierten Configs
print(f"Generiertes Grid: {len(PARAMETER_GRID)} Konfigurationen.")





PARAMETER_GRID = [
    {
        "name_suffix": "qb5_p4_unstructured_E",
        "run_settings": {
            "dataset_name": "EgeIIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 5, "unpruned_neurons": 4}

    },
    
    
]


    
    {
        "name_suffix": "qb3_p4_unstructured_C",
        "run_settings": {
            "dataset_name": "CICIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 3, "unpruned_neurons": 4}

    },
    {
        "name_suffix": "qb5_p4_unstructured_C",
        "run_settings": {
            "dataset_name": "CICIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 5, "unpruned_neurons": 4}
    
    },
    {
        "name_suffix": "qb7_p4_unstructured_C",
        "run_settings": {
            "dataset_name": "CICIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 7, "unpruned_neurons": 4}

    },
    {
        "name_suffix": "qb3_p4_unstructured_E",
        "run_settings": {
            "dataset_name": "EgeIIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 3, "unpruned_neurons": 4}

    },
    {
        "name_suffix": "qb5_p4_unstructured_E",
        "run_settings": {
            "dataset_name": "EgeIIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 5, "unpruned_neurons": 4}

    },
    {
        "name_suffix": "qb7_p4_unstructured_E",
        "run_settings": {
            "dataset_name": "EgeIIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 7, "unpruned_neurons": 4}

    },



 
    
    {
        "name_suffix": "qb3_p32_structured_E",
        "run_settings": {
            "dataset_name": "EgeIIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 3, "unpruned_neurons": 32}

    },
    {
        "name_suffix": "qb5_p32_structured_E",
        "run_settings": {
            "dataset_name": "EgeIIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 5, "unpruned_neurons": 32}

    },
    
    


    

{
        "name_suffix": "qb3_p32_pruned_A", # Wichtig: Eindeutiger Name!
        "run_settings": {
            "dataset_name": "EdgeIIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 3, "unpruned_neurons": 32},
        
        # HIER die neuen Parameter einfügen
        "training_params": {
            "neuron_pruning": {
                "enable": true,
                "run_at_epoch": 5,
                "pruning_ratio_fc1": 0.55, # Rate A
                "pruning_ratio_fc2": 0.68  # Rate A
            }
        }
    }, 
    {
        "name_suffix": "qb3_p32", 
        "run_settings": {
            "dataset_name": "CICIoT"
        },
        "data_params": {
            "npz_file_path": "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
        },
        "model_params": {"quantization_bits": 3, "unpruned_neurons": 32}
    },
    {
        "name_suffix": "hidden128_qb3_p8_focal_alpha03_gamma15_ep1",
        "model_params": {"n_hidden": 128, "quantization_bits": 3, "unpruned_neurons": 8},
        "training_params": {
            "num_epochs": 1,
            "criterion": {"name": "FocalLoss", "focal_loss_alpha": 0.3, "focal_loss_gamma": 1.5}
        }
    },
    {
        "name_suffix": "hidden128_qb3_p32_celoss_ep1",
        "model_params": {"n_hidden": 128, "quantization_bits": 3, "unpruned_neurons": 32},
        "training_params": {
            "num_epochs": 1,
            "criterion": {"name": "CrossEntropyLoss"}
        }
    },
    {
        "name_suffix": "hidden256_qb5_p4_schedulerRLP_ep1",
        "model_params": {"n_hidden": 256, "quantization_bits": 5, "unpruned_neurons": 4},
        "training_params": {
            "num_epochs": 1,
            "scheduler": {"name": "ReduceLROnPlateau"}
        }
    },
    {
        "name_suffix": "hidden256_qb7_p16_lr001_schedulerRLP",
        "model_params": {"n_hidden": 256, "quantization_bits": 7, "unpruned_neurons": 16},
        "training_params": {
            "learning_rate": 0.001,
            "scheduler": {"name": "ReduceLROnPlateau"}
        },

        
    {"name_suffix": "default_config"} # Läuft mit der Basis-Konfiguration
    
    }
'''

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing Startmethode auf 'spawn' gesetzt.")
    except RuntimeError:
        print("Multiprocessing Startmethode konnte nicht auf 'spawn' gesetzt werden oder war bereits gesetzt.")

    try:
        with open(BASE_CONFIG_PATH, 'r') as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FEHLER: Basis-Konfigurationsdatei '{BASE_CONFIG_PATH}' nicht gefunden.")
        exit(1) # Beenden, da base_config benötigt wird
    except Exception as e:
        print(f"FEHLER beim Laden der Basis-Konfigurationsdatei '{BASE_CONFIG_PATH}': {e}")
        exit(1) # Beenden

    meta_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Stelle sicher, dass results_base_dir aus der base_config gelesen wird für den Haupt-Meta-Run Ordner
    # oder setze einen Standardwert, falls nicht in base_config vorhanden.
    meta_run_parent_dir = base_config.get("run_settings", {}).get("results_base_dir", "results/qat_runs")
    
    main_output_dir_for_meta_run = os.path.join(
        meta_run_parent_dir, # Dieser Pfad kommt aus der base_config
        f"meta_run_{meta_run_timestamp}_CICIoT_GridSearch" # Aussagekräftiger Name für den Meta-Run
    )
    os.makedirs(main_output_dir_for_meta_run, exist_ok=True)
    print(f"Alle Ergebnisse dieses Meta-Laufs werden in Unterordnern von '{main_output_dir_for_meta_run}' gespeichert.")

    temp_configs_storage_dir = os.path.join(main_output_dir_for_meta_run, "run_configs")
    os.makedirs(temp_configs_storage_dir, exist_ok=True)
    print(f"Temporäre Konfigurationsdateien werden in '{temp_configs_storage_dir}' gespeichert.")

    tasks_to_run = []
    for i_config_set, param_variation_overrides in enumerate(PARAMETER_GRID):
        config_set_name = param_variation_overrides.get('name_suffix', f"configset_{i_config_set+1}")
        
        for i_rep in range(1, NUM_REPETITIONS_PER_CONFIG + 1):
            current_run_config = copy.deepcopy(base_config)
            
            # Überschreibe 'dataset_name' und 'npz_file_path' in der base_config
            # durch die Werte aus dem aktuellen PARAMETER_GRID Eintrag.
            # deep_update kümmert sich um die korrekte Verschachtelung.
            current_overrides = {
                "run_settings": param_variation_overrides.get("run_settings", {}),
                "data_params": param_variation_overrides.get("data_params", {}),
                "model_params": param_variation_overrides.get("model_params", {}),
                "training_params": param_variation_overrides.get("training_params", {}),
                "evaluation_params": param_variation_overrides.get("evaluation_params", {})
            }
            current_run_config = deep_update(current_run_config, current_overrides)
            
            current_run_config.setdefault("meta_run_info", {})
            current_run_config["meta_run_info"]["config_set_name"] = config_set_name
            current_run_config["meta_run_info"]["repetition_number"] = i_rep
            
            unique_run_output_dir = os.path.join(main_output_dir_for_meta_run, config_set_name, f"rep_{i_rep}")
            os.makedirs(unique_run_output_dir, exist_ok=True)
            
            current_run_config.setdefault("run_settings", {})["results_base_dir"] = unique_run_output_dir

            temp_config_filename = f"config_{config_set_name}_rep{i_rep}.yaml"
            temp_config_path = os.path.join(temp_configs_storage_dir, temp_config_filename)

            with open(temp_config_path, 'w') as f_temp_cfg:
                yaml.dump(current_run_config, f_temp_cfg, sort_keys=False, indent=2)
            
            tasks_to_run.append((config_set_name, i_rep, temp_config_path, NUM_REPETITIONS_PER_CONFIG, unique_run_output_dir))

    print(f"Insgesamt {len(tasks_to_run)} Trainingstasks vorbereitet.")
    all_runs_summary_data = []
    
    if tasks_to_run:
        print(f"Starte parallele Ausführung mit maximal {MAX_PARALLEL_PROCESSES} Prozessen...")
        with mp.Pool(processes=MAX_PARALLEL_PROCESSES) as pool:
            results = pool.map(run_single_experiment_wrapper, tasks_to_run)
        
        for result in results:
            if result and result.get("status") == "success":
                json_log_path = result["json_log_path"]
                pipeline_result_dir = result["results_directory"]
                config_set_name_from_result = result["config_set_name"] # Korrigiert von config_set_name
                i_rep_from_result = result["repetition"] # Korrigiert von i_rep
                
                try:
                    with open(json_log_path, 'r') as f_res:
                        run_log_data = json.load(f_res)
                    run_config_used = run_log_data.get("input_configuration_from_yaml", {})
                    
                    # Holen des tatsächlichen Dataset-Namens aus dem Log des Laufs
                    actual_dataset_name = get_nested_val(run_config_used, ["run_settings", "dataset_name"], "N/A_Dataset")

                    run_summary = {
                        "config_set_name": config_set_name_from_result, 
                        "repetition": i_rep_from_result,
                        "dataset_name_from_log": actual_dataset_name, # Wichtig für die spätere Analyse
                        "timestamp": get_nested_val(run_log_data,["run_overview", "run_timestamp"]),
                        "results_directory": pipeline_result_dir, 
                        "json_log_path": json_log_path,
                        "n_hidden": get_nested_val(run_config_used, ["model_params", "n_hidden"]),
                        "quant_bits": get_nested_val(run_config_used, ["model_params", "quantization_bits"]),
                        "unpruned_neurons": get_nested_val(run_config_used, ["model_params", "unpruned_neurons"]),
                        "learning_rate": get_nested_val(run_config_used, ["training_params", "learning_rate"]),
                        "criterion": get_nested_val(run_config_used, ["training_params", "criterion", "name"]),
                        "best_val_f1_weighted": get_nested_val(run_log_data,["best_model_metrics_achieved_val", "best_f1_weighted_val"]),
                        "pytorch_test_f1_weighted": get_nested_val(run_log_data,["evaluation_execution_details", "pytorch_eval_summary", "f1_weighted"]),
                    }
                    fhe_sim_summary = get_nested_val(run_log_data,["evaluation_execution_details", "fhe_evaluations", "simulate"], {})
                    if fhe_sim_summary and fhe_sim_summary.get("status") != "skipped" and "f1_weighted" in fhe_sim_summary:
                        run_summary["fhe_simulate_f1_weighted"] = fhe_sim_summary.get("f1_weighted")
                    
                    fhe_exec_summary = get_nested_val(run_log_data,["evaluation_execution_details", "fhe_evaluations", "execute"], {})
                    if fhe_exec_summary and fhe_exec_summary.get("status") != "skipped" and "f1_weighted" in fhe_exec_summary:
                        run_summary["fhe_execute_f1_weighted"] = fhe_exec_summary.get("f1_weighted")
                    
                    all_runs_summary_data.append(run_summary)
                    print(f"Ergebnisse für {config_set_name_from_result} Rep {i_rep_from_result} erfolgreich verarbeitet.")
                except Exception as e_collect:
                    print(f"FEHLER beim Sammeln der Ergebnisse für {config_set_name_from_result} Rep {i_rep_from_result} (JSON: {json_log_path}): {e_collect}")
            elif result:
                print(f"Fehlerhafter Lauf: Config '{result.get('config_set_name')}', Rep {result.get('repetition')}. Nachricht: {result.get('message')}")

    if all_runs_summary_data:
        summary_df = pd.DataFrame(all_runs_summary_data)
        summary_csv_path = os.path.join(main_output_dir_for_meta_run, f"meta_experiment_summary_{meta_run_timestamp}.csv")
        try:
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\n{'='*20} GESAMTZUSAMMENFASSUNG {'='*20}")
            print(f"Gesamtzusammenfassung aller Läufe gespeichert in: {summary_csv_path}")
            print("Ausgabe der ersten Zeilen der Zusammenfassung:")
            print(summary_df.head().to_string())
            if not summary_df.empty and 'best_val_f1_weighted' in summary_df.columns:
                print("\nDurchschnittliche Metriken pro Konfigurations-Set (Beispiel: best_val_f1_weighted):")
                summary_df['best_val_f1_weighted'] = pd.to_numeric(summary_df['best_val_f1_weighted'], errors='coerce')
                
                # Gruppiere nach config_set_name und dem tatsächlichen Datensatznamen aus dem Log
                grouping_cols_summary = ['config_set_name', 'dataset_name_from_log', 'quant_bits', 'unpruned_neurons']
                # Filtere Spalten, die tatsächlich im DataFrame existieren
                grouping_cols_summary = [col for col in grouping_cols_summary if col in summary_df.columns]

                if grouping_cols_summary:
                    agg_results = summary_df.groupby(grouping_cols_summary)['best_val_f1_weighted'].agg(['mean', 'std', 'count', 'min', 'max'])
                    print(agg_results.to_string())
                else:
                    print("Konnte nicht nach config_set_name und dataset_name_from_log gruppieren, da Spalten fehlen.")

        except Exception as e_csv_summary:
            print(f"FEHLER beim Speichern oder Anzeigen der Gesamtzusammenfassung: {e_csv_summary}")
    print(f"\n{'='*20} ALLE EXPERIMENTE ABGESCHLOSSEN {'='*20}")