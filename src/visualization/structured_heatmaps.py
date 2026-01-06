#!/usr/bin/env python
# coding: utf-8

"""
Visualisierungs-Skript zur Analyse des strukturierten Prunings.

Dieses Skript nimmt den Pfad zu einem QAT-Ergebnisordner,
lädt automatisch das Modell (.pth) und die Konfiguration (.json),
und generiert eine Heatmap, die anzeigt, welche Neuronen (Kanäle/Zeilen)
in den Layern fc1 und fc2 durch strukturiertes Pruning entfernt wurden.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import argparse  # Neu: Für Kommandozeilen-Argumente
import traceback

# WICHTIG: Wir importieren die Modelldefinition direkt aus dem 'models'-Modul.
# Das stellt sicher, dass wir immer die korrekte Architektur verwenden.
try:
    from src.models.qat_model import QATPrunedSimpleNet
    # Stellen Sie sicher, dass Brevitas-Importe in qat_model.py korrekt sind
except ImportError:
    print("FEHLER: Konnte QATPrunedSimpleNet aus 'src.models.qat_model' nicht importieren.")
    print("Stellen Sie sicher, dass Sie dieses Skript mit 'python -m src.visualization.structured_heatmaps ...' ausführen.")
    exit(1)


def find_run_files(run_dir: str) -> (str, str):
    """
    Sucht im angegebenen Ordner nach der Modelldatei (.pth) und der Log-Datei (.json).
    
    Args:
        run_dir (str): Der Pfad zum Ergebnisordner (z.B. .../rep_1/).

    Returns:
        (str, str): Ein Tupel (model_pfad, log_pfad).
    
    Raises:
        FileNotFoundError: Wenn eine der Dateien nicht gefunden wird.
    """
    model_path = None
    log_path = None
    
    # Suchen Sie nach den relevanten Dateien
    for fname in os.listdir(run_dir):
        if fname.endswith(".pth"):
            model_path = os.path.join(run_dir, fname)
        if fname.endswith("_full_run_log.json"):
            log_path = os.path.join(run_dir, fname)
            
    # Fehlerprüfung
    if not model_path:
        raise FileNotFoundError(f"Keine Modelldatei (.pth) in {run_dir} gefunden.")
    if not log_path:
        raise FileNotFoundError(f"Keine JSON-Log-Datei ('_full_run_log.json') in {run_dir} gefunden.")
        
    print(f"Modell gefunden: {os.path.basename(model_path)}")
    print(f"Log gefunden: {os.path.basename(log_path)}")
    return model_path, log_path


def extract_params_from_log(log_path: str) -> Dict[str, Any]:
    """
    Extrahiert die notwendigen Modell-Hyperparameter aus der JSON-Log-Datei.
    
    Args:
        log_path (str): Pfad zur _full_run_log.json Datei.

    Returns:
        Dict[str, Any]: Ein Dictionary mit den Modellparametern.
    """
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    params = {}
    
    # Extrahiere die Laufzeit-Parameter (die tatsächlich verwendet wurden)
    try:
        params['input_size'] = log_data["data_summary_runtime"]["input_features_detected"]
        params['num_classes'] = log_data["data_summary_runtime"]["output_classes_detected"]
        params['n_hidden'] = log_data["model_details_runtime"]["n_hidden_applied"]
        params['quantization_bits'] = log_data["model_details_runtime"]["quantization_bits_applied"]
        params['dropout_rate1'] = log_data["model_details_runtime"]["dropout_rate1_applied"]
        params['dropout_rate2'] = log_data["model_details_runtime"]["dropout_rate2_applied"]
        
        # Rekonstruiere die Brevitas-Config-Dicts, die der Konstruktor erwartet
        # Annahme: Diese Parameter sind standardisiert (basierend auf qat_model.py)
        qbits = params['quantization_bits']
        params['qlinear_args_config'] = {"weight_bit_width": qbits, "bias": True}
        params['qidentity_args_config'] = {"bit_width": qbits, "return_quant_tensor": True}
        params['qrelu_args_config'] = {"bit_width": qbits, "return_quant_tensor": True}

    except KeyError as e:
        print(f"FEHLER: Schlüssel {e} nicht in der JSON-Log-Datei gefunden.")
        print("Die JSON-Log-Datei ist möglicherweise veraltet oder unvollständig.")
        raise
        
    return params


def plot_neuron_sparsity(layer_name: str, weight_matrix: torch.Tensor, ax: plt.Axes):
    """
    Erstellt eine Heatmap, die geprunte Neuronen (Zeilen/Ausgänge) visualisiert.
    
    Args:
        layer_name (str): Name des Layers (z.B. 'FC1').
        weight_matrix (torch.Tensor): Die Gewichtsmatrix des Layers.
        ax (plt.Axes): Die Matplotlib-Achse, auf die geplottet werden soll.
    """
    # Schwellenwert für die L2-Norm, um ein "gepruntes" Neuron zu identifizieren
    PRUNING_THRESHOLD = 1e-6 

    # Berechne die L2-Norm jeder Zeile (jedes Output-Neurons)
    # dim=1 sind die Inputs, dim=0 sind die Outputs (Neuronen)
    l2_norms = torch.linalg.norm(weight_matrix, ord=2, dim=1).cpu().numpy()
    
    # Bestimme, welche Neuronen als "geprunt" gelten (L2-Norm nahe Null)
    is_pruned = (l2_norms <= PRUNING_THRESHOLD).astype(int) # 1 für geprunt, 0 für aktiv
    
    # Erstelle ein 2D-Array für die Heatmap: (Anzahl Neuronen x 1)
    heatmap_data = is_pruned.reshape(-1, 1)

    # Plot
    # cmap 'cividis_r' (umgekehrt) zeigt 0 (aktiv) als Gelb und 1 (geprunt) als Dunkelblau
    cax = ax.imshow(heatmap_data, cmap='cividis_r', aspect='auto', interpolation='nearest')
    ax.set_title(f'Pruning Status für {layer_name}')
    ax.set_ylabel('Neuron Index (Output Channel)')
    ax.set_xticks([]) # Keine X-Ticks, da nur eine Spalte
    
    # Setze Y-Ticks, aber zeige nur alle 10 Ticks, wenn es zu viele Neuronen sind
    num_neurons = len(is_pruned)
    if num_neurons > 50:
        step = num_neurons // 10
        ax.set_yticks(np.arange(0, num_neurons, step))
    else:
        ax.set_yticks(np.arange(num_neurons))
    ax.tick_params(axis='y', labelsize=8)

    # Farbbar hinzufügen
    cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Aktiv (Nicht-Null)', 'Geprunt (Null)'])

    num_pruned = np.sum(is_pruned)
    total_neurons = len(is_pruned)
    print(f"{layer_name}: {num_pruned}/{total_neurons} ({num_pruned/total_neurons:.1%}) Neuronen (Ausgänge) geprunt.")


def main():
    """
    Hauptfunktion zum Ausführen des Skripts.
    """
    # --- 1. Kommandozeilen-Argumente parsen ---
    parser = argparse.ArgumentParser(
        description="Visualisiert das strukturierte Pruning (entfernte Neuronen) "
                    "eines trainierten QAT-Modells aus einem Ergebnisordner.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_directory", 
        type=str, 
        help="Pfad zum spezifischen Ergebnisordner (z.B. .../meta_run_.../rep_1/)"
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.run_directory):
        print(f"FEHLER: Verzeichnis nicht gefunden: {args.run_directory}")
        return

    try:
        # --- 2. Dateien finden ---
        print(f"Analysiere Ordner: {args.run_directory}")
        model_path, log_path = find_run_files(args.run_directory)

        # --- 3. Parameter extrahieren ---
        print("Extrahiere Modellparameter aus JSON-Log...")
        model_params = extract_params_from_log(log_path)
        print(f"Parameter extrahiert: {model_params}")

        # --- 4. Modell instanziieren ---
        print("Instanziiere Modellarchitektur...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Verwende die dynamisch geladenen Parameter
        model = QATPrunedSimpleNet(
            input_size=model_params['input_size'],
            num_classes=model_params['num_classes'],
            n_hidden=model_params['n_hidden'],
            qlinear_args_config=model_params['qlinear_args_config'],
            qidentity_args_config=model_params['qidentity_args_config'],
            qrelu_args_config=model_params['qrelu_args_config'],
            dropout_rate1=model_params['dropout_rate1'],
            dropout_rate2=model_params['dropout_rate2']
        ).to(device)

        # --- 5. Gewichte laden ---
        print(f"Lade Gewichte von {os.path.basename(model_path)}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Wichtig: Modell in den Evaluationsmodus setzen

        # --- 6. Heatmaps plotten ---
        print("Generiere Pruning-Heatmaps...")
        fig, axes = plt.subplots(1, 2, figsize=(10, 7)) # 1 Zeile, 2 Spalten für fc1 und fc2

        if hasattr(model, 'fc1'):
            plot_neuron_sparsity('FC1 Layer', model.fc1.weight.data, axes[0])
        else:
            axes[0].set_title("FC1 nicht gefunden")

        if hasattr(model, 'fc2'):
            plot_neuron_sparsity('FC2 Layer', model.fc2.weight.data, axes[1])
        else:
            axes[1].set_title("FC2 nicht gefunden")
        
        fig.suptitle(f"Visualisierung des strukturierten Prunings\n(Modell: {os.path.basename(model_path)})", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Platz für Haupttitel lassen
        
        # --- 7. Plot speichern ---
        save_path = os.path.join(args.run_directory, "structured_pruning_heatmap.png")
        plt.savefig(save_path)
        print(f"Heatmap erfolgreich gespeichert in: {save_path}")
        # plt.show() # Auskommentieren, wenn das Skript in einer Umgebung ohne GUI läuft

    except FileNotFoundError as e:
        print(f"FEHLER: {e}")
    except KeyError as e:
        print(f"FEHLER: Fehlender Schlüssel {e} in der JSON-Log-Datei. Das Log ist möglicherweise nicht kompatibel.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten:")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()