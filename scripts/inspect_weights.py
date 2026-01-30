import torch
import numpy as np
import sys
import os
import argparse

def inspect_pth(model_path, output_txt=None):
    if not os.path.exists(model_path):
        print(f"Fehler: Datei nicht gefunden: {model_path}")
        return

    print(f"Lade Modell: {model_path}")
    
    # Lade das State-Dict (auf CPU, um GPU-Fehler zu vermeiden)
    try:
        state_dict = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Fehler beim Laden: {e}")
        return

    # Falls output_txt nicht angegeben, erstelle einen Namen basierend auf dem Input
    if output_txt is None:
        output_txt = model_path + "_WEIGHTS.txt"

    print(f"Schreibe Gewichte nach: {output_txt}")

    # Numpy Konfiguration: Alles anzeigen, nichts kürzen
    np.set_printoptions(threshold=sys.maxsize, linewidth=200, precision=4, suppress=True)

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(f"ANALYSE DES MODELLS: {os.path.basename(model_path)}\n")
        f.write("="*100 + "\n\n")

        # Iteriere über alle Keys im State Dict
        for key, value in state_dict.items():
            f.write(f"PARAMETER: {key}\n")
            
            # Prüfen ob es ein Tensor ist (manchmal sind auch Metadaten gespeichert)
            if torch.is_tensor(value):
                # In Numpy wandeln
                np_data = value.detach().cpu().numpy()
                
                # Statistiken berechnen
                shape = np_data.shape
                size = np_data.size
                min_val = np_data.min()
                max_val = np_data.max()
                mean_val = np_data.mean()
                
                # Sparsity (Wie viele exakte Nullen?)
                zeros = np.sum(np_data == 0)
                sparsity = (zeros / size) * 100 if size > 0 else 0
                
                f.write(f"Typ: Tensor ({value.dtype})\n")
                f.write(f"Shape: {shape}\n")
                f.write(f"Wertebereich: [{min_val:.6f}, {max_val:.6f}] (Mean: {mean_val:.6f})\n")
                f.write(f"Sparsity (Nullen): {zeros}/{size} ({sparsity:.2f}%)\n")
                
                # Die eigentlichen Werte schreiben
                f.write("-" * 30 + " WERTE " + "-" * 30 + "\n")
                f.write(np.array2string(np_data, separator=', '))
            else:
                # Falls es kein Tensor ist (z.B. Versionsnummer oder Config)
                f.write(f"Typ: {type(value)}\n")
                f.write(f"Wert: {value}\n")
            
            f.write("\n\n" + "="*100 + "\n\n")

    print("Fertig.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exportiert Gewichte aus einer .pth Datei in eine lesbare .txt Datei.")
    parser.add_argument("model_path", type=str, help="Pfad zur .pth Datei")
    parser.add_argument("--out", type=str, default=None, help="Optionaler Pfad für die Output .txt Datei")
    
    args = parser.parse_args()
    
    inspect_pth(args.model_path, args.out)