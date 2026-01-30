import sys
import os
import glob
import re
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat as CommonInt8WeightPerTensorFloat
from concrete.ml.torch.compile import compile_brevitas_qat_model

# --- 1. IMPORT FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_eval_path = os.path.join(current_dir, 'src', 'evaluation')
sys.path.append(src_eval_path if os.path.exists(src_eval_path) else current_dir)

# Wir nutzen die dedizierte Execute-Library
try:
    from concrete_evaluate_execute import evaluate_fhe_model
except ImportError:
    print(f"KRITISCHER FEHLER: 'concrete_evaluate_execute.py' nicht gefunden.")
    print(f"Bitte stelle sicher, dass die Datei in '{src_eval_path}' liegt.")
    sys.exit(1)

# --- 2. KONFIGURATION ---
CIC_PATH = "/home/jovyan/TenSEAL_projects/ciciot_dataset_all.npz"
EDGE_PATH = "/home/jovyan/TenSEAL_projects/edgeiiot_dataset_all.npz"
FHE_MODE = "execute"      
P_ERROR = None  

# --- 3. MODELL DEFINITION ---
class QATPrunedSimpleNet(nn.Module):
    def __init__(self, n_features, n_hidden, n_output, quantization_bits, 
                 unpruned_neurons=0, pbt_layers=None, dropout_rate1=0.0, dropout_rate2=0.0):
        super(QATPrunedSimpleNet, self).__init__()
        self.bit_width = quantization_bits
        self.return_quant = True 
        self.quant_inp = qnn.QuantIdentity(bit_width=self.bit_width, return_quant_tensor=self.return_quant)
        self.fc1 = qnn.QuantLinear(n_features, n_hidden, bias=True, weight_bit_width=self.bit_width, weight_quant=CommonInt8WeightPerTensorFloat)
        self.quant_relu1 = qnn.QuantReLU(bit_width=self.bit_width, return_quant_tensor=self.return_quant)
        self.fc2 = qnn.QuantLinear(n_hidden, n_hidden, bias=True, weight_bit_width=self.bit_width, weight_quant=CommonInt8WeightPerTensorFloat)
        self.quant_relu2 = qnn.QuantReLU(bit_width=self.bit_width, return_quant_tensor=self.return_quant)
        self.fc3 = qnn.QuantLinear(n_hidden, n_output, bias=True, weight_bit_width=self.bit_width, weight_quant=CommonInt8WeightPerTensorFloat)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.fc1(x)
        x = self.quant_relu1(x)
        x = self.fc2(x)
        x = self.quant_relu2(x)
        x = self.fc3(x)
        return x

# --- 4. EXPORT & HELPER ---

def get_file_size_mb(file_path):
    if os.path.exists(file_path): return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

def calculate_sparsity(model):
    zeros = 0
    total = 0
    for m in model.modules():
        if isinstance(m, qnn.QuantLinear):
            w = m.weight.detach().cpu().numpy()
            zeros += np.count_nonzero(w == 0)
            total += w.size
    return (zeros / total * 100) if total > 0 else 0.0

def export_pytorch_weights(pytorch_model, output_path):
    """
    (Verbessert) Exportiert Float-Gewichte inkl. Statistiken (Min/Max/Mean) wie in deinen alten Logs.
    """
    np.set_printoptions(threshold=sys.maxsize, linewidth=200, edgeitems=sys.maxsize, suppress=True, precision=4)
    with open(output_path, "w") as f:
        f.write(f"MODELL GEWICHTE EXPORT (Detailed)\nZeitstempel: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        for name, module in pytorch_model.named_modules():
            if isinstance(module, qnn.QuantLinear):
                w = module.weight.detach().cpu().numpy()
                zeros = np.count_nonzero(w == 0)
                sparsity = (zeros / w.size) * 100
                
                # --- DAS IST NEU (aus deinem alten File übernommen) ---
                w_min = w.min()
                w_max = w.max()
                w_mean = w.mean()
                # ------------------------------------------------------
                
                f.write(f"LAYER: {name}\n")
                f.write(f"Shape: {w.shape}\n")
                f.write(f"Nullen: {zeros} ({sparsity:.2f}%)\n")
                f.write(f"Statistik: Min={w_min:.4f}, Max={w_max:.4f}, Mean={w_mean:.4f}\n")
                f.write("-" * 20 + " WERTE " + "-" * 20 + "\n")
                f.write(f"{w}\n\n")
                
                if module.bias is not None:
                    b = module.bias.detach().cpu().numpy()
                    f.write(f"LAYER: {name} (Bias)\n")
                    f.write(f"Statistik: Min={b.min():.4f}, Max={b.max():.4f}\n")
                    f.write(f"{b}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
                
    print(f" -> [REPORT] Original Weights (Detailed): {os.path.basename(output_path)}")

def export_compressed_weights(model, output_path):
    """Exportiert Integer-Gewichte."""
    np.set_printoptions(threshold=sys.maxsize, linewidth=200, edgeitems=sys.maxsize)
    with open(output_path, "w") as f:
        f.write(f"=== COMPRESSED INTEGER WEIGHTS ===\nFile: {os.path.basename(output_path)}\n\n")
        for name, module in model.named_modules():
            if isinstance(module, qnn.QuantLinear):
                try:
                    w_int = module.quant_weight().int().detach().cpu().numpy()
                    f.write(f"LAYER: {name}\n  Shape: {w_int.shape}\n  Values (Int):\n{w_int}\n\n")
                    if module.bias is not None:
                        b = module.bias.detach().cpu().numpy()
                        f.write(f"LAYER: {name} (Bias - Float Ref)\n  Values:\n{b}\n")
                    f.write("-" * 80 + "\n")
                except: pass
    print(f" -> [REPORT] Compressed Weights (Int): {os.path.basename(output_path)}")

def parse_config_from_path(folder_path):
    match = re.search(r"qb(\d+)_p(\d+)", os.path.basename(folder_path))
    if match: return int(match.group(1))
    return 3 

# --- 5. HAUPTLOGIK ---
def run_home_eval(target_path, n_samples, weights_only=False):
    # Dataset Detect
    if "_E" in target_path or "_edge" in target_path.lower():
        current_data_path = EDGE_PATH; print(f" -> Erkannt: EdgeIIoT.")
    else:
        current_data_path = CIC_PATH; print(f" -> Erkannt: CICIoT.")

    if not os.path.exists(current_data_path): print("FEHLER: Daten fehlen."); return
    data = np.load(current_data_path)
    
    # Robust Load
    try:
        X_test = data['X_test_scaled']; y_test = data['y_test_encoded']
        if y_test.ndim > 1: y_test = y_test.flatten()
    except KeyError:
        if 'X_test' in data: X_test = data['X_test']; y_test = data['y_test']
        else: print("FEHLER: Keine Daten gefunden."); return

    le = LabelEncoder().fit(y_test)
    calibration_data = X_test[:100]

    # Modelle suchen
    print(f"Suche Modelle in: {target_path}")
    search_pattern = os.path.join(target_path, "**", "*.pth")
    all_files = glob.glob(search_pattern, recursive=True)
    # Filter: Nur Basis .pth Dateien
    model_files = [f for f in all_files if "rep_" in f and "_COMPRESSED" not in f and ".txt" not in f]
    model_files.sort()

    if not model_files: print("Keine Modelle gefunden."); return
    print(f"Gefunden: {len(model_files)} Basis-Modelle.")

    for model_path in model_files:
        run_dir = os.path.dirname(model_path)
        parent_dir = os.path.dirname(run_dir)
        print(f"\n--- Bearbeite: {os.path.basename(parent_dir)} / {os.path.basename(run_dir)} ---")

        compressed_path = model_path.replace(".pth", "_COMPRESSED.pth")
        weights_pytorch_path = model_path.replace(".pth", "_weights_original.txt")
        weights_compressed_path = model_path.replace(".pth", "_weights_compressed.txt")
        
        # 1. LOAD & SPARSITY
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            real_hidden = state_dict['fc1.weight'].shape[0]
            real_features = state_dict['fc1.weight'].shape[1]
            real_output = state_dict['fc3.weight'].shape[0]
        except Exception as e: print(f"File Error: {e}"); continue
        
        qbits = parse_config_from_path(parent_dir)
        model = QATPrunedSimpleNet(real_features, real_hidden, real_output, qbits)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        sparsity = calculate_sparsity(model)
        size_original_mb = get_file_size_mb(model_path)
        
        # EXPORT 1 (Jetzt mit Statistik!)
        export_pytorch_weights(model, weights_pytorch_path)
        print(f" -> Global Sparsity: {sparsity:.2f}%")

        if weights_only:
            continue

        # 2. KOMPILIEREN
        quantized_module = None
        if os.path.exists(compressed_path):
            print(" -> Lade vorhandenes Compressed Model...")
            try: quantized_module = torch.load(compressed_path)
            except: pass
        
        if quantized_module is None:
            print(f" -> Kompiliere neu...")
            try:
                quantized_module = compile_brevitas_qat_model(
                    model, torch.from_numpy(calibration_data), p_error=P_ERROR, verbose=False
                )
                try: torch.save(quantized_module, compressed_path)
                except: print(" -> Warnung: Konnte Compressed Model nicht serialisieren (Pickle).")
            except Exception as e: print(f"Compile Error: {e}"); continue

        size_compressed_mb = get_file_size_mb(compressed_path)
        
        # EXPORT 2
        export_compressed_weights(model, weights_compressed_path)

        # 3. KEY MEASUREMENT
        key_size_mb = "N/A"
        if FHE_MODE == "execute":
            print(" -> Generiere Keys für Messung...")
            try:
                quantized_module.fhe_circuit.keygen()
                tmp = os.path.join(run_dir, "temp.bin")
                quantized_module.fhe_circuit.server.save(tmp)
                key_size_mb = f"{get_file_size_mb(tmp):.2f} MB"
                os.remove(tmp)
            except: key_size_mb = "Error"

        metrics_dict = {
            "Original Size": f"{size_original_mb:.2f} MB",
            "Compressed Size": f"{size_compressed_mb:.2f} MB",
            "FHE Key Size": key_size_mb,
            "Sparsity": f"{sparsity:.2f}%",
            "FHE Bits": qbits
        }

        # 4. EXECUTE
        print(f" -> Starte {FHE_MODE} Evaluation ({n_samples} Samples)...")
        try:
            evaluate_fhe_model(
                quantized_module, 
                X_test, 
                y_test, 
                le, 
                os.path.basename(model_path), 
                run_dir, 
                FHE_MODE, 
                n_samples, 
                footprint_metrics=metrics_dict
            )
        except Exception as e: 
            print(f"Eval Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str, help="Pfad zum Meta-Run oder Rep-Ordner")
    parser.add_argument("--samples", type=int, default=10, help="Anzahl Samples (Default: 10)")
    parser.add_argument("--weights-only", action="store_true", help="Nur Gewichte exportieren.")
    
    args = parser.parse_args()
    
    run_home_eval(args.folder, args.samples, args.weights_only)