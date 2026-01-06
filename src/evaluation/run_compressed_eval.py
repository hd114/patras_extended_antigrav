import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat as CommonInt8WeightPerTensorFloat
from concrete.ml.torch.compile import compile_brevitas_qat_model

# --- 1. SETUP & IMPORTS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from concrete_evaluate_execute import evaluate_fhe_model
except ImportError:
    print(f"KRITISCHER FEHLER: 'concrete_evaluate_execute.py' nicht in '{current_dir}' gefunden.")
    sys.exit(1)

# --- 2. MODELL DEFINITION ---
class QATPrunedSimpleNet(nn.Module):
    def __init__(self, n_features, n_output, n_hidden_1, n_hidden_2, quantization_bits):
        super(QATPrunedSimpleNet, self).__init__()
        self.bit_width = quantization_bits
        self.return_quant = True 
        
        self.quant_inp = qnn.QuantIdentity(bit_width=self.bit_width, return_quant_tensor=self.return_quant)
        
        self.fc1 = qnn.QuantLinear(n_features, n_hidden_1, bias=True, 
                                   weight_bit_width=self.bit_width, weight_quant=CommonInt8WeightPerTensorFloat)
        self.relu1 = qnn.QuantReLU(bit_width=self.bit_width, return_quant_tensor=self.return_quant)
        
        self.fc2 = qnn.QuantLinear(n_hidden_1, n_hidden_2, bias=True, 
                                   weight_bit_width=self.bit_width, weight_quant=CommonInt8WeightPerTensorFloat)
        self.relu2 = qnn.QuantReLU(bit_width=self.bit_width, return_quant_tensor=self.return_quant)
        
        self.fc3 = qnn.QuantLinear(n_hidden_2, n_output, bias=True, 
                                   weight_bit_width=self.bit_width, weight_quant=CommonInt8WeightPerTensorFloat)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# --- 3. HELPER ---
def get_file_size_mb(file_path):
    if os.path.exists(file_path): return os.path.getsize(file_path) / (1024 * 1024)
    return 0.0

def load_compressed_model(pth_path, qbits):
    state_dict = torch.load(pth_path, map_location="cpu")
    n_features = state_dict['fc1.weight'].shape[1]
    n_hidden_1 = state_dict['fc1.weight'].shape[0]
    n_hidden_2 = state_dict['fc2.weight'].shape[0]
    n_output   = state_dict['fc3.weight'].shape[0]
    
    print(f"   [Loader] Erkannt: In={n_features}, H1={n_hidden_1}, H2={n_hidden_2}, Out={n_output}, Bits={qbits}")
    
    model = QATPrunedSimpleNet(n_features, n_output, n_hidden_1, n_hidden_2, qbits)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- 4. MAIN PIPELINE ---
def run_evaluation_pipeline(model_path, data_path, bits, mode, n_samples):
    print(f"--- Evaluiere: {os.path.basename(model_path)} ---")
    
    if not os.path.exists(data_path):
        print(f"FEHLER: Daten nicht gefunden: {data_path}")
        return
    
    data = np.load(data_path)
    if 'X_test_scaled' in data:
        X_test = data['X_test_scaled']
        y_test = data['y_test_encoded']
    else:
        X_test = data['X_test']
        y_test = data['y_test']

    if y_test.ndim > 1: y_test = y_test.flatten()
    le = LabelEncoder().fit(y_test)
    calibration_data = X_test[:100] 
    
    # 1. Modell laden
    try:
        model = load_compressed_model(model_path, bits)
    except Exception as e:
        print(f"FEHLER beim Laden des Modells: {e}")
        return
    
    # 2. FHE Kompilierung
    fhe_circuit_path = model_path.replace(".pth", ".fhe")
    quantized_module = None

    # Versuch zu laden (falls schon da)
    if os.path.exists(fhe_circuit_path):
        print("   [FHE] Lade existierenden FHE Circuit...")
        try: quantized_module = torch.load(fhe_circuit_path)
        except: pass
        
    # Neu kompilieren falls nötig
    if quantized_module is None:
        print("   [FHE] Starte Kompilierung...")
        start_c = time.time()
        try:
            quantized_module = compile_brevitas_qat_model(
                model, torch.from_numpy(calibration_data), verbose=False
            )
            print(f"   [FHE] Kompilierung fertig in {time.time() - start_c:.2f}s")
            
            # --- DER RETTER: Try-Except um das Speichern ---
            try:
                torch.save(quantized_module, fhe_circuit_path)
            except Exception as e_save:
                print(f"   [WARNUNG] Konnte kompiliertes Modell nicht speichern (Pickle-Fehler). Mache weiter im RAM. Fehler: {e_save}")
            # -----------------------------------------------

        except Exception as e:
            print(f"FEHLER bei Kompilierung: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 3. Key Generation
    key_size_mb = "N/A"
    if mode == "execute":
        print("   [FHE] Generiere Keys...")
        start_k = time.time()
        quantized_module.fhe_circuit.keygen()
        print(f"   [FHE] KeyGen fertig in {time.time() - start_k:.2f}s")
        
        # Key Größe messen
        # Key Größe messen
        tmp_key = model_path + ".keys.tmp"
        try:
            quantized_module.fhe_circuit.server.save(tmp_key)
            
            # FIX: Prüfen, ob eine ZIP erstellt wurde (Concrete ML Standard)
            if os.path.exists(tmp_key + ".zip"):
                size_mb = get_file_size_mb(tmp_key + ".zip")
                key_size_mb = f"{size_mb:.2f} MB"
                os.remove(tmp_key + ".zip")
            elif os.path.exists(tmp_key):
                size_mb = get_file_size_mb(tmp_key)
                key_size_mb = f"{size_mb:.2f} MB"
                os.remove(tmp_key)
            else:
                key_size_mb = "0.00 MB (File not found)"
                
        except Exception as e:
            key_size_mb = f"Error ({e})"

    # 4. Evaluierung
    run_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).replace(".pth", "")
    
    metrics = {
        "Model Size": f"{get_file_size_mb(model_path):.2f} MB",
        "FHE Key Size": key_size_mb,
        "FHE Bits": bits
    }
    
    evaluate_fhe_model(
        quantized_module,
        X_test,
        y_test,
        le,
        model_name,
        run_dir,
        mode,
        n_samples,
        footprint_metrics=metrics
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--mode", default="execute")
    parser.add_argument("--n_samples", type=int, default=100)
    
    args = parser.parse_args()
    
    run_evaluation_pipeline(
        args.model_path, 
        args.data_path, 
        args.bits, 
        args.mode, 
        args.n_samples
    )