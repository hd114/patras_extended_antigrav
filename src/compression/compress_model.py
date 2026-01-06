# src/compression/compress_model.py

"""
Modul zur Komprimierung von trainierten, strukturell ODER unstrukturell geprunten QAT-Modellen.

Liest ein 'state_dict' (.pth) eines großen Modells, 
analysiert die aktiven Neuronen (Zeilen) und erstellt ein neues, 
physisch kleineres Modell, das nur die aktiven Gewichte enthält.

Unterstützt:
- Standard Structured Pruning (.weight Keys)
- PyTorch Unstructured Pruning (.weight_orig + .weight_mask Keys)

Ausführung via CLI:
  python -m src.compression.compress_model --model_path <PFAD> --bits <BITS>
"""

import torch
import os
import logging
import argparse
import sys
from typing import Dict, Any, Optional
from collections import OrderedDict

# --- 1. Import Setup (aus deinem alten Skript übernommen) ---
# Versuche lokalen Import, sonst füge Project Root hinzu
try:
    from src.models.qat_model import QATPrunedSimpleNet
except ImportError:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    try:
        from src.models.qat_model import QATPrunedSimpleNet
    except ImportError:
        # Fallback: Die Klasse lokal definieren, falls Importe fehlschlagen (für Standalone)
        # Dies ist nur ein Fallback, normalerweise sollte der Import oben klappen
        from src.evaluation.run_compressed_eval import QATPrunedSimpleNet

# Logger Setup
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ZERO_THRESHOLD = 1e-9

# --- 2. Neue Hilfsfunktionen (für Robustheit) ---

def get_weight(state_dict, layer_name):
    """
    Holt Gewichte robust, egal ob 'weight' oder 'weight_orig' + 'mask' (PyTorch Pruning).
    """
    # Fall A: Normales Gewicht
    if f"{layer_name}.weight" in state_dict:
        return state_dict[f"{layer_name}.weight"]
    
    # Fall B: PyTorch Pruning (unstructured Container)
    elif f"{layer_name}.weight_orig" in state_dict and f"{layer_name}.weight_mask" in state_dict:
        # logger.info(f"   [Info] Rekonstruiere geprunte Gewichte für {layer_name} (aus _orig + _mask)...")
        w_orig = state_dict[f"{layer_name}.weight_orig"]
        mask = state_dict[f"{layer_name}.weight_mask"]
        return w_orig * mask # Maske anwenden
    
    # Fall C: DataParallel Wrapper
    elif f"module.{layer_name}.weight" in state_dict:
        return state_dict[f"module.{layer_name}.weight"]
        
    else:
        # Debugging Hilfe: Zeige verfügbare Keys, die ähnlich aussehen
        available = [k for k in state_dict.keys() if layer_name in k]
        raise KeyError(f"Konnte Gewichte für '{layer_name}' nicht finden. Verfügbare Keys für diesen Layer: {available}")

def get_bias(state_dict, layer_name):
    """Holt Bias robust."""
    if f"{layer_name}.bias" in state_dict:
        return state_dict[f"{layer_name}.bias"]
    elif f"module.{layer_name}.bias" in state_dict:
        return state_dict[f"module.{layer_name}.bias"]
    else:
        return None

# --- 3. Hauptlogik ---

def analyze_pruned_structure(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Analysiert ein state_dict auf aktive (nicht-null) Zeilen."""
    logger.info("Analysiere Pruning-Struktur...")

    # Robustes Laden der Gewichte (Hier stürzte das alte Skript ab)
    try:
        fc1_w = get_weight(state_dict, "fc1")
        fc2_w = get_weight(state_dict, "fc2")
    except KeyError as e:
        logger.error(f"Struktur-Analyse fehlgeschlagen: {e}")
        raise e

    # 1. Analysiere fc1 (Zeilenaktivität)
    fc1_row_activity = torch.sum(torch.abs(fc1_w), dim=1)
    active_h1_indices = torch.where(fc1_row_activity > ZERO_THRESHOLD)[0]
    n_hidden1_new = len(active_h1_indices)

    if n_hidden1_new == 0:
        logger.error("Keine aktiven Neuronen in fc1 gefunden! Abbruch.")
        raise ValueError("fc1 ist vollständig tot.")
    logger.info(f"  fc1: {n_hidden1_new} von {fc1_w.shape[0]} Neuronen sind aktiv.")

    # 2. Analysiere fc2 (Zeilenaktivität)
    fc2_row_activity = torch.sum(torch.abs(fc2_w), dim=1)
    active_h2_indices = torch.where(fc2_row_activity > ZERO_THRESHOLD)[0]
    n_hidden2_new = len(active_h2_indices)
    
    if n_hidden2_new == 0:
        logger.error("Keine aktiven Neuronen in fc2 gefunden! Abbruch.")
        raise ValueError("fc2 ist vollständig tot.")
    logger.info(f"  fc2: {n_hidden2_new} von {fc2_w.shape[0]} Neuronen sind aktiv.")

    return {
        "active_h1_indices": active_h1_indices,
        "active_h2_indices": active_h2_indices,
        "n_hidden1_new": n_hidden1_new,
        "n_hidden2_new": n_hidden2_new
    }

def create_and_copy_weights(
    orig_state_dict: Dict[str, torch.Tensor],
    analysis: Dict[str, Any],
    model_config: Dict[str, Any]
) -> QATPrunedSimpleNet:
    """Erstellt ein neues, kleineres Modell und kopiert Gewichte."""
    
    n_h1_new = analysis["n_hidden1_new"]
    n_h2_new = analysis["n_hidden2_new"]
    idx_h1 = analysis["active_h1_indices"]
    idx_h2 = analysis["active_h2_indices"]
    
    logger.info(f"Erstelle neues Modell: (In: {model_config['input_size']}, H1: {n_h1_new}, H2: {n_h2_new}, Out: {model_config['num_classes']})")

    # Flexibles Modell instanziieren
    # Wir nutzen hier eine generische Initialisierung, die mit deiner QATPrunedSimpleNet Klasse kompatibel ist
    # HINWEIS: Falls deine Klasse unterschiedliche Init-Argumente hat, passe dies an.
    # Hier nutze ich die Signatur, die wir in run_compressed_eval.py definiert haben:
    try:
        # Versuch 1: Init wie im Eval Skript (einfach)
        new_model = QATPrunedSimpleNet(
            n_features=model_config["input_size"],
            n_output=model_config["num_classes"],
            n_hidden_1=n_h1_new,
            n_hidden_2=n_h2_new,
            quantization_bits=model_config["qlinear_args_config"]["weight_bit_width"]
        )
    except TypeError:
        # Versuch 2: Init wie im alten compress Skript (detailliert)
        new_model = QATPrunedSimpleNet(
            input_size=model_config["input_size"],
            num_classes=model_config["num_classes"],
            n_hidden_1=n_h1_new,
            n_hidden_2=n_h2_new,
            qlinear_args_config=model_config["qlinear_args_config"],
            qidentity_args_config=model_config["qidentity_args_config"],
            qrelu_args_config=model_config["qrelu_args_config"],
            dropout_rate1=model_config.get("dropout_rate1", 0.0),
            dropout_rate2=model_config.get("dropout_rate2", 0.0)
        )

    new_state_dict = new_model.state_dict()
    copied_state_dict = OrderedDict()

    logger.info("Starte Kopiervorgang der Gewichte...")

    # Wir iterieren explizit über die Layer, um die get_weight Funktion zu nutzen
    # FC1
    w1 = get_weight(orig_state_dict, "fc1")
    copied_state_dict["fc1.weight"] = w1[idx_h1, :]
    
    b1 = get_bias(orig_state_dict, "fc1")
    if b1 is not None: copied_state_dict["fc1.bias"] = b1[idx_h1]

    # FC2
    w2 = get_weight(orig_state_dict, "fc2")
    temp = w2[idx_h2, :] # Zeilen filtern
    copied_state_dict["fc2.weight"] = temp[:, idx_h1] # Spalten filtern
    
    b2 = get_bias(orig_state_dict, "fc2")
    if b2 is not None: copied_state_dict["fc2.bias"] = b2[idx_h2]

    # FC3
    w3 = get_weight(orig_state_dict, "fc3")
    copied_state_dict["fc3.weight"] = w3[:, idx_h2] # Spalten filtern
    
    b3 = get_bias(orig_state_dict, "fc3")
    if b3 is not None: copied_state_dict["fc3.bias"] = b3 # Output Bias bleibt

    # Metadaten kopieren (wenn vorhanden)
    for key, param in orig_state_dict.items():
        if "quant" in key and key in new_state_dict:
             if param.shape == new_state_dict[key].shape:
                 copied_state_dict[key] = param

    # Laden
    new_model.load_state_dict(copied_state_dict, strict=False)
    return new_model

def compress_model_pipeline(
    original_pth_path: str,
    output_pth_path: str,
    model_config: Dict[str, Any]
) -> Optional[QATPrunedSimpleNet]:
    """Haupt-Pipeline."""
    try:
        logger.info(f"Lade state_dict: {original_pth_path}")
        if not os.path.exists(original_pth_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {original_pth_path}")
            
        orig_state_dict = torch.load(original_pth_path, map_location="cpu")
        
        # Automatische Erkennung Input/Output
        if model_config["input_size"] is None:
             # Nutze get_weight um auch bei unstructured pruning die Shape zu finden
             w1 = get_weight(orig_state_dict, "fc1")
             model_config["input_size"] = w1.shape[1]
             logger.info(f"Input-Size automatisch erkannt: {model_config['input_size']}")
        
        if model_config["num_classes"] is None:
             w3 = get_weight(orig_state_dict, "fc3")
             model_config["num_classes"] = w3.shape[0]
             logger.info(f"Num-Classes automatisch erkannt: {model_config['num_classes']}")

        analysis = analyze_pruned_structure(orig_state_dict)
        
        compressed_model = create_and_copy_weights(
            orig_state_dict,
            analysis,
            model_config
        )

        torch.save(compressed_model.state_dict(), output_pth_path)
        logger.info(f"GESPEICHERT: {output_pth_path}")
        return compressed_model

    except Exception as e:
        logger.error(f"Fehler: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Komprimiert ein gepruntes QAT Modell.")
    parser.add_argument("--model_path", type=str, required=True, help="Pfad zur .pth Datei")
    parser.add_argument("--output_path", type=str, default=None, help="Optionaler Output Pfad")
    parser.add_argument("--bits", type=int, default=3, help="Quantisierungs-Bits")
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--num_classes", type=int, default=None)

    args = parser.parse_args()
    
    orig_path = args.model_path
    if args.output_path:
        out_path = args.output_path
    else:
        base, ext = os.path.splitext(orig_path)
        out_path = f"{base}_COMPRESSED{ext}"

    config = {
        "input_size": args.input_size, 
        "num_classes": args.num_classes,
        "qlinear_args_config": {"weight_bit_width": args.bits, "bias": True},
        "qidentity_args_config": {"bit_width": args.bits, "return_quant_tensor": True},
        "qrelu_args_config": {"bit_width": args.bits, "return_quant_tensor": True},
        "dropout_rate1": 0.0, "dropout_rate2": 0.0
    }
    
    compress_model_pipeline(orig_path, out_path, config)