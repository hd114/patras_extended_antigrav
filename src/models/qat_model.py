import torch
from torch import nn
import brevitas.nn as qnn
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType as CoreQuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector, value
from brevitas.quant.solver import ActQuantSolver, WeightQuantSolver

# =============================================================================
# 1. Brevitas Quantisierungs-Konfigurationen
# =============================================================================

class CommonQuant(ExtendedInjector):
    """
    Basis-Konfiguration für Quantisierung.
    Definiert Standardverhalten wie 'Narrow Range' (für Symmetrie)
    und Zero-Point bei 0 (wichtig für FHE).
    """
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width: int):
        if bit_width is None:
            return CoreQuantType.FP
        if bit_width == 1:
            return CoreQuantType.BINARY
        return CoreQuantType.INT


class CommonWeightQuant(CommonQuant, WeightQuantSolver):
    """
    Optimierte Konfiguration für GEWICHTE:
    1. scaling_impl_type = STATS:
       Der Skalierungsfaktor wird dynamisch anhand des maximalen Gewichtswerts
       (max(|w|)) gesetzt. Verhindert Model-Collapse bei Initialisierung.
    2. scaling_per_output_channel = True:
       Jedes Neuron (Output Channel) erhält einen eigenen Skalierungsfaktor.
       Massiver Genauigkeitsgewinn bei niedrigen Bitbreiten (2-4 Bit).
    """
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = 'max'
    scaling_per_output_channel = True


class CommonInputQuant(CommonQuant, ActQuantSolver):
    """
    Konfiguration für den INPUT-Layer:
    - Muss signed=True sein (Default von CommonQuant), da normalisierte 
      Input-Daten negativ sein können (z.B. StandardScaler).
    - PACT (PARAMETER): Lernt den optimalen Clipping-Bereich.
    """
    scaling_impl_type = ScalingImplType.PARAMETER
    min_val = -1.0
    max_val = 1.0


class CommonReLUQuant(CommonQuant, ActQuantSolver):
    """
    Spezialisierte Konfiguration für ReLU-Aktivierungen:
    - signed = False: WICHTIG! Da ReLU nie negativ ist, nutzen wir die Bits
      nur für positive Werte. Verdoppelt die Präzision (bei 2-Bit: 4 Stufen statt 2).
    - PACT (PARAMETER): Lernt den optimalen Clipping-Wert (z.B. schneidet erst bei 6.0 ab).
    """
    scaling_impl_type = ScalingImplType.PARAMETER
    scaling_stats_op = 'max'
    
    # Verhindert, dass der lernbare Scale <= 1e-10 wird.
    restrict_scaling_type = RestrictValueType.LOG_FP
    
    min_val = 0.0
    max_val = 6.0   # Guter Startwert für ReLU
    signed = False  # <--- Der "Trick" für 2-Bit Performance


# =============================================================================
# 2. Hilfsfunktionen
# =============================================================================

def get_pruning_summary(model):
    """
    Erstellt eine Zusammenfassung der Sparsity (Null-Werte) im Modell.
    """
    summary = {}
    for name, module in model.named_modules():
        if isinstance(module, qnn.QuantLinear):
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.detach().cpu().numpy()
                total = weight.size
                
                # Prüfe auf exakte Nullen oder Masken
                if hasattr(module, 'weight_mask'):
                     zeros = (weight == 0).sum()
                else:
                     zeros = (weight == 0.0).sum()
                
                sparsity = float(zeros / total if total > 0 else 0)
                summary[name] = {
                    "total_weights": int(total),
                    "zero_weights": int(zeros),
                    "sparsity": round(sparsity, 4)
                }
    return summary


# =============================================================================
# 3. Modell-Definition
# =============================================================================

class QATPrunedSimpleNet(nn.Module):
    """
    Angepasste QAT Architektur mit optimierten Quantisierern.
    Unterstützt PBT (Pruning Before Training) auf konfigurierbaren Layern.
    """
    def __init__(
        self, 
        input_size, 
        num_classes, 
        n_hidden_1,        # Größe Layer 1
        n_hidden_2,        # Größe Layer 2
        qlinear_args_config, 
        qidentity_args_config, 
        qrelu_args_config, 
        dropout_rate1=0.0, 
        dropout_rate2=0.0,
        unpruned_neurons=None, # <--- Optional für PBT
        pbt_layers=None        # <--- Liste der Layer für PBT
    ):
        super().__init__()
        self.pruned_layers = set()
        
        # Speichere Dimensionen
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2

        # PBT Layer Default
        if pbt_layers is None:
            pbt_layers = []

        # --- Argument-Dictionaries zusammenstellen ---
        
        # 1. Gewichte: Nutze CommonWeightQuant (Per-Channel, Stats)
        qlinear_args = {
            "weight_quant": CommonWeightQuant,
            "bias_quant": qlinear_args_config.get("bias_quant", None),
            "weight_bit_width": qlinear_args_config.get("weight_bit_width", 8),
            "bias": qlinear_args_config.get("bias", True)
        }
        
        # 2. Input: Nutze CommonInputQuant (Signed, PACT)
        qidentity_args = {
            "act_quant": CommonInputQuant,
            "bit_width": qidentity_args_config.get("bit_width", 8),
            "return_quant_tensor": qidentity_args_config.get("return_quant_tensor", True)
        }
        
        # 3. ReLU: Nutze CommonReLUQuant (Unsigned, PACT)
        qrelu_args = {
            "act_quant": CommonReLUQuant,
            "bit_width": qrelu_args_config.get("bit_width", 8),
            "return_quant_tensor": qrelu_args_config.get("return_quant_tensor", True)
        }

        # --- Layer Definition ---
        
        # Eingang (Input Quantization)
        self.quant_inp = qnn.QuantIdentity(**qidentity_args)
        
        # FC1: Input -> Hidden 1
        self.fc1 = qnn.QuantLinear(input_size, n_hidden_1, **qlinear_args)
        self.relu1 = qnn.QuantReLU(**qrelu_args)
        self.dropout1 = nn.Dropout(p=dropout_rate1)
        
        # FC2: Hidden 1 -> Hidden 2
        self.fc2 = qnn.QuantLinear(n_hidden_1, n_hidden_2, **qlinear_args)
        self.relu2 = qnn.QuantReLU(**qrelu_args)
        self.dropout2 = nn.Dropout(p=dropout_rate2)
        
        # FC3: Hidden 2 -> Output
        self.fc3 = qnn.QuantLinear(n_hidden_2, num_classes, **qlinear_args)

        # --- Initialisierung ---
        # Wichtig: Dank ScalingImplType.STATS (in CommonWeightQuant) 
        # führt Xavier-Init jetzt nicht mehr zum Model-Collapse bei 2-Bit.
        for m in self.modules():
            if isinstance(m, qnn.QuantLinear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

        # --- PBT (Pruning Before Training) Simulation ---
        if unpruned_neurons is not None and unpruned_neurons > 0 and pbt_layers:
             self._apply_pbt(unpruned_neurons, pbt_layers)


    def _apply_pbt(self, unpruned_neurons, target_layers):
        """
        Wendet PBT persistent auf die in 'target_layers' angegebenen Layer an.
        """
        import torch.nn.utils.prune as prune

        for layer_name in target_layers:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                
                if not hasattr(layer, 'weight'):
                    continue
                    
                weights = layer.weight.data
                n_out, n_in = weights.shape
                
                if unpruned_neurons >= n_in:
                    # Wenn wir mehr behalten wollen als Inputs da sind, macht Pruning keinen Sinn
                    continue

                # 1. Maske erstellen
                mask = torch.zeros_like(weights)
                for i in range(n_out):
                    indices = torch.randperm(n_in)[:unpruned_neurons]
                    mask[i, indices] = 1.0
                
                # 2. Hook registrieren
                if prune.is_pruned(layer):
                    prune.remove(layer, 'weight')
                
                prune.custom_from_mask(layer, name='weight', mask=mask)
                print(f"PBT (Unstructured) angewendet auf: {layer_name} (Limit: {unpruned_neurons} inputs)")
            else:
                pass # Layer nicht gefunden (ignoriere)


    def forward(self, x):
        x = self.quant_inp(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    def unprune(self):
        from torch.nn.utils import prune
        for name, layer in self.named_modules():
            if hasattr(layer, 'weight_mask'):
                prune.remove(layer, "weight")