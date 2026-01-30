# src/models/fhe_tabular_transformer.py

import torch
import torch.nn as nn
import brevitas.nn as qnn 
from typing import Optional, Type, List

FHE_FRIENDLY_ACTIVATION = qnn.QuantReLU

# Klassen MultiHeadSelfAttention, PositionwiseFeedForward, TransformerEncoderBlock 
# bleiben wie in der vorherigen Version (mit dem Sqrt-Fix in MultiHeadSelfAttention).
# Ich füge sie hier der Vollständigkeit halber ein, die Änderung ist nur im TabularFeatureEmbedder.

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, 
                 quant_linear_layer: Type[nn.Module] = qnn.QuantLinear, 
                 dropout_rate: float = 0.1,
                 bias_in_linear: bool = True,
                 weight_bit_width: Optional[int] = None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model muss durch n_heads teilbar sein"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        linear_kwargs = {"bias": bias_in_linear}
        if quant_linear_layer == qnn.QuantLinear and weight_bit_width is not None:
            linear_kwargs["weight_bit_width"] = weight_bit_width
            
        self.q_linear = quant_linear_layer(d_model, d_model, **linear_kwargs)
        self.k_linear = quant_linear_layer(d_model, d_model, **linear_kwargs)
        self.v_linear = quant_linear_layer(d_model, d_model, **linear_kwargs)
        self.out_linear = quant_linear_layer(d_model, d_model, **linear_kwargs)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1) 
        self.scale_factor = (self.head_dim ** -0.5) if self.head_dim > 0 else 1.0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        q = self.q_linear(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20")) 
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.dropout(attention_probs) 
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.out_linear(context)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, 
                 quant_linear_layer: Type[nn.Module] = qnn.QuantLinear,
                 activation_module_class: Type[nn.Module] = FHE_FRIENDLY_ACTIVATION,
                 dropout_rate: float = 0.1,
                 bias_in_linear: bool = True,
                 weight_bit_width: Optional[int] = None,
                 activation_bit_width: Optional[int] = None):
        super().__init__()
        linear_kwargs = {"bias": bias_in_linear}
        act_kwargs = {}
        if quant_linear_layer == qnn.QuantLinear and weight_bit_width is not None:
            linear_kwargs["weight_bit_width"] = weight_bit_width
        if hasattr(activation_module_class, "bit_width") and activation_bit_width is not None:
             act_kwargs["bit_width"] = activation_bit_width
        elif issubclass(activation_module_class, (qnn.QuantReLU, qnn.QuantSigmoid, qnn.QuantHardTanh, qnn.QuantIdentity)) and activation_bit_width is not None:
             act_kwargs["bit_width"] = activation_bit_width
        self.linear1 = quant_linear_layer(d_model, ffn_dim, **linear_kwargs)
        self.activation = activation_module_class(**act_kwargs) if act_kwargs else activation_module_class()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = quant_linear_layer(ffn_dim, d_model, **linear_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 quant_linear_layer: Type[nn.Module] = qnn.QuantLinear,
                 activation_module_class: Type[nn.Module] = FHE_FRIENDLY_ACTIVATION,
                 dropout_rate: float = 0.1,
                 weight_bit_width: Optional[int] = None,
                 activation_bit_width: Optional[int] = None):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            d_model, n_heads, quant_linear_layer, dropout_rate, 
            weight_bit_width=weight_bit_width
        )
        self.norm1 = nn.LayerNorm(d_model) 
        self.dropout1 = nn.Dropout(dropout_rate)
        self.ffn = PositionwiseFeedForward(
            d_model, ffn_dim, quant_linear_layer, activation_module_class, dropout_rate,
            weight_bit_width=weight_bit_width, activation_bit_width=activation_bit_width
        )
        self.norm2 = nn.LayerNorm(d_model) 
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

# src/models/fhe_tabular_transformer.py

# ... (andere Klassen bleiben gleich) ...

class TabularFeatureEmbedder(nn.Module):
    """ Embeddet tabellarische Features. """
    def __init__(self, num_features: int, embedding_dim: int, 
                 quant_linear_layer: Type[nn.Module] = qnn.QuantLinear,
                 weight_bit_width: Optional[int] = None,
                 # NEU: activation_bit_width für Input-Quantisierung der einzelnen Embedder
                 activation_bit_width: Optional[int] = None): 
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        
        linear_kwargs = {"bias": True}
        if quant_linear_layer == qnn.QuantLinear and weight_bit_width is not None:
            linear_kwargs["weight_bit_width"] = weight_bit_width

        self.feature_embedders = nn.ModuleList()
        for _ in range(num_features):
            module_list = []
            # NEU: Füge einen QuantIdentity vor jedem QuantLinear hinzu, wenn activation_bit_width gegeben ist
            if activation_bit_width is not None and quant_linear_layer == qnn.QuantLinear:
                module_list.append(qnn.QuantIdentity(bit_width=activation_bit_width, return_quant_tensor=True))
            module_list.append(quant_linear_layer(1, embedding_dim, **linear_kwargs))
            self.feature_embedders.append(nn.Sequential(*module_list))


    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        if hasattr(x_input, 'value') and isinstance(x_input.value, torch.Tensor):
            tensor_to_slice = x_input.value
        else:
            tensor_to_slice = x_input

        if tensor_to_slice.ndim != 2 or tensor_to_slice.shape[1] != self.num_features:
            raise ValueError(
                f"TabularFeatureEmbedder: Unerwartete Form für tensor_to_slice. "
                f"Erwartet: (batch_size, {self.num_features}), Bekommen: {tensor_to_slice.shape}"
            )
            
        embedded_features = []
        for i in range(self.num_features):
            feature_slice = tensor_to_slice[:, i].unsqueeze(-1) 
            # Der Sequential Layer enthält jetzt ggf. den QuantIdentity
            embedded_features.append(self.feature_embedders[i](feature_slice)) 
        
        return torch.stack(embedded_features, dim=1)


class FHETabularTransformer(nn.Module):
    def __init__(self, num_features: int, num_classes: int, d_model: int, n_heads: int,
                 num_encoder_layers: int, ffn_dim: int,
                 dropout_rate: float = 0.1,
                 quant_linear_layer_name: str = "QuantLinear", 
                 activation_module_name: str = "QuantReLU", 
                 weight_bit_width: Optional[int] = None, 
                 activation_bit_width: Optional[int] = None 
                 ):
        super().__init__()
        # ... (Auflösung der Layer-Typen bleibt gleich) ...
        try:
            quant_linear_class = getattr(qnn, quant_linear_layer_name) if hasattr(qnn, quant_linear_layer_name) else getattr(nn, quant_linear_layer_name)
        except AttributeError:
            print(f"WARNUNG: quant_linear_layer '{quant_linear_layer_name}' nicht in brevitas.nn oder torch.nn gefunden. Fallback zu nn.Linear.")
            quant_linear_class = nn.Linear

        try:
            activation_class = getattr(qnn, activation_module_name) if hasattr(qnn, activation_module_name) else getattr(nn, activation_module_name)
        except AttributeError:
            print(f"WARNUNG: activation_module '{activation_module_name}' nicht in brevitas.nn oder torch.nn gefunden. Fallback zu nn.ReLU.")
            activation_class = nn.ReLU
            
        # Eingangsquantisierung für das GESAMTE Modell (bleibt wichtig)
        if activation_bit_width is not None:
            self.input_quant = qnn.QuantIdentity(
                bit_width=activation_bit_width,
                return_quant_tensor=True 
            )
            print(f"FHETabularTransformer: Globale Input QuantIdentity mit {activation_bit_width} bits initialisiert.")
        else:
            self.input_quant = nn.Identity()
            print("FHETabularTransformer: Globale Input QuantIdentity nicht initialisiert (activation_bit_width is None).")

        self.feature_embedder = TabularFeatureEmbedder(
            num_features, d_model, quant_linear_class, weight_bit_width,
            activation_bit_width=activation_bit_width # activation_bit_width hier durchreichen
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, ffn_dim, 
                                     quant_linear_class, activation_class, 
                                     dropout_rate, weight_bit_width, activation_bit_width)
             for _ in range(num_encoder_layers)]
        )
        
        linear_kwargs_head = {"bias": True}
        if quant_linear_class == qnn.QuantLinear and weight_bit_width is not None: # type: ignore
            linear_kwargs_head["weight_bit_width"] = weight_bit_width

        self.fc_out = quant_linear_class(d_model, num_classes, **linear_kwargs_head)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_quantized_input = self.input_quant(x) 
        x_embedded = self.feature_embedder(x_quantized_input) 
        
        batch_size = x.shape[0] 
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        
        x_processed = torch.cat([cls_tokens, x_embedded], dim=1) 
        x_processed = self.dropout(x_processed)

        for layer in self.encoder_layers:
            x_processed = layer(x_processed, mask=None) 
            
        cls_output = x_processed[:, 0, :] 
        output_logits = self.fc_out(cls_output) 
        return output_logits


class FHETabularTransformer(nn.Module):
    """
    Ein Transformer-Modell für tabellarische Daten mit FHE-freundlichen Komponenten.
    """
    def __init__(self, num_features: int, num_classes: int, d_model: int, n_heads: int,
                 num_encoder_layers: int, ffn_dim: int,
                 dropout_rate: float = 0.1,
                 quant_linear_layer_name: str = "QuantLinear", 
                 activation_module_name: str = "QuantReLU", 
                 weight_bit_width: Optional[int] = None, 
                 activation_bit_width: Optional[int] = None 
                 ):
        super().__init__()
        
        try:
            quant_linear_class = getattr(qnn, quant_linear_layer_name) if hasattr(qnn, quant_linear_layer_name) else getattr(nn, quant_linear_layer_name)
        except AttributeError:
            print(f"WARNUNG: quant_linear_layer '{quant_linear_layer_name}' nicht in brevitas.nn oder torch.nn gefunden. Fallback zu nn.Linear.")
            quant_linear_class = nn.Linear

        try:
            activation_class = getattr(qnn, activation_module_name) if hasattr(qnn, activation_module_name) else getattr(nn, activation_module_name)
        except AttributeError:
            print(f"WARNUNG: activation_module '{activation_module_name}' nicht in brevitas.nn oder torch.nn gefunden. Fallback zu nn.ReLU.")
            activation_class = nn.ReLU

        if activation_bit_width is not None:
            self.input_quant = qnn.QuantIdentity(
                bit_width=activation_bit_width,
                return_quant_tensor=True 
            )
            print(f"FHETabularTransformer: Input QuantIdentity mit {activation_bit_width} bits initialisiert.")
        else:
            self.input_quant = nn.Identity()
            print("FHETabularTransformer: Input QuantIdentity nicht initialisiert (activation_bit_width is None). Dies führt wahrscheinlich zu FHE-Kompilierungsfehlern.")

        self.feature_embedder = TabularFeatureEmbedder(
            num_features, d_model, quant_linear_class, weight_bit_width
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderBlock(d_model, n_heads, ffn_dim, 
                                     quant_linear_class, activation_class, 
                                     dropout_rate, weight_bit_width, activation_bit_width)
             for _ in range(num_encoder_layers)]
        )
        
        linear_kwargs_head = {"bias": True}
        if quant_linear_class == qnn.QuantLinear and weight_bit_width is not None: # type: ignore
            linear_kwargs_head["weight_bit_width"] = weight_bit_width

        self.fc_out = quant_linear_class(d_model, num_classes, **linear_kwargs_head)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_quantized_input = self.input_quant(x) 
        x_embedded = self.feature_embedder(x_quantized_input) 
        
        batch_size = x.shape[0] # Verwende ursprüngliches x für batch_size, da Form klar ist
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        
        x_processed = torch.cat([cls_tokens, x_embedded], dim=1) 
        x_processed = self.dropout(x_processed)

        for layer in self.encoder_layers:
            x_processed = layer(x_processed, mask=None) 
            
        cls_output = x_processed[:, 0, :] 
        output_logits = self.fc_out(cls_output) 
        return output_logits