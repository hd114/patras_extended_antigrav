import torch
import pytest
import sys
import os

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.qat_model import QATPrunedSimpleNet

def test_model_initialization():
    """Test if model initializes correctly with default parameters."""
    input_size = 10
    num_classes = 2
    n_hidden_1 = 8
    n_hidden_2 = 8
    
    qlinear_args = {"weight_bit_width": 3, "bias": True}
    qidentity_args = {"bit_width": 3, "return_quant_tensor": True}
    qrelu_args = {"bit_width": 3, "return_quant_tensor": True}

    model = QATPrunedSimpleNet(
        input_size=input_size, 
        num_classes=num_classes, 
        n_hidden_1=n_hidden_1, 
        n_hidden_2=n_hidden_2,
        qlinear_args_config=qlinear_args,
        qidentity_args_config=qidentity_args,
        qrelu_args_config=qrelu_args
    )
    
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_model_forward():
    """Test if model forward pass works with random input."""
    input_size = 10
    num_classes = 2
    n_hidden_1 = 8
    n_hidden_2 = 8
    
    qlinear_args = {"weight_bit_width": 3, "bias": True}
    qidentity_args = {"bit_width": 3, "return_quant_tensor": True}
    qrelu_args = {"bit_width": 3, "return_quant_tensor": True}

    model = QATPrunedSimpleNet(
        input_size=input_size, 
        num_classes=num_classes, 
        n_hidden_1=n_hidden_1, 
        n_hidden_2=n_hidden_2,
        qlinear_args_config=qlinear_args,
        qidentity_args_config=qidentity_args,
        qrelu_args_config=qrelu_args
    )
    
    # Create dummy input
    batch_size = 5
    dummy_input = torch.randn(batch_size, input_size)
    
    # Forward pass
    output = model(dummy_input)
    
    assert output is not None
    assert output.shape == (batch_size, num_classes)
