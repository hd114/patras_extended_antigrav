import torch
from torch import nn

class MLPNet(nn.Module):
    """
    Simple Multilayer Perceptron for multiclass classification.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_classes: int = 15):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class EncMLPNet:
    """
    Encrypted version of MLPNet using TenSEAL for FHE inference.
    Supports multiple ReLU approximations.
    """
    def __init__(self, torch_nn: nn.Module, relu_variant: str = "relu3"):
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

        # Select ReLU approximation method
        self.relu_variant = relu_variant
        self.relu_map = {
            "relu1": self.approx_relu1,
            "relu2": self.approx_relu2,
            "relu3": self.approx_relu3
        }
        if relu_variant not in self.relu_map:
            raise ValueError(f"Unknown ReLU variant '{relu_variant}'. Choose from {list(self.relu_map.keys())}.")

    def forward(self, enc_x):
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = self.relu_map[self.relu_variant](enc_x)
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    # --- Different ReLU approximations ---
    @staticmethod
    def approx_relu1(enc_x):
        # Simple linear approximation: P(x) = 0.01 * x for x < 0, else x
        return enc_x.polyval([0, 1])

    @staticmethod
    def approx_relu2(enc_x):
        # Cubic polynomial approximation: P(x) = 0.1 * x^3 + 0.1 * x^2 + 0.9 * x
        return enc_x.polyval([0, 0.9, 0.1, 0.1])

    @staticmethod
    def approx_relu3(enc_x):
        # Best performing approximation: P(x) = 0.1 * x^3 + 0.9 * x
        return enc_x.polyval([0, 0.9, 0, 0.1])

    @staticmethod
    def sigmoid(enc_x):
        """
        Polynomial approximation of sigmoid.
        """
        return enc_x.polyval([0.5, 0.197, 0, -0.004])

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
