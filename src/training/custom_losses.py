#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss.
    
    Focal Loss was introduced in 'Focal Loss for Dense Object Detection' (https://arxiv.org/abs/1708.02002)
    to address class imbalance by down-weighting easy examples and focusing training on hard negatives.

    Args:
        alpha (float, optional): Weighting factor for modulating the loss.
                                 A common value is 0.25.
        gamma (float, optional): Focusing parameter. Higher values give more weight to
                                 hard, misclassified examples. A common value is 2.0.
        weight (Tensor, optional): A manual rescaling weight given to each class.
                                   If given, has to be a Tensor of size C (number of classes).
        reduction (str, optional): Specifies the reduction to apply to the output:
                                   'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                                   'mean': the sum of the output will be divided by the number of
                                   elements in the output, 'sum': the output will be summed.
                                   Default: 'mean'.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        if alpha is not None and not (0 <= alpha <= 1):
            raise ValueError(f"Alpha must be between 0 and 1, or None. Got {alpha}")
        if gamma < 0:
            raise ValueError(f"Gamma must be non-negative. Got {gamma}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction mode not supported: {reduction}")

        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight # Should be a Tensor of shape (C,) where C is num_classes
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits from the model (before Softmax).
                                   Shape: (N, C) where N is batch size, C is number of classes.
            targets (torch.Tensor): Ground truth labels.
                                    Shape: (N,) where each value is 0 <= targets[i] < C.
        Returns:
            torch.Tensor: The calculated focal loss.
        """
        num_classes = inputs.shape[1]
        if targets.max() >= num_classes or targets.min() < 0:
            raise ValueError(
                f"Target values must be in [0, {num_classes-1}]. "
                f"Got min: {targets.min()}, max: {targets.max()}"
            )

        # Calculate Cross Entropy loss without reduction, but with per-class weights if provided.
        # The 'weight' parameter in F.cross_entropy handles class weighting.
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)

        # Calculate pt (probability of the true class)
        pt = torch.exp(-ce_loss) # pt = p_t for the correct class

        # Calculate Focal Loss: alpha_t * (1 - pt)^gamma * ce_loss
        # The alpha term can be a scalar or per-class.
        # If self.alpha is a scalar, it's applied uniformly.
        # If self.alpha is intended to be per-class (like self.weight),
        # it needs to be gathered based on 'targets'.
        # For this implementation, we assume self.alpha is a scalar as per the init.
        # If you want per-class alpha, you'd need to adjust how alpha_t is derived.
        
        # Example of how alpha_t could be handled if self.alpha was a tensor (not current implementation):
        # if self.alpha is not None and isinstance(self.alpha, torch.Tensor):
        #     alpha_t = self.alpha.gather(0, targets.data.view(-1)) # Assuming self.alpha is (C,)
        #     focal_term = alpha_t * torch.pow(1 - pt, self.gamma)
        # elif self.alpha is not None: # Scalar alpha
        #     focal_term = self.alpha * torch.pow(1 - pt, self.gamma)
        # else: # No alpha modulation
        #     focal_term = torch.pow(1 - pt, self.gamma)

        # Current implementation with scalar alpha:
        if self.alpha is not None:
            focal_loss = self.alpha * torch.pow(1 - pt, self.gamma) * ce_loss
        else: # No alpha, behaves more like a standard CE scaled by (1-pt)^gamma
            focal_loss = torch.pow(1 - pt, self.gamma) * ce_loss
            
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: # 'none'
            return focal_loss

if __name__ == '__main__':
    # Example Usage (for testing the FocalLoss class directly)
    print("Testing FocalLoss...")
    num_classes_example = 5
    batch_size_example = 4

    # Mock model outputs (logits) and targets
    mock_logits = torch.randn(batch_size_example, num_classes_example)
    mock_targets = torch.randint(0, num_classes_example, (batch_size_example,))

    print(f"Mock Logits (shape: {mock_logits.shape}):\n{mock_logits}")
    print(f"Mock Targets (shape: {mock_targets.shape}):\n{mock_targets}")

    # 1. Test with default parameters
    print("\n--- Test 1: Default parameters ---")
    loss_fn_default = FocalLoss()
    loss_default = loss_fn_default(mock_logits, mock_targets)
    print(f"Loss (default): {loss_default.item()}")

    # 2. Test with custom gamma and alpha
    print("\n--- Test 2: Custom gamma and alpha ---")
    loss_fn_custom_params = FocalLoss(alpha=0.5, gamma=1.0)
    loss_custom_params = loss_fn_custom_params(mock_logits, mock_targets)
    print(f"Loss (alpha=0.5, gamma=1.0): {loss_custom_params.item()}")

    # 3. Test with class weights
    print("\n--- Test 3: With class weights ---")
    # Ensure weights are on the same device as logits/targets if using GPU
    mock_class_weights = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2]) 
    if mock_logits.is_cuda:
        mock_class_weights = mock_class_weights.to(mock_logits.device)
    
    loss_fn_weights = FocalLoss(weight=mock_class_weights)
    loss_with_weights = loss_fn_weights(mock_logits, mock_targets)
    print(f"Loss (with class weights {mock_class_weights.tolist()}): {loss_with_weights.item()}")

    # 4. Test with 'sum' reduction
    print("\n--- Test 4: Reduction 'sum' ---")
    loss_fn_sum = FocalLoss(reduction='sum')
    loss_sum = loss_fn_sum(mock_logits, mock_targets)
    print(f"Loss (reduction='sum'): {loss_sum.item()}")

    # 5. Test with 'none' reduction
    print("\n--- Test 5: Reduction 'none' ---")
    loss_fn_none = FocalLoss(reduction='none')
    loss_none = loss_fn_none(mock_logits, mock_targets)
    print(f"Loss (reduction='none', shape: {loss_none.shape}):\n{loss_none}")

    # Example demonstrating a case where a target is out of bounds (should raise error)
    # print("\n--- Test 6: Invalid target (should error) ---")
    # invalid_targets = torch.tensor([0, 1, num_classes_example, 3]) # one target is too high
    # try:
    #     loss_fn_default(mock_logits, invalid_targets)
    # except ValueError as e:
    #     print(f"Caught expected error: {e}")

    print("\nFocalLoss testing complete.")