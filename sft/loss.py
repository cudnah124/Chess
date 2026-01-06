"""
AlphaZero Loss - Combined policy and value loss for SFT
"""
import torch
import torch.nn as nn


class AlphaZeroLoss(nn.Module):
    """
    Combined loss function for AlphaZero-style training
    
    Loss = Policy Loss + Value Loss
    - Policy Loss: Cross-entropy between predicted and target policy
    - Value Loss: MSE between predicted and target value
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, policy_pred, value_pred, policy_target, value_target):
        """
        Compute combined loss
        
        Args:
            policy_pred: Policy logits from model (batch, 4672)
            value_pred: Value predictions from model (batch, 1)
            policy_target: Target policy distribution (batch, 4672)
            value_target: Target values (batch, 1)
        
        Returns:
            total_loss: Combined loss
            policy_loss: Policy component
            value_loss: Value component
        """
        # Policy loss: cross-entropy with soft targets
        log_policy = torch.log_softmax(policy_pred, dim=1)
        policy_loss = -(policy_target * log_policy).sum(dim=1).mean()
        
        # Value loss: MSE
        value_loss = self.mse(value_pred, value_target)
        
        # Combined loss
        total_loss = policy_loss + value_loss
        
        return total_loss, policy_loss, value_loss
