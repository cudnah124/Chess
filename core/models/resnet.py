"""
SmallResNet - Unified model architecture for SFT and RL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with batch normalization"""
    
    def __init__(self, num_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class SmallResNet(nn.Module):
    """
    AlphaZero-style ResNet for Chess
    
    Architecture:
    - Input: (batch, 32, 8, 8) board state
    - Backbone: Residual tower
    - Output: (batch, 4672) policy logits + (batch, 1) value
    
    Args:
        config: Configuration object with model hyperparameters
    """
    
    def __init__(self, config):
        super(SmallResNet, self).__init__()
        self.config = config
        
        # Input convolution
        self.conv_input = nn.Conv2d(
            config.INPUT_CHANNELS, 
            config.NUM_CHANNELS, 
            kernel_size=3, 
            padding=1, 
            stride=1, 
            bias=False
        )
        self.bn_input = nn.BatchNorm2d(config.NUM_CHANNELS)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(config.NUM_CHANNELS) 
            for _ in range(config.NUM_RES_BLOCKS)
        ])
        
        # Policy head (Actor)
        self.policy_conv = nn.Conv2d(config.NUM_CHANNELS, 32, kernel_size=1, stride=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, config.ACTION_SIZE)
        
        # Value head (Critic)
        self.value_conv = nn.Conv2d(config.NUM_CHANNELS, 3, kernel_size=1, stride=1, bias=False)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(3 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 32, 8, 8)
        
        Returns:
            policy_logits: (batch, 4672) raw logits (NOT softmax)
            value: (batch, 1) position evaluation in [-1, 1]
        """
        # Input block
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 32 * 8 * 8)
        policy_logits = self.policy_fc(p)  # Raw logits
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 3 * 8 * 8)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))  # [-1, 1]
        
        return policy_logits, value
    
    def get_architecture_info(self):
        """Return model architecture information for validation"""
        return {
            'num_res_blocks': self.config.NUM_RES_BLOCKS,
            'num_channels': self.config.NUM_CHANNELS,
            'action_size': self.config.ACTION_SIZE,
            'input_channels': self.config.INPUT_CHANNELS,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
