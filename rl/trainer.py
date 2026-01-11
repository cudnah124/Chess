import torch
import torch.nn as nn


class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, policy_pred, value_pred, policy_target, value_target):
        log_policy = torch.log_softmax(policy_pred, dim=1)
        policy_loss = -(policy_target * log_policy).sum(dim=1).mean()
        value_loss = self.mse(value_pred, value_target)
        total_loss = policy_loss + value_loss
        return total_loss, policy_loss, value_loss


def train_on_buffer(model, replay_buffer, optimizer, criterion, device, epochs=5, batch_size=256):
    model.train()
    
    batches_per_epoch = max(1, len(replay_buffer) // batch_size)
    total_steps = batches_per_epoch * epochs
    
    for step in range(total_steps):
        batch = replay_buffer.sample(batch_size)
        states, policies, values = zip(*batch)
        
        states = torch.stack([torch.FloatTensor(s) for s in states]).to(device)
        policies = torch.stack([torch.FloatTensor(p) for p in policies]).to(device)
        values = torch.FloatTensor(values).unsqueeze(1).to(device)
        
        policy_pred, value_pred = model(states)
        loss, p_loss, v_loss = criterion(policy_pred, value_pred, policies, values)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 20 == 0:
            epoch = step // batches_per_epoch + 1
            print(f"  Epoch {epoch}/{epochs}, Step {step}/{total_steps}: "
                  f"Loss={loss.item():.4f}, P={p_loss.item():.4f}, V={v_loss.item():.4f}")
