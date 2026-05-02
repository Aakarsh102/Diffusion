"""
Utility functions for LoMDM training and evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import math
from typing import Optional, Tuple
import os


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    loss: float,
    path: str,
):
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "loss": loss,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: Optional[torch.device] = None,
) -> Tuple[int, float]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device or "cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint.get("step", 0), checkpoint.get("loss", 0.0)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_grad_norm(model: nn.Module) -> float:
    """Compute gradient norm for logging."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


class GradScaler:
    """
    Simple gradient scaler for mixed precision training.
    
    Wraps torch.cuda.amp.GradScaler with a fallback for CPU.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        if self.enabled:
            self._scaler = torch.GradScaler()
        else:
            self._scaler = None
    
    def scale(self, loss):
        if self.enabled:
            return self._scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        if self.enabled:
            self._scaler.step(optimizer)
        else:
            optimizer.step()
    
    def update(self):
        if self.enabled:
            self._scaler.update()
    
    def unscale_(self, optimizer):
        if self.enabled:
            self._scaler.unscale_(optimizer)


