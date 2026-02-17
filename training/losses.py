# training/losses.py    
import torch
import torch.nn as nn

def build_weighted_ce(class_counts, device, label_smoothing=0.02):
    counts = torch.tensor(class_counts, dtype=torch.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(class_counts)

    return nn.CrossEntropyLoss(
        weight=weights.to(device),
        label_smoothing=float(label_smoothing)
    )
