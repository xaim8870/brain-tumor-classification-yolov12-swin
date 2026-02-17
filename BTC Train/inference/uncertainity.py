import torch
import numpy as np

def enable_mc_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

@torch.no_grad()
def mc_dropout_predict(model, x, runs=20):
    model.eval()
    enable_mc_dropout(model)

    probs = []
    for _ in range(runs):
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    probs = np.stack(probs, axis=0)
    mean = probs.mean(axis=0)
    std = probs.std(axis=0)

    return mean, std
