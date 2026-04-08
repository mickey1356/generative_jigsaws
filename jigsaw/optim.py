import numpy as np

def sigmoid_scheduler(min_beta, max_beta, current_step, warmup_steps, max_steps, k=10):
    if current_step < warmup_steps:
        return min_beta
    progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_beta + (max_beta - min_beta) * (1 / (1 + np.exp(-k * (progress - 0.5))))

def cosine_warmup_lmb(it, warmup, total):
    if it < 0:
        return 0
    elif it < warmup:
        return it / warmup
    else:
        progress = (it - warmup) / max(1, total - warmup)
        return max(0, 0.5 * (1 + np.cos(np.pi * progress)))
