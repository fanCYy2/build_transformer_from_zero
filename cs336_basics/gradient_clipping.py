import torch

def GradinetClipping(params: list, max_l2: float):
    params = list(params)
    sum = 0
    for p in params:
        if (p.grad is None):
             continue
        sum += torch.sum(p.grad ** 2)
    g_l2 = torch.sqrt(sum)
    if (g_l2 > torch.tensor(max_l2, dtype=torch.float64)):
        for p in params:
            if (p.grad is None):
                continue
            p.grad.mul_(max_l2 / (g_l2 + 1e-6))
    
    return None