import math

def LrCosineSchedule(it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters):
    if (it < warmup_iters):
        a_t = it / warmup_iters * max_lr
    if (it <= cosine_cycle_iters and it >= warmup_iters):
        a_t =  min_lr + 0.5 * (1 + math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters) * math.pi)) * (max_lr - min_lr)
    if(it > cosine_cycle_iters):
        a_t = min_lr
    return a_t