import torch
import os
from typing import IO, Any, BinaryIO

def save_checkpoint(model, optimizer, iteration, out):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    all_state = {"model_state": model_state, "optimizer_state": optimizer_state, "iteration" :iteration}

    torch.save(all_state, out)
   


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    all_state = torch.load(src)
    model.load_state_dict(all_state["model_state"])
    optimizer.load_state_dict(all_state["optimizer_state"])
    iteration = all_state["iteration"]
    
    return iteration
    