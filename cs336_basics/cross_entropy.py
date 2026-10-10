import torch

def LossCrossEntropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 减去最大值
    inputs = inputs - inputs.max(dim=-1,keepdim=True).values
    exp_inputs = torch.exp(inputs)
    targets = torch.unsqueeze(targets,dim=-1).long()
    right_logits = torch.gather(inputs,-1 , targets)
    loss = (torch.log(exp_inputs.sum(dim=-1, keepdim=True))  - right_logits).mean()

    return loss


