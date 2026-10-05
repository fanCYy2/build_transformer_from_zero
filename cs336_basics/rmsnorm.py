import torch

class RMSnorm(torch.nn.Module):
    def __init__(self, d_model: int, eps = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device= device, dtype= dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## 这里我们需要转化类型，因为不这样做可能会溢出
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt((torch.sum(x ** 2, dim = -1, keepdim=True) / self.d_model + self.eps))
        result = (x * self.weight) / rms

        return result.to(in_dtype)