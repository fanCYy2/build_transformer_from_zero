import torch 
from jaxtyping import Float

from cs336_basics.linear import Linear


def silu(in_features: Float[torch.Tensor, " ..."]) -> Float[torch.Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)

class SwiGlu(torch.nn.Module):
    

    def __init__(self, d_ff, d_model, device = None, dtype = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))


if __name__ == "__main__" :
    print("hello swiglu")