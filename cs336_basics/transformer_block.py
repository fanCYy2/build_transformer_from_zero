import torch

from cs336_basics.rmsnorm import RMSnorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.swiglu import SwiGlu

class Transformer_block(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int | None = None, theta: float| None = None):
        super().__init__()
        self.rmsnorm1 = RMSnorm(d_model)
        self.swiglu = SwiGlu(d_ff, d_model)
        self.rmsnorm2 = RMSnorm(d_model)
        if theta is not None:
            self.theta = theta
            self.max_seq_len  = max_seq_len
        else :
            self.theta = theta
            self.max_seq_len = max_seq_len

        self.mha = MultiheadSelfAttention(d_model, num_heads, self.theta, self.max_seq_len)

    def forward(self, x: torch.Tensor):
        y = x + self.mha(self.rmsnorm1(x)) 
        y = y + self.swiglu(self.rmsnorm2(y))

        return y
