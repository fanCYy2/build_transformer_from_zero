import torch

from cs336_basics.transformer_block import Transformer_block
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSnorm
from cs336_basics.linear import Linear
from cs336_basics.softmax import softmax

class TransformerLm(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformers_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.transformers_blocks.append(Transformer_block(d_model, num_heads, d_ff, context_length, rope_theta))
        self.rmsnorm = RMSnorm(d_model)
        self.linear = Linear(d_model, vocab_size)
    def forward(self, input: torch.Tensor):
        x = self.embedding(input)
        for block in self.transformers_blocks:
            x = block(x)
        y = self.rmsnorm(x)
        logits = self.linear(y)

        return logits
