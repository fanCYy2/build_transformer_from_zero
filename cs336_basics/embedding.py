import torch

class Embedding(torch.nn.Module):
    """
    input: num_embeddings就是词表的大小, embedding_dim是每个token映射到高维向量空间
    本质上embedding的过程是一个查表,只不过这个表需要训练
    """
    def __init__(self, num_embeddings, embedding_dim, device = None, dtype = None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
        
