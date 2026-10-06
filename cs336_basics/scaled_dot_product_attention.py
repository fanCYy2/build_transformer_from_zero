import torch
import math

from cs336_basics.softmax import softmax

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = q.shape[-1]
    attn_scores = torch.einsum("...ij, ...kj->...ik", q, k) / math.sqrt(d_k)
    attn_scores = torch.where(mask, attn_scores, torch.full_like(attn_scores, float('-inf')))

    output = torch.einsum("...ij, ...jk->...ik", softmax(attn_scores, -1), v)

    return output

def main():
    print("hello attention!")

if __name__ == "__main__":
    main()
