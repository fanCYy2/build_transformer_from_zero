import torch
from einops import rearrange

from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.scaled_dot_product_attention import attention

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, 
                d_model: int, 
                num_heads: int,
                theta : float | None = None,
                max_seq_len :int | None = None,
                device = None,
                dtype = None):
        
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.w_q = Linear(d_model, d_model, device= device, dtype= dtype)
        self.w_k = Linear(d_model, d_model, device= device, dtype= dtype)
        self.w_v = Linear(d_model, d_model, device= device, dtype= dtype)
        self.w_o = Linear(d_model, d_model, device= device, dtype= dtype)
        self.d_k = d_model // num_heads
        if theta is not None :
            self.rope = RotaryPositionalEmbedding(theta, d_k= self.d_k, max_seq_len = max_seq_len, device= device)
        else :
            self.rope = None
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
            # 投影
            q = self.w_q(x) # output shape: (... seq_len, d_model) 
            k = self.w_k(x)
            v = self.w_v(x)
            # 拆头  
            q = rearrange(q, "... seq_len (h d_k)-> ... h seq_len d_k", h = self.num_heads)
            k = rearrange(k, "... seq_len (h d_k)-> ... h seq_len d_k", h = self.num_heads)
            v = rearrange(v, "... seq_len (h d_k) -> ... h seq_len d_k", h = self.num_heads)
            # 旋转位置编码
            if self.rope is not None:
                if token_positions is None:
                    token_positions = torch.arange(x.shape[-2], device= x.device)  

                q = self.rope(q, token_positions)
                k = self.rope(k, token_positions)
            
            # 生成mask
            mask = torch.tril(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device), diagonal= 0)
            # 注意力结果
            output = attention(q, k, v, mask)
            # 合并头
            output = rearrange(output, "... h seq_len d_k-> ... seq_len (h d_k)")

            return self.w_o(output)


    
def main():
    print("hello MHA!")

if __name__ == "__main__":
    main()
