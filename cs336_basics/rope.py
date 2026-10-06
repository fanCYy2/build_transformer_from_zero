import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        position_vec = torch.arange(max_seq_len, device= device)
        freq_vec = 1 / torch.pow(torch.as_tensor(theta),(torch.arange(0, d_k, 2, device= device) / d_k)) 

        angle = torch.einsum("i,j->ij",position_vec, freq_vec)
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)
        self.register_buffer("sin_angle", sin_angle, persistent=False)
        self.register_buffer("cos_angle", cos_angle, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 查表: (..., seq_len) -> (..., seq_len, d_k/2)
        sin = self.sin_angle[token_positions]
        cos = self.cos_angle[token_positions]

        # 拆对: 偶数位是每对的第一个元素, 奇数位是第二个
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 2D 旋转
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # 交错拼回: (..., d_k/2, 2) -> (..., d_k)
        return torch.stack((out_even, out_odd), dim=-1).flatten(start_dim=-2)
        