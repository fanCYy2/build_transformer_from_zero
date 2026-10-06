import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.max(dim=dim,keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim,keepdim=True)

def main():
    print("hello softmax!")

if __name__ == "__main__":
    main()