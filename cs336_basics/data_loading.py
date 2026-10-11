import torch
import numpy as np
import numpy.typing as npt

def GetBatch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # 起点 i 需满足 i + context_length <= len(dataset) - 1, randint 上界不包含
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = []
    targets = []
    for i in starts:
        inputs.append(dataset[i : i + context_length]) # 只切片, 对 memmap 只会读取需要的部分
        targets.append(dataset[i + 1 : i + 1 + context_length]) # target 是 input 右移一位
    inputs = torch.from_numpy(np.stack(inputs)).long().to(device)
    targets = torch.from_numpy(np.stack(targets)).long().to(device)

    return inputs, targets
