import argparse
import os
import time

import numpy as np
import torch

from cs336_basics.transformer_lm import TransformerLm
from cs336_basics.cross_entropy import LossCrossEntropy
from cs336_basics.adamw import AdamW
from cs336_basics.lr_schedule import LrCosineSchedule
from cs336_basics.gradient_clipping import GradinetClipping
from cs336_basics.data_loading import GetBatch
from cs336_basics.checkpointing import save_checkpoint, run_load_checkpoint as load_checkpoint

@torch.no_grad()
def evaluate(model, valid_data, args) -> float:
    # 取多个 batch 求平均，单个 batch 的 loss 噪声太大
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        inputs, targets = GetBatch(valid_data, args.batch_size, args.context_length, args.device)
        logits = model(inputs)
        losses.append(LossCrossEntropy(logits, targets).item())
    model.train()
    return sum(losses) / len(losses)


def training_together(args):
    # memmap 只建立映射，切片时才从磁盘读取
    train_data = np.memmap(args.train_data, dtype=args.data_dtype, mode="r")
    valid_data = np.memmap(args.valid_data, dtype=args.data_dtype, mode="r")
    print(f"train tokens: {len(train_data)}, valid tokens: {len(valid_data)}")

    model = TransformerLm(args.vocab_size, args.context_length, args.d_model, args.num_layers,
                          args.num_heads, args.d_ff, args.rope_theta).to(args.device)
    print(f"params: {sum(p.numel() for p in model.parameters())}")

    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay,
                      betas=tuple(args.betas), eps=args.eps)

    # 已有 checkpoint 就从中断处继续
    start_iter = 0
    if os.path.exists(args.checkpoint_path):
        start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
        print(f"resume from iter {start_iter}")
    os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)

    model.train()
    start_time = time.time()
    for it in range(start_iter, args.max_iters):
        # 学习率调度：写进 param_groups，AdamW.step 从这里读 lr
        lr = LrCosineSchedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = GetBatch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(inputs)
        loss = LossCrossEntropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        # 裁剪必须在 backward 之后、step 之前
        GradinetClipping(model.parameters(), args.max_l2)
        optimizer.step()

        if it % args.log_interval == 0:
            print(f"iter {it} | train loss {loss.item():.4f} | lr {lr:.2e} | time {time.time() - start_time:.1f}s")

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            valid_loss = evaluate(model, valid_data, args)
            print(f"iter {it} | valid loss {valid_loss:.4f}")

        # 保存的是下一步要跑的 iter，恢复时不会重复训练这一步
        if (it + 1) % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, it + 1, args.checkpoint_path)

    save_checkpoint(model, optimizer, args.max_iters, args.checkpoint_path)
    print(f"done, checkpoint saved to {args.checkpoint_path}")

def register_params():
    parser = argparse.ArgumentParser(description="注册参数")

    # 模型结构
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--context_length", type=int, default=256, help="上下文长度")
    parser.add_argument("--d_model", type=int, default=512, help="隐藏维度")
    parser.add_argument("--num_layers", type=int, default=4, help="transformer block 层数")
    parser.add_argument("--num_heads", type=int, default=16, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=1344, help="前馈层维度")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE 的 theta")

    # 优化器
    parser.add_argument("--max_lr", type=float, default=1e-3, help="最大学习率")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="最小学习率")
    parser.add_argument("--warmup_iters", type=int, default=500, help="warmup 步数")
    parser.add_argument("--cosine_cycle_iters", type=int, default=5000, help="余弦退火结束的步数")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.95], help="AdamW 的 beta1 beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW 的 eps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减")
    parser.add_argument("--max_l2", type=float, default=1.0, help="梯度裁剪的 L2 范数上限")

    # 训练过程
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--max_iters", type=int, default=5000, help="总训练步数")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                        help="训练设备")
    parser.add_argument("--log_interval", type=int, default=10, help="每隔多少步打印训练 loss")
    parser.add_argument("--eval_interval", type=int, default=200, help="每隔多少步评估验证 loss")
    parser.add_argument("--eval_iters", type=int, default=20, help="评估时取多少个 batch 求平均")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="每隔多少步保存 checkpoint")

    # 路径
    parser.add_argument("--train_data", type=str, required=True, help="训练集 token 文件路径")
    parser.add_argument("--valid_data", type=str, required=True, help="验证集 token 文件路径")
    parser.add_argument("--data_dtype", type=str, default="uint16", help="token 文件的 numpy dtype")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/ckpt.pt", help="checkpoint 保存路径")

    return parser.parse_args()


def main():
    args = register_params()
    training_together(args)

if __name__ == "__main__":
    main()