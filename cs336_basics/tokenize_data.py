import argparse
import multiprocessing as mp
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization import find_chunk_boundaries

SPECIAL_TOKEN = "<|endoftext|>"

# 每个子进程各自持有一份 tokenizer，避免每个 chunk 都重新加载词表
_tokenizer = None


def init_worker(vocab_path: str, merges_path: str):
    global _tokenizer
    _tokenizer = Tokenizer.from_files(vocab_path, merges_path, [SPECIAL_TOKEN])


def encode_chunk(task: tuple[str, int, int]) -> np.ndarray:
    # 切分点都落在 special token 起始处，所以每个 chunk 可以独立编码
    input_path, start, end = task
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return np.array(_tokenizer.encode(text), dtype=np.uint16)


def tokenize_file(input_path: str, output_path: str, vocab_path: str, merges_path: str, num_workers: int, num_chunks: int):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, SPECIAL_TOKEN.encode("utf-8"))
    tasks = [(input_path, s, e) for s, e in zip(boundaries[:-1], boundaries[1:])]

    start_time = time.time()
    total = 0
    with mp.Pool(num_workers, initializer=init_worker, initargs=(vocab_path, merges_path)) as pool, open(output_path, "wb") as out:
        # imap 保证按顺序返回，直接追加写入二进制文件
        for i, ids in enumerate(pool.imap(encode_chunk, tasks)):
            ids.tofile(out)
            total += len(ids)
            print(f"[{i + 1}/{len(tasks)}] tokens: {total}, elapsed: {time.time() - start_time:.1f}s", flush=True)

    print(f"done: {output_path}, {total} tokens, dtype uint16")


def main():
    parser = argparse.ArgumentParser(description="把原始文本编码成 token id 并存成 uint16 二进制文件")
    parser.add_argument("--input", type=str, required=True, help="原始 txt 路径")
    parser.add_argument("--output", type=str, required=True, help="输出 .bin 路径")
    parser.add_argument("--vocab", type=str, default="cs336_basics/vocab.json", help="词表路径")
    parser.add_argument("--merges", type=str, default="cs336_basics/merges.txt", help="merges 路径")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(), help="进程数")
    parser.add_argument("--num_chunks", type=int, default=200, help="切成多少块分发给进程")
    args = parser.parse_args()

    tokenize_file(args.input, args.output, args.vocab, args.merges, args.num_workers, args.num_chunks)


if __name__ == "__main__":
    main()
