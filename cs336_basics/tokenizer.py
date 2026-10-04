from collections.abc import Iterable, Iterator
import regex as re
from cs336_basics.train_bpe import read_data

# 与训练时一致的 GPT-2 预分词正则
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = special_tokens or []

        # special token 不在词表里就追加进去
        existing = set(self.vocab.values())
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in existing:
                self.vocab[len(self.vocab)] = b
                existing.add(b)

        # 反向词表 bytes -> id
        self.byte_to_id = {v: k for k, v in self.vocab.items()}
        # merge 优先级：越早创建的 merge 排名越靠前
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        # 预分词 -> token id 列表 的缓存
        self.cache: dict[bytes, list[int]] = {}

        # 按长度降序，保证重叠的 special token 优先匹配最长的那个
        if self.special_tokens:
            escaped = [re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)]
            self.special_pat = re.compile("(" + "|".join(escaped) + ")")
        else:
            self.special_pat = None
        self.special_set = set(self.special_tokens)

    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        vocab, merges_list = read_data(vocab_path, merges_path)
        return cls(vocab, merges_list, special_tokens)

    def _bpe(self, pretoken: bytes) -> list[int]:
        """对单个预分词按 merge 优先级反复合并，返回 token id 列表"""
        if pretoken in self.cache:
            return self.cache[pretoken]

        parts = [bytes([b]) for b in pretoken]
        while len(parts) > 1:
            # 找出当前排名最高（rank 最小）的相邻对
            best_rank, best_pair = None, None
            for pair in zip(parts, parts[1:]):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            # 合并所有该相邻对
            merged = best_pair[0] + best_pair[1]
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == best_pair[0] and parts[i + 1] == best_pair[1]:
                    new_parts.append(merged)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        ids = [self.byte_to_id[p] for p in parts]
        self.cache[pretoken] = ids
        return ids

    def _split_special(self, text: str) -> list[str]:
        """按 special token 切分，保留 special token 本身"""
        if self.special_pat is None:
            return [text]
        return [s for s in self.special_pat.split(text) if s]

    def encode(self, text: str) -> list[int]:
        ids = []
        for segment in self._split_special(text):
            if segment in self.special_set:
                ids.append(self.byte_to_id[segment.encode("utf-8")])
                continue
            for m in PAT.finditer(segment):
                ids.extend(self._bpe(m.group().encode("utf-8")))
        return ids

    def _safe_cut(self, text: str) -> int:
        """
        找到 text 中一个安全的切分位置：切点之前的部分单独编码，
        与和后续文本拼接后再编码的结果一致。
        """
        # 末尾可能是某个 special token 的前缀，切点不能越过它
        limit = len(text)
        for tok in self.special_tokens:
            for k in range(min(len(tok) - 1, len(text)), 0, -1):
                if text.endswith(tok[:k]):
                    limit = min(limit, len(text) - k)
                    break

        # 收集所有预分词 / special token 的起始位置
        starts = []
        offset = 0
        for segment in self._split_special(text):
            if segment in self.special_set:
                starts.append(offset)
            else:
                starts.extend(offset + m.start() for m in PAT.finditer(segment))
            offset += len(segment)

        # 最后一个片段可能随后续输入继续增长，不能切在它之后
        cut = 0
        for s in starts[:-1]:
            if s <= limit:
                cut = s
        return cut

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式编码，内存只保留一小段尚未确定边界的缓冲"""
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            cut = self._safe_cut(buffer)
            if cut > 0:
                yield from self.encode(buffer[:cut])
                buffer = buffer[cut:]
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files("cs336_basics/vocab.json", "cs336_basics/merges.txt", ["<|endoftext|>"])
    sample = "Once upon a time, there was a little girl.<|endoftext|>The end."
    ids = tokenizer.encode(sample)
    print(ids)
    print(tokenizer.decode(ids))
