from collections.abc import Iterable
from cs336_basics.train_bpe import read_data

class Tokenizer() :
    def __init__(self, vocab :dict, merges: list, special_tokens = None):

    @classmethod
    def from_files(cls, vocab_path : str, merges_path : str, special_tokens = None) :

        vocab, merges_list = read_data(vocab_path, merges_path)
        return cls(vocab, merges_list, special_tokens)
    
    def encode(self, text : str) -> list[int] :

    def encode_iterable(self, iterable : Iterable[str]) -> Iterable[int] :

    def decode(self, ids : list[str]) -> str :

if __name__ == "__main__" :
