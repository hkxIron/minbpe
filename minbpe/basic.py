"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

But:
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.
"""

from .base import Tokenizer, get_bigram_stats, replace_bigram_by_id
from typing import List,Optional,Dict,Tuple


class BasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text:str, vocab_size:int, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        """
        一个中文字符一般需要3个byte
        >>> "你好".encode('utf8')
        b'\xe4\xbd\xa0\xe5\xa5\xbd'
        >>> list("你好".encode('utf8'))
        [228, 189, 160, 229, 165, 189]
        """
        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        bigram_merge_table:Dict[(int, int), int] = {} # (int, int) -> int
        vocab:Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive bigram appears
            # Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
            bigram_to_count:Dict[(int, int), int] = get_bigram_stats(ids)
            # find the bigram with the highest count
            bigram = max(bigram_to_count, key=bigram_to_count.get)
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of bigram in ids with idx
            # Example: ids = [1, 2, 3, 1, 2], bigram = (1, 2), idx = 4
            # -> [4, 3, 4]
            ids = replace_bigram_by_id(ids, bigram, idx)
            # save the merge
            bigram_merge_table[bigram] = idx
            vocab[idx] = vocab[bigram[0]] + vocab[bigram[1]] # 两个btypes直接相加
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {bigram} -> {idx}, ({vocab[bigram[0]]},{vocab[bigram[1]]}->{vocab[idx]}) for {vocab[idx]} had {bigram_to_count[bigram]} occurrences")

        # save class variables
        # merges_table:value的值越小代表pair在语料中出现得越频繁
        self.bigram_merge_table:Dict[(int, int), int] = bigram_merge_table # used in encode()
        self.vocab:Dict[int, bytes] = vocab   # used in decode()

    def decode(self, ids:List[int])->str:
        # given ids (list of integers), return Python string
        text_bytes:bytes = b"".join(self.vocab[idx] for idx in ids)
        text:str = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text:str)->List[int]:
        """
        >>> list("你好".encode('utf8'))
        [228, 189, 160, 229, 165, 189]
        """
        # given a string text, return the token ids
        text_bytes:bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes) # list of integers in range 0..255

        while len(ids) >= 2:
            # find the bigram with the lowest merge index
            bigram_to_count:Dict[(int, int), int] = get_bigram_stats(ids)
            # 从ids中获取这样的pair,该pair在merge_table中具有最小的idx, 这样的idx是train时在语料中出现最频繁的bigram
            bigram = min(bigram_to_count, key=lambda p: self.bigram_merge_table.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single bigram, and the min will be
            # just the first bigram in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if bigram not in self.bigram_merge_table:
                break # nothing else can be merged anymore

            # otherwise let's merge the best bigram (lowest merge index)
            idx = self.bigram_merge_table[bigram]
            ids = replace_bigram_by_id(ids, bigram, idx)
        return ids
