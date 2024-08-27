"""
Implements the GPT-4 Tokenizer as a light wrapper around the RegexTokenizer.
Note that this is a pretrained tokenizer. By default and inside init(), it
loads the pretrained tokenizer from the `cl100k_base` tokenizer of tiktoken.
"""
from typing import *

import tiktoken
from .regex_tokenizer import RegexTokenizer
from .util import merge_bigram_by_table


def byte_pair_encode(mergeable_ranks:Dict[bytes, int], pair_bytes:bytes, max_rank:int)->List[bytes]:
    # 将
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    byte_list:List[bytes] = [bytes([b]) for b in pair_bytes]
    # 寻找byte的bigram
    while True:
        min_idx = None
        min_rank = None
        for i, bigram in enumerate(zip(byte_list[:-1], byte_list[1:])):
            rank = mergeable_ranks.get(bigram[0] + bigram[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        byte_bigram = byte_list[min_idx] + byte_list[min_idx + 1]
        byte_list = byte_list[:min_idx] + [byte_bigram] + byte_list[min_idx + 2:]

    return byte_list

def recover_merge_table(mergeable_ranks:dict[bytes, int]):
    # the `bigram_merge_table` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    bigram_merge_table:Dict[Tuple[int,int], int] = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(byte_pair_encode(mergeable_ranks, pair_bytes=token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0:int = mergeable_ranks[pair[0]]
        ix1:int = mergeable_ranks[pair[1]]
        bigram_merge_table[(ix0, ix1)] = rank

    return bigram_merge_table

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches GPT-4's tokenizer."""

    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        # get the official tokenizer and its bigram_merge_table
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks:dict[bytes, int] = enc._mergeable_ranks
        # the bigram_merge_table are those of gpt4, but we have to recover them
        self.bigram_merge_table = recover_merge_table(mergeable_ranks)
        # reconstruct the vocab from the bigram_merge_table
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.bigram_merge_table.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab

        # now here is another tricky part.
        # for some reason, the tokens corresponding to individual bytes
        # are permuted in a different order. This is completely non-sensical
        # and probably historical, but therefore we have to deal with it here.
        """
        bytes([48]), 将整数转为对应的 bytes
        >>> bytes([48])
        b'0'
        """
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        # finally register the special tokens
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)

    def _encode_chunk(self, text_bytes):
        # before we start processing bytes, we have to permute them
        text_bytes = bytes(self.byte_shuffle[b] for b in text_bytes)
        #ids = super()._encode_chunk(text_bytes)
        ids = merge_bigram_by_table(list(text_bytes), self.bigram_merge_table)
        return ids

    def decode(self, ids):
        # we have to un-permute the bytes before we decode
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.inverse_byte_shuffle[b] for b in text_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    # this is a pretrained tokenizer, it is not intended to be trained
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError

    # save/load would require some thought.
    # we'd have to change save/load of base to add support for byte_shuffle...
    # alternatively, we could move byte_shuffle to base class, but that would
    # mean that we're making ugly our beautiful Tokenizer just to support
    # the GPT-4 tokenizer and its weird historical quirks around byte_shuffle.
    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, model_file):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")

    def save_vocab(self, vocab_file):
        # just for visualization purposes let's output the GPT-4 tokens
        # in the exact same format as the base class would.
        # simple run as:
        # python -c "from minbpe import GPT4Tokenizer; GPT4Tokenizer().save_vocab('gpt4.vocab')"
        from .util import render_token
        # build vocab being mindful of the byte shuffle
        vocab = {idx: bytes([self.inverse_byte_shuffle[idx]]) for idx in range(256)}
        for (p0, p1), idx in self.bigram_merge_table.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        # now merge the shuffled bytes and write to file
        inverted_merges = {idx: pair for pair, idx in self.bigram_merge_table.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")
