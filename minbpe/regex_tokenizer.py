"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""
from typing import Dict, Tuple, List, Union, Set

import regex as re
from .util import Tokenizer, get_bigram_stats, replace_bigram_by_id, bigram_merge, merge_bigram_by_table


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens:Dict[str, int] = {}
        self.inverse_special_tokens:Dict[int, str] = {}

    # def train(self, text:str, vocab_size:int, verbose=False):
    #     assert vocab_size >= 256
    #     num_merges = vocab_size - 256
    #
    #     """
    #     提前将文章用正则切分成多个小片段，基本以'空格+单词'为单位进行切分
    #     eg:
    #     'Copy paste of the Wikipedia article on Taylor Swift, as---\n\n'
    #     切分成下面的list
    #     ['Copy', ' paste', ' of', ' the', ' Wikipedia', ' article', ' on', ' Taylor', ' Swift', ',', ' as', '---\n\n']
    #     所以，可以看到 gpt3.5,gpt4里都是以 空格 + 单词 作为一个token
    #     """
    #     # split the text up into text chunks
    #     text_chunks:List[str] = re.findall(self.compiled_pattern, text)
    #
    #     # input text preprocessing
    #     chunk_id_list:List[List[int]] = [list(chunk.encode("utf-8")) for chunk in text_chunks]
    #
    #     # iteratively merge the most common pairs to create new tokens
    #     bigram_merge_table = {} # (int, int) -> int
    #     vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
    #     for i in range(num_merges):
    #         # count the number of times every consecutive pair appears
    #         bigram_to_count = {}
    #         for chunk_ids in chunk_id_list:
    #             # passing in stats will update it in place, adding up counts
    #             get_bigram_stats(chunk_ids, bigram_to_count)
    #         # find the pair with the highest count
    #         bigram:Tuple[int, int] = max(bigram_to_count, key=bigram_to_count.get)
    #         # mint a new token: assign it the next available id
    #         idx = 256 + i
    #         # replace all occurrences of pair in ids with idx
    #         chunk_id_list = [replace_bigram_by_id(chunk_ids, bigram, idx) for chunk_ids in chunk_id_list]
    #         # save the merge
    #         bigram_merge_table[bigram] = idx
    #         vocab[idx] = vocab[bigram[0]] + vocab[bigram[1]]
    #         # prints
    #         if verbose:
    #             print(f"merge {i+1}/{num_merges}: {bigram} -> {idx}, ({vocab[bigram[0]]},{vocab[bigram[1]]}) ->{vocab[idx]} for {vocab[idx]} had {bigram_to_count[bigram]} occurrences")
    #
    #     # save class variables
    #     self.bigram_merge_table = bigram_merge_table # used in encode()
    #     self.vocab = vocab   # used in decode()

    def train(self, text:str, vocab_size:int, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        """
        提前将文章用正则切分成多个小片段，基本以'空格+单词'为单位进行切分
        eg:
        'Copy paste of the Wikipedia article on Taylor Swift, as---\n\n'
        切分成下面的list
        ['Copy', ' paste', ' of', ' the', ' Wikipedia', ' article', ' on', ' Taylor', ' Swift', ',', ' as', '---\n\n']
        所以，可以看到 gpt3.5,gpt4里都是以 空格 + 单词 作为一个token
        """
        # split the text up into text chunks
        text_chunks:List[str] = re.findall(self.compiled_pattern, text)
        # input text preprocessing
        chunk_id_list:List[List[int]] = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        merge_table, vocab = bigram_merge(chunk_id_list, num_merges, verbose)
        self.bigram_merge_table:Dict[Tuple[int, int], int] = merge_table # used in encode()
        self.vocab:Dict[int, bytes] = vocab   # used in decode()

    def register_special_tokens_set(self, special_tokens:Set[str]):
        vocab_size = len(self.vocab)
        token2id = {t:i+vocab_size for i,t in enumerate(special_tokens)}
        self.register_special_tokens(token2id)
    """
    注意：special_tokens中的id都是自行设置的,缺点是会造成id不连续
    """
    def register_special_tokens(self, special_tokens:Dict[str, int] ):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids:List[int])->str:
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens: # id为special_token,直接查special_token表
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8")) # 注意：此处是append special_token的utf8的bytes，而不是string
            else:
                raise ValueError(f"invalid token id: {idx}")

        text_bytes = b"".join(part_bytes) # 在bytes维度直接join
        text = text_bytes.decode("utf-8", errors="replace")
        return text


    def encode_ignore_speical_tokens(self, text:str)->List[int]:
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            #chunk_ids = self._encode_chunk(chunk_bytes)
            #ids = list(text_bytes)
            chunk_ids = merge_bigram_by_table(list(chunk_bytes), self.bigram_merge_table)
            ids.extend(chunk_ids)
        return ids

    # def _encode_chunk(self, text_bytes:bytes)->List[int]:
    #     # return the token ids
    #     # let's begin. first, convert all bytes to integers in range 0..255
    #     ids = list(text_bytes)
    #     ids = merge_bigram_by_table(ids, self.bigram_merge_table)
    #     return ids
    # def _encode_chunk(self, text_bytes:bytes)->List[int]:
    #     # return the token ids
    #     # let's begin. first, convert all bytes to integers in range 0..255
    #     ids = list(text_bytes)
    #     while len(ids) >= 2:
    #         # find the pair with the lowest merge index
    #         stats = get_bigram_stats(ids)
    #         bigram = min(stats, key=lambda p: self.bigram_merge_table.get(p, float("inf")))
    #         # subtle: if there are no more bigram_merge_table available, the key will
    #         # result in an inf for every single pair, and the min will be
    #         # just the first pair in the list, arbitrarily
    #         # we can detect this terminating case by a membership check
    #         if bigram not in self.bigram_merge_table:
    #             break # nothing else can be merged anymore
    #         # otherwise let's merge the best pair (lowest merge index)
    #         idx = self.bigram_merge_table[bigram]
    #         ids = replace_bigram_by_id(ids, bigram, idx)
    #     return ids

    def encode(self, text:str, allowed_special:Union[str, Set]="none_raise")->List[int]:
        """
        Unlike encode_ignore_special_tokens, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special:Dict[str, int] = None
        if allowed_special == "all":
            # 所有special_token都生效
            special = self.special_tokens
        elif allowed_special == "none":
            # 所有special token都不生效
            special = {}
        elif allowed_special == "none_raise":
            # 所有的special token不允许在text中出现
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            # 只对allowed_special中的token作为special_token
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ignore_speical_tokens(text)

        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included

        """
         >>> re.escape('^abc')
         '\\^abc'
        special_chunks = re.split(special_pattern, text)
        >>>
        ['hello, world ',
         '|im_start|',
         ' is the start of message, then ',
         '|im_end|',
         ' is the end']
        """
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks:List[str] = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for token in special_chunks:
            if token in special:
                # special token直接查表
                # this is a special token, encode it separately as a special case
                ids.append(special[token])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ignore_speical_tokens(token))
        return ids
