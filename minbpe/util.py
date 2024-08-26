"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.

BPE算法：The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings
"""
import unicodedata
from typing import List,Optional,Dict,Tuple


# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

def get_bigram_stats(ids:List[int], counts:Optional[Dict[int,int]]=None):
    """
    统计所有的2-gram出现次数

    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def replace_bigram_by_id(ids:List[int], pair:Tuple[int, int], new_idx:int):
    """
    将ids中的所有的pair替换成new_idx

    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), new_idx=4
        -> [4, 3, 4]
    """
    replaced_ids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            replaced_ids.append(new_idx)
            i += 2  # 指针加2
        else:
            replaced_ids.append(ids[i])
            i += 1
    return replaced_ids

"""
ids_in_chunk_list: 每个chunk代表一段文本,里面包含很多id list
"""
def bigram_merge(ids_in_chunk_list:List[List[int]], num_merges:int, verbose=False):
    # iteratively merge the most common pairs to create new tokens
    bigram_merge_table:Dict[(int, int), int] = {} # (int, int) -> int
    vocab:Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
    for i in range(num_merges):
        # 注意：每次merge中,bigram_count需要重新统计
        # count the number of times every consecutive pair appears
        # count up the number of times every consecutive bigram appears
        # Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        bigram_to_count:Dict[(int,int), int] = {}
        for ids in ids_in_chunk_list:
            # passing in stats will update it in place, adding up counts
            # Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
            get_bigram_stats(ids, bigram_to_count)
        # find the pair with the highest count
        bigram:Tuple[int, int] = max(bigram_to_count, key=bigram_to_count.get)
        # mint a new token: assign it the next available id
        idx = 256 + i

        # replace all occurrences of bigram in ids with idx
        # Example: ids = [1, 2, 3, 1, 2], bigram = (1, 2), idx = 4
        # -> [4, 3, 4]
        # replace all occurrences of pair in ids with idx
        ids_in_chunk_list = [replace_bigram_by_id(chunk_ids, bigram, idx) for chunk_ids in ids_in_chunk_list]
        # save the merge
        bigram_merge_table[bigram] = idx
        vocab[idx] = vocab[bigram[0]] + vocab[bigram[1]]
        # prints
        if verbose:
            print(f"merge {i+1}/{num_merges}: {bigram} -> {idx}, ({vocab[bigram[0]]},{vocab[bigram[1]]}) ->{vocab[idx]} for {vocab[idx]} had {bigram_to_count[bigram]} occurrences")
    return bigram_merge_table, vocab

# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else: # control char
            # ord(x)获取x的ascii码
            # 将\n打印成: \\u0010, 否则会真的换行
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters

    # 有些字符不能被解码为utf8,使用"�"代替
    # errors='replace' to replace them with the replacement char �.
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        """
        [l][o] -> [lo] 491
        (97, 98 ) -> (491)
        """
        self.bigram_merge_table:Dict[(int, int), int] = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens:Dict[str,int] = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab:Dict[int, bytes] = self._build_vocab() # int -> bytes


    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab:Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)} # 32 -> b' ', 48 -> b'0', 65 -> b'A'
        for (p0, p1), idx in self.bigram_merge_table.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8") # 'endoftext'.encode('utf-8') = b"endoftext"
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n") # 特殊字符的长度
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # 注意：0`256的字符都没有存储
            # the merges dict
            for idx1, idx2 in self.bigram_merge_table:
                f.write(f"{idx1} {idx2}\n")

        # write the vocab: for the human to look at or debug
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.bigram_merge_table.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # 由两个子token组合
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file:str):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges:Dict[(int,int), int] = {}
        special_tokens:Dict[str,int] = {} # token ->  index
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1

        self.bigram_merge_table = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
