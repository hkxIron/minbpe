"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.

BPE算法：The BPE algorithm is "byte-level" because it runs on UTF-8 encoded strings
"""
import unicodedata
from typing import List,Optional,Dict,Tuple, Any
from itertools import pairwise


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
    #for pair in zip(ids, ids[1:]): # iterate consecutive elements, pairwise(ids)
    for pair in pairwise(ids):  # iterate consecutive elements, pairwise(ids)
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
>>> list("你好".encode('utf8'))
[228, 189, 160, 229, 165, 189]

ids_in_chunk_list: 每个chunk代表一段文本,里面包含很多id list, 每个id list都是一段话的utf8的id
"""
def bigram_merge(ids_in_chunk_list:List[List[int]], num_merges:int, verbose=False):
    # iteratively merge the most common pairs to create new tokens
    bigram_merge_table:Dict[(int, int), int] = {} # (int, int) -> int, 指示哪两个id pair被merge了,生成一个new_id
    # id转bytes
    # 先将0～255的所有id转二进制bytes
    # 48 -> b'0',前者为数字48的signed int32的4个字节的二进制表示(48是int, bin(48)='0b110000'),
    # 后者为字符‘0’的ascii码(1一个字节)对应的二进制bytes表示
    vocab:Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
    for i in range(num_merges):
        # 注意：每次merge中,bigram_to_count需要重新统计
        # count the number of times every consecutive pair appears
        # count up the number of times every consecutive bigram appears
        # Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        bigram_to_count:Dict[(int,int), int] = {}
        for ids in ids_in_chunk_list:
            # passing in stats will update it in place, adding up counts
            # Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
            get_bigram_stats(ids, bigram_to_count) # inplace，统计结果放在bigram_to_count中

        # find the pair with the highest count, 统计出现次数最多的bigram
        highest_bigram:Tuple[int, int] = max(bigram_to_count, key=bigram_to_count.get)

        # mint a new token: assign it the next available id
        new_id = 256 + i

        """
        将所有的bigram替换为new_id
        """
        # replace all occurrences of bigram in ids with new_id
        # Example: ids = [1, 2, 3, 1, 2], bigram = (1, 2), new_id = 4
        # -> [4, 3, 4]
        # replace all occurrences of pair in ids with new_id
        ids_in_chunk_list = [replace_bigram_by_id(chunk_ids, highest_bigram, new_id) for chunk_ids in ids_in_chunk_list]
        # save the merge
        bigram_merge_table[highest_bigram] = new_id
        # 直接将bytes进行相加,在字符串级别为concat,即 b'e'+b'n' = b'en',即认为'en'经常在一起，可以合并为一个新token
        vocab[new_id] = vocab[highest_bigram[0]] + vocab[highest_bigram[1]]
        # prints
        if verbose:
            print(f"merge {i+1}/{num_merges}: {highest_bigram} -> {new_id}, ({vocab[highest_bigram[0]]},{vocab[highest_bigram[1]]}) ->{vocab[new_id]} for {vocab[new_id]} had {bigram_to_count[highest_bigram]} occurrences")
    return bigram_merge_table, vocab

def merge_bigram_by_table(ids:List[int], bigram_merge_table:Dict[Tuple[int, int],int])->List[int]:
    while len(ids) >= 2:
        # 每次都要重新统计，寻找count最小的bigram
        # find the bigram with the lowest merge index
        bigram_to_count:Dict[(int, int), int] = get_bigram_stats(ids)

        """
        从ids中获取这样的pair,该pair在merge_table中具有最小的idx, 因为最小的idx是train时在语料中出现最频繁的bigram, 也即是最早合并的bigram
        """
        bigram_of_min_count:Tuple[(int,int)] = min(bigram_to_count, key=lambda pair: bigram_merge_table.get(pair, float("inf")))

        # subtle: if there are no more bigram_merge_table available, the key will
        # result in an inf for every single bigram, and the min will be
        # just the first bigram in the list, arbitrarily
        # we can detect this terminating case by a membership check

        # 直到所有的bigram不能再合并就停止encode
        if bigram_of_min_count not in bigram_merge_table:
            break # nothing else can be merged anymore

        # 在merge_table中找到bigram对应的idx, 然后根据idx进行合并
        # otherwise let's merge the best bigram (lowest merge index)
        idx = bigram_merge_table[bigram_of_min_count]
        ids = replace_bigram_by_id(ids, bigram_of_min_count, idx)
    return ids

# first two helper functions...
def replace_control_characters(string: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse), 比如换行符直接打印时会真正换行
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in string:
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
        # default: vocab size of 256 (all bytes), no bigram_merge_table, no patterns
        """
        [l][o] -> [lo] 491
        (97, 98 ) -> (491)
        """
        self.bigram_merge_table:Dict[(int, int), int] = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens:Dict[str,int] = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab:Dict[int, bytes] = self._build_vocab() # int -> bytes

    def __len__(self):
        return len(self.vocab) + len(self.special_tokens)

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
        # vocab is simply and deterministically derived from bigram_merge_table
        vocab:Dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)} # 32 -> b' ', 48 -> b'0', 65 -> b'A'
        for (p0, p1), idx in self.bigram_merge_table.items():
            vocab[idx] = vocab[p0] + vocab[p1] # 两个bytes直接join在一起
        for special, idx in self.special_tokens.items(): # 特殊token
            vocab[idx] = special.encode("utf-8") # 'endoftext'.encode('utf-8') = b"endoftext"
        return vocab

    def save(self, file_prefix:str):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        # .model是真正用于加载时使用的文件
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and bigram_merge_table, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n") # 特殊字符的长度
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # 注意：0`256的字符都没有存储
            # the bigram_merge_table dict
            for idx1, idx2 in self.bigram_merge_table:
                f.write(f"{idx1} {idx2}\n") # 存的是bigram token_id, eg: 12 34

        # write the vocab: for the human to look at or debug
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: bigram_id for bigram_id, idx in self.bigram_merge_table.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token_bytes in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                readable_str = render_token(token_bytes)
                # find the children of this token_bytes, if any
                if idx in inverted_merges:
                    # 由两个子token组合而来
                    # if this token_bytes has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    readable_str0 = render_token(self.vocab[idx0])
                    readable_str1 = render_token(self.vocab[idx1])
                    f.write(f"[{readable_str0}][{readable_str1}] -> [{readable_str}] {idx}\n")
                else:
                    # otherwise this is leaf token_bytes, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{readable_str}] {idx}\n")

    def load(self, model_file:str):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges:Dict[(int,int), int] = {}
        special_tokens:Dict[str,int] = {} # token ->  index
        index = 256
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
            # read the bigram_merge_table
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = index
                index += 1

        self.bigram_merge_table = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
