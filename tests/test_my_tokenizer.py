from typing import Callable

import pytest
import tiktoken
import os

from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

# -----------------------------------------------------------------------------
# common test data

# a few strings to test the tokenizers on
test_strings = [
    "", # empty string
    "?", # single character
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
]
def my_decorator(func:Callable):
    def my_wrapper(*args, **kwargs):
        #start_time = time.time()
        print("="*30+f" {func.__name__} begin "+"="*30)
        result = func(*args, **kwargs)
        print(f"{func.__name__} end\n\n")
        #end_time = time.time()
        return result
    return my_wrapper

@my_decorator
def test_tokenizer():
    tokenizer = BasicTokenizer()
    text = "aaabdaaabac"
    tokenizer.train(text, vocab_size=256 + 3, verbose=True)  # 256 are the byte tokens, then do 3 merges
    print("merges:")
    print(tokenizer.bigram_merge_table)
    ids = tokenizer.encode(text)
    print("ids")
    print(ids)

    print("id to token:")
    print([str(id)+":"+tokenizer.decode([id]) for id in ids])

    # [258, 100, 258, 97, 99]
    assert ids == [258, 100, 258, 97, 99]
    print(tokenizer.decode(ids))
    # aaabdaaabac
    assert (tokenizer.decode(ids)==text)

    os.makedirs("../models", exist_ok=True)
    model_name = os.path.join("../models", "toy_abc")
    tokenizer.save(model_name)

if __name__ == "__main__":
    test_tokenizer()