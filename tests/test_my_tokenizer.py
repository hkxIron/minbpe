from typing import Callable

import pytest
import tiktoken
import os

from minbpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer


def my_decorator_gen(need_test:bool=True):
    def my_decorator(func:Callable):
        def my_wrapper(*args, **kwargs):
            #start_time = time.time()
            print("="*30+f" {func.__name__} begin "+"="*30)
            if need_test:
                result = func(*args, **kwargs)
                #end_time = time.time()
            else:
                print(f"{func.__name__} no need to test, skip!")
                result = None
            print(f"{func.__name__} end\n\n\n")
        return my_wrapper
    return my_decorator

# -----------------------------------------------------------------------------
# common test data

# a few strings to test the tokenizers on
test_strings = [
    "", # empty string
    "?", # single character
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
]

def unicode_test():
    print(type(ord('中')))
    print("‘中’的unicode code point:", ord('中'))

    print(dir())
    print(os.path.abspath(__file__))

@my_decorator_gen(need_test=True)
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

@my_decorator_gen(need_test=True)
def test_tokenizer2():
    tokenizer = BasicTokenizer()
    taylorswift_file = "taylorswift.txt"
    text = open(taylorswift_file, "r", encoding="utf-8").read()
    tokenizer.train(text, vocab_size=256 + 30, verbose=True)  # 256 are the byte tokens, then do 3 merges
    print("merges:")
    print(tokenizer.bigram_merge_table)

    text="Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. Her versatile artistry"
    ids = tokenizer.encode(text)
    print("ids:")
    print(ids)

    print("id to token:")
    print([str(id)+":"+tokenizer.decode([id]) for id in ids])

    # [258, 100, 258, 97, 99]
    expect_ids = [84, 97, 121, 108, 283, 65, 108, 105, 115, 279, 282, 266, 40, 98, 111, 114, 110, 32, 68, 101, 99, 101, 109, 98, 272, 49, 51, 257, 49, 57, 56, 57, 41, 32, 105, 262, 270, 32, 65, 109, 101, 265, 99, 270, 32, 115, 263, 103, 278, 45, 115, 264, 103, 119, 265, 116, 278, 259, 72, 272, 118, 278, 115, 97, 116, 105, 108, 256, 271, 116, 105, 115, 116, 114, 121]
    assert ids == expect_ids
    print(tokenizer.decode(ids))
    # aaabdaaabac
    assert (tokenizer.decode(ids)==text)

@my_decorator_gen(need_test=True)
def test_regex_tokenizer():
    tokenizer = RegexTokenizer()
    taylorswift_file = "taylorswift.txt"
    text = open(taylorswift_file, "r", encoding="utf-8").read()

    tokenizer.train(text, vocab_size=256 + 5, verbose=True)  # 256 are the byte tokens, then do 3 merges
    print("merges:")
    print(tokenizer.bigram_merge_table)

    text="Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. Her versatile artistry"
    ids = tokenizer.encode(text)
    print(ids)
    expect_ids = [84, 97, 121, 108, 258, 32, 65, 108, 105, 115, 111, 110, 32, 83, 119, 105, 102, 116, 32, 40, 98, 258, 110, 32, 68, 101, 99, 101, 109, 98, 256, 32, 49, 51, 44, 32, 49, 57, 56, 57, 41, 32, 105, 115, 32, 97, 110, 32, 65, 109, 256, 105, 99, 97, 110, 32, 115, 259, 103, 256, 45, 115, 111, 110, 103, 119, 114, 105, 116, 256, 46, 32, 72, 256, 32, 118, 256, 115, 97, 116, 105, 108, 101, 32, 97, 114, 116, 105, 115, 116, 114, 121]
    assert ids == expect_ids
    assert (tokenizer.decode(ids)==text)


    #ids = tokenizer.encode(text)
    #print("ids")
    #print(ids)

    #print("id to token:")
    #print([str(id)+":"+tokenizer.decode([id]) for id in ids])

if __name__ == "__main__":
    print(dir())
    test_tokenizer2()
    test_regex_tokenizer()
    unicode_test()
    test_tokenizer()