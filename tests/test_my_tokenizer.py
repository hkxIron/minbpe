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
    tokenizer.train(text, vocab_size=256 + 3, verbose=True)  # 256 are the byte tokens, then do 3 bigram_merge_table
    print("bigram_merge_table:")
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
    tokenizer.train(text, vocab_size=256 + 30, verbose=True)  # 256 are the byte tokens, then do 3 bigram_merge_table
    print("bigram_merge_table:")
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

    tokenizer.train(text, vocab_size=256 + 5, verbose=True)  # 256 are the byte tokens, then do 3 bigram_merge_table
    print("bigram_merge_table:")
    print(tokenizer.bigram_merge_table)

    text="Taylor Alison Swift (born December 13, 1989) is an American singer-songwriter. Her versatile artistry"
    ids = tokenizer.encode(text)
    print(ids)
    expect_ids = [84, 97, 121, 108, 258, 32, 65, 108, 105, 115, 111, 110, 32, 83, 119, 105, 102, 116, 32, 40, 98, 258, 110, 32, 68, 101, 99, 101, 109, 98, 256, 32, 49, 51, 44, 32, 49, 57, 56, 57, 41, 32, 105, 115, 32, 97, 110, 32, 65, 109, 256, 105, 99, 97, 110, 32, 115, 259, 103, 256, 45, 115, 111, 110, 103, 119, 114, 105, 116, 256, 46, 32, 72, 256, 32, 118, 256, 115, 97, 116, 105, 108, 101, 32, 97, 114, 116, 105, 115, 116, 114, 121]
    assert ids == expect_ids
    assert (tokenizer.decode(ids)==text)

@my_decorator_gen(need_test=True)
def test_regex_with_special():
    specials_string = """
    <|endoftext|>Hello world this is one document
    <|endoftext|>And this is another document
    <|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
    <|endoftext|>Last document!!! 👋<|endofprompt|>
    """.strip()
    special_tokens = {
        '<|endoftext|>': 100257,
        '<|fim_prefix|>': 100258,
        '<|fim_middle|>': 100259,
        '<|fim_suffix|>': 100260,
        '<|endofprompt|>': 100276
    }
    llama_text = """
    <|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
    Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
    The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
    <|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
    """.strip()

    tokenizer = RegexTokenizer()
    print("vocab size:", len(tokenizer))
    tokenizer.register_special_tokens(special_tokens)
    print("vocab size:", len(tokenizer))

    tokenizer.train(llama_text, vocab_size=256 + 100, verbose=True)  # 256 are the byte tokens, then do 3 bigram_merge_table
    print("bigram_merge_table:")
    print(tokenizer.bigram_merge_table)

    print("ids:")
    ids = tokenizer.encode(specials_string, allowed_special='all')
    print(ids)

    """
    注意，控制字符都会打印为:'�', 而所有与控制字符组成的bigram也因此都会打印为'�' 
    '33:!', '32: ', '240:�', '159:�', '145:�', '139:�', '100276:<|endofprompt|>'
    """
    print("id to token:")
    print([str(id)+":"+tokenizer.decode([id]) for id in ids])

    assert (tokenizer.decode(ids)==specials_string)

    text="<|endoftext|> and my teacher <|endofprompt|> endoftext| run.你好"
    ids = tokenizer.encode(text, allowed_special='all')
    print(ids)
    print(tokenizer.decode(ids))

@my_decorator_gen(need_test=True)
def test_gpt4_tokenizer():
    specials_string = """
    <|endoftext|>Hello world this is one document
    <|endoftext|>And this is another document
    <|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
    <|endoftext|>Last document!!! 👋<|endofprompt|>
    """.strip()
    special_tokens = {
        '<|endoftext|>': 100257,
        '<|fim_prefix|>': 100258,
        '<|fim_middle|>': 100259,
        '<|fim_suffix|>': 100260,
        '<|endofprompt|>': 100276
    }
    tokenizer = GPT4Tokenizer()
    gpt4_tokenizer_ids = tokenizer.encode(specials_string, allowed_special="all")
    #tokenizer.save_vocab("../models/gpt4.vocab")
    print(gpt4_tokenizer_ids)

@my_decorator_gen(need_test=True)
def test_regex_tokenizer_zh():
    tokenizer = RegexTokenizer()

    text = """由于《西游记》剧情走向对大部分中国人来说耳熟能详，冯骥与扬奇等人就剧本进行多次迭代，最终决定以“寻根之旅”为核心进行剧情展开。游戏主角“天命人”一路上会遇到很多《西游记》里出现过的著名角色，通过与他们战斗，或是成为伙伴，玩家再去尝试搞清楚“悟空是谁”以及“我是谁” [43]。
在设计游戏环境过程中，由于自然景观缺乏现实的中式素材，开发团队与各地文保部门合作前往实地考察，对陵川二仙庙，晋城青莲寺等现实古建筑和塑像进行扫描，以此为蓝本进行重视建筑的设计 [45]。为了让场景更为逼真，采用虚幻5引擎和NVIDIA光线追踪等技术以提升画面效果"""
    tokenizer.train(text, vocab_size=256 + 50, verbose=True)  # 256 are the byte tokens, then do 3 bigram_merge_table

    print("bigram_merge_table:")
    print(tokenizer.bigram_merge_table)
    print("vocab:")
    print(tokenizer.vocab)
    # 一般中文utf8 占3个bytes
    # 可以看到有部分中文字编码为一个新token
    print("vocab decode:")
    for index, vocab_bytes in tokenizer.vocab.items():
        print(f"{index} bytes:{vocab_bytes} char:{vocab_bytes.decode('utf-8', errors='replace')}")
    """
    结果如下：
    277 bytes:b'\xe8\xbf\x9b' char:进
    278 bytes:b'\xe8\xbf\x9b\xe8' char:进�
    279 bytes:b'\xe8\xbf\x9b\xe8\xa1' char:进�
    280 bytes:b'\xe8\xbf\x9b\xe8\xa1\x8c' char:进行
    281 bytes:b'\xe4\xbb\xa5' char:以
    282 bytes:b'\xe2\x80\x9c' char:“
    """

    text="剧情走向对大部分中国人来说耳熟能详"
    ids = tokenizer.encode(text)
    print(f"ids:{ids}")

if __name__ == "__main__":
    print(dir())
    if False:
        test_gpt4_tokenizer()
        test_tokenizer2()
        test_regex_tokenizer()
        unicode_test()
        test_tokenizer()
        test_regex_with_special()
    test_regex_tokenizer_zh()