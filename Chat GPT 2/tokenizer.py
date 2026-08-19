import tiktoken


def get_gpt2_tokenizer():
    return tiktoken.get_encoding("gpt2")
