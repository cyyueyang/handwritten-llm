import os
import json
import re

def read_merges(path):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    merges = []
    for line in lines:
        line = line.strip().split()
        if len(line) == 2:
            merges.append(line)

    return merges

def read_vocab(path):
    with open(path, 'r', encoding="utf-8") as f:
        vocab = json.load(f)

    return vocab


class Tokenizer:
    def __init__(self, merges_path, vocab_path):
        self.merges = read_merges(merges_path)
        self.vocab = read_vocab(vocab_path)

        self.space_token = 'Ġ'
        self.newline_token = 'Ċ'
        self.vocab['<|im_start|>'] = 151644
        self.vocab['<|im_end|>'] = 151645

        self.id2token = {v: k for k, v in self.vocab.items()}

    def _tokenize(self, text):
        text = text.replace(' ', self.space_token).replace('\n', self.newline_token)
        tokens = list(text)

        for pair in self.merges:
            first, second = pair
            new_token = first + second
            new_tokens = []

            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def tokenize(self, text):
        tokens = []
        pattern = r"<\|w+\|>"

        t = 0
        for m in re.finditer(pattern, text):
            match_text = m.group()
            pre_text = text[t: m.start()]
            if pre_text:
                tokens.extend(self._tokenize(pre_text))
            tokens.append(match_text)
            t = m.end()
        if t < len(text):
            tokens.extend(self._tokenize(text[t: ]))

        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

    def decode(self, token_ids):
        tokens = [self.id2token.get(id, '<unk>') for id in token_ids]
        text = "".join(tokens)
        text = text.replace(self.space_token, " ").replace(self.newline_token, "\n")

        return text



