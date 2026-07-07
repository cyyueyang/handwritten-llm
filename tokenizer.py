import os
import json
import re
from typing import List, Dict, Tuple

class Tokenizer:
    def __init__(self, path):
        self.vocab = self.read_vocab(os.path.join(path, 'vocab.json'))
        self.merges = self.read_merges(os.path.join(path, 'merges.txt'))

        self.space_token = "Ġ"
        self.newline_token = "Ċ"
        self.vocab["<|im_start|>"] = 151644
        self.vocab["<|im_end|>"] = 151645
        self.id2token = {v: k for k, v in self.vocab.items()}

    def _tokenize(self, text) -> List[str]:
        text = text.replace(" ", self.space_token).replace("\n", self.newline_token)
        tokens = list(text)

        while True:
            # Find the highest-priority (earliest in merges) bigram currently present
            best_idx = -1
            for merge_idx, (first, second) in enumerate(self.merges):
                for i in range(len(tokens) - 1):
                    if tokens[i] == first and tokens[i + 1] == second:
                        best_idx = merge_idx
                        break
                if best_idx != -1:
                    break

            if best_idx == -1:
                break  # no merges applicable

            first, second = self.merges[best_idx]
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

    def tokenize(self, text) -> List[str]:
        tokens = []
        pattern = r"<\|\w+\|>"
        matches = re.finditer(pattern, text)

        t = 0
        for m in matches:
            match_text = m.group()
            pre_text = text[t: m.start()]
            if pre_text:
                tokens.extend(self._tokenize(pre_text))
            tokens.append(match_text)
            t = m.end()
        if t < len(text):
            tokens.extend(self._tokenize(text[t: ]))

        return tokens

    def encode(self, text) -> List[int]:
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

    def decode(self, token_ids) -> str:
        tokens = [self.id2token[token_id] for token_id in token_ids]
        text = "".join(tokens)
        text = text.replace(self.space_token, " ").replace(self.newline_token, "\n")
        return text

    @classmethod
    def read_vocab(cls, vocab_path) -> Dict[str, int]:
        with open(vocab_path, 'r', encoding="utf-8") as f:
            vocab = json.load(f)

        return vocab

    @classmethod
    def read_merges(cls, merges_path) -> List[Tuple[str, str]]:
        merges = []
        with open(merges_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) == 2:
                merges.append(line)

        return merges

if __name__ == "__main__":
    text = "Hello World!"
    tokenizer = Tokenizer(r"./")
    if tokenizer.decode(tokenizer.encode(text)) == text:
        print("test passed")

