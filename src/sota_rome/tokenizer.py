from .config import DEFAULT_TOKENIZER_CONFIG, TokenizerConfig


class Tokenizer:
    def __init__(self, config: TokenizerConfig = DEFAULT_TOKENIZER_CONFIG):
        self.config = config
        self.tok_to_idx = dict(zip(self.config.vocab, range(len(self.config.vocab))))
        self.idx_to_tok = {v:k for k,v in self.tok_to_idx.items()}

    def encode(self, text: str):
        input_ids = [self.tok_to_idx[c] for c in text] + [self.config.vocab.index("<endoftext>")]
        return input_ids

    def decode(self, idxs: list):
        # TODO: skip pad tokens
        return "".join([self.idx_to_tok[i] for i in idxs])
