from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    vocab: list = field(default_factory=lambda: list(
        r" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890!@#$%^&*()-_=+[]{}\|;:'\"\,./<>?`~") + ["<pad>", "<endoftext>"]
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab.index("<pad>")


@dataclass
class ModelConfig:
    hidden_dim: int = 768
    n_heads: int = 8
    sequence_len: int = 1024


DEFAULT_TOKENIZER_CONFIG = TokenizerConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()