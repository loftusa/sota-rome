
from __future__ import annotations

from typing import cast

import torch
from jaxtyping import Float, Int

from sota_rome.config import DEFAULT_MODEL_CONFIG
from sota_rome.model import Attn, Block, LLM


def _random_hidden(batch: int, seq: int, *, seed: int = 0):
    """
    Deterministic helper for generating hidden states used in tests.
    """
    torch.manual_seed(seed)
    hidden_dim = DEFAULT_MODEL_CONFIG.hidden_dim
    hidden = torch.randn(batch, seq, hidden_dim, dtype=torch.float32)
    assert hidden.shape == (batch, seq, hidden_dim)
    return hidden


def test_attn_forward_shape():
    attn = Attn()
    batch, seq = 2, 5
    hidden = _random_hidden(batch, seq)
    output = attn(hidden)
    assert output.shape == (batch, seq, DEFAULT_MODEL_CONFIG.hidden_dim)
    assert torch.isfinite(output).all()


def test_block_forward_shape():
    block = Block()
    batch, seq = 2, 6
    hidden = _random_hidden(batch, seq, seed=1)
    output = block(hidden)
    assert output.shape == (batch, seq, DEFAULT_MODEL_CONFIG.hidden_dim)
    assert torch.isfinite(output).all()


def test_llm_forward_logits_shape():
    llm = LLM()
    batch, seq = 2, 7
    vocab_size = llm.tokenizer.config.vocab_size
    torch.manual_seed(2)
    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch, seq),
        dtype=torch.long,
    )
    logits = llm(tokens)
    assert logits.shape == (batch, seq, vocab_size)
    assert torch.isfinite(logits).all()
