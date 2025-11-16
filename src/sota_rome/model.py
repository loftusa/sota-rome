#%%
import torch
from torch import nn
import math
import torch.nn.functional as F
try:
    from .tokenizer import Tokenizer
    from .config import DEFAULT_MODEL_CONFIG
except ImportError:
    from pathlib import Path
    import sys
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "src"))
    from sota_rome.tokenizer import Tokenizer
    from sota_rome.config import DEFAULT_MODEL_CONFIG


hidden_dim = DEFAULT_MODEL_CONFIG.hidden_dim
n_heads = DEFAULT_MODEL_CONFIG.n_heads
sequence_len = DEFAULT_MODEL_CONFIG.sequence_len


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_up = nn.Linear(hidden_dim, hidden_dim*4, bias=False)
        self.activation = nn.SiLU()
        self.linear_down = nn.Linear(hidden_dim*4, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.linear_up(x)
        x = self.activation(x)
        x = self.linear_down(x)
        return x


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.QKV = nn.Linear(hidden_dim, hidden_dim*3, bias=False)
        self.O = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # TODO: attention masking
        mask = torch.triu(torch.full(size=(sequence_len, sequence_len), fill_value=float('-inf')), diagonal=1)
        self.register_buffer('mask', mask)
    
    def forward(self, x: torch.Tensor):
        """
        multi-head attention.
        - linearly transform x into keys k, queries q, and values v
        - implement qk^t/sqrt(d) * v
        """
        # x : (b, s, d)
        assert len(x.shape) == 3
        b, s, d = x.shape

        qkv = self.QKV(x)  # b, s, d*3
        q, k, v = torch.split(qkv, d, dim=2)

        # turn each into size b, h, s, d//h
        q = q.reshape(b, n_heads, s, d//n_heads)
        k = k.reshape(b, n_heads, s, d//n_heads)
        v = v.reshape(b, n_heads, s, d//n_heads)

        # self attention operation
        a = (q @ k.transpose(2, 3))/math.sqrt(d//n_heads)  # b, h, s, s
        a = a + self.mask[:s, :s]
        a = F.softmax(a, dim=-1)
        y = a @ v  # s,s * s,d//h -> b, h, s, d//h
        y = y.transpose(1,2).contiguous().view(b,s,d)
        return self.O(y)



#%%
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.attn = Attn()
        self.rmsnorm = nn.RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.rmsnorm(x))
        x = x + self.mlp(self.rmsnorm(x))
        return x


class LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = Tokenizer()
        self.embedder = nn.Embedding(self.tokenizer.config.vocab_size, hidden_dim)
        self.unembedder = nn.Linear(hidden_dim, self.tokenizer.config.vocab_size)
        self.transformer = nn.Sequential(
                *[Block() for _ in range(4)],
        )

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len)
        x = self.embedder(x)
        x = self.transformer(x)
        x = self.unembedder(x)
        return x

    def predict(self, text):
        idxs = torch.tensor(self.tokenizer.encode(text), dtype=int).unsqueeze(0)  # (B, S)
        print(idxs.shape)
        logits = self.forward(idxs)  # (B, S, V)
        return logits[:, -1, :]



llm = LLM()
logits = llm.predict("Hello worl")
print(logits.shape)
print(logits)

