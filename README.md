Inspired by Karpathy.

A personal learning repo to hack on an LLM for fun that I built myself and implemented myself and didn't use claude for

gives me a codebase I know intimately to implement new research and see if the ideas are any good

Vague plans to copy a sota MoE architecture (e.g., kimi-k2, kwen, etc)

Current TODOs:
- variable-length attention masking / padding
  - pad shorter sequences with pad id (during data loading)
  - move attention padding to data loading. padding mask as an argument in the forward pass.
  - dont waste flops on pad tokens
- rope 
- kv cache

Long-term plans (may change):
- add dataloading
- add a training loop (use deepspeed?)
- pretraining with fineweb parquet files
    - maybe eventually just put kimi's weights in
- posttraining on info about me
- Host on my website as an interactive resume