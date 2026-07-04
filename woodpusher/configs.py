"""Training presets — the scaling ladder.

target_tokens follows Chinchilla (~20x params). batch_size is the
micro-batch (sequences); effective tokens/step = batch * block * grad_accum.
Micro-batches are sized to fit an 8 GB card for smoke/5m/25m; the 100m rung
is meant for a rented GPU.
"""

from dataclasses import dataclass


@dataclass
class Preset:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    batch_size: int
    grad_accum: int
    lr: float
    target_tokens: int


PRESETS = {
    "smoke": Preset(2, 4, 128, 256, 32, 1, 1e-3, 3_000_000),
    "5m": Preset(6, 8, 256, 512, 32, 2, 1e-3, 120_000_000),
    "25m": Preset(8, 8, 512, 512, 16, 4, 6e-4, 560_000_000),
    "100m": Preset(12, 12, 768, 512, 16, 8, 3e-4, 1_800_000_000),
}
