# ♟️ woodpusher

Woodpusher is a language model that learns chess purely by reading text-encoded games, built from
scratch as a hands-on way to work through the full foundation-model lifecycle, albeit at hobbyist scale.
Chess is usually considered solved by state-of-the-art RL engines like
[AlphaZero](https://arxiv.org/abs/1712.01815), but treating it as a language-modelling problem
opens up different lines of inquiry into what a transformer picks up from next-move prediction alone.
Because a capable model at this scale stays small enough to train on a single GPU, it presents a
convenient toy system for probing interpretability and scaling laws against a problem space with clean ground truths.
Longer-term directions include blending imitation pretraining with RL post-training.

The name is an homage to John Hamlen's
**[Woodpusher (1989)](https://www.chessprogramming.org/Woodpusher)**, a sub-64K experimental chess program born as a
university research project, and which campaigned at World Computer Chess Championships for two decades.

## ♜ Design decisions

| Decision | Choice |
|---|---|
| Tokenizer | Move-level UCI (one token per move, ~4.2k vocab, generated in code) |
| Conditioning | Elo-bucket prefix tokens `<welo:B> <belo:B>` (a playing-strength dial) |
| Architecture | Decoder-only transformer: RoPE, RMSNorm, SwiGLU (swappable, see `model.py`) |
| Training | Hand-rolled single-GPU PyTorch loop, bf16, AdamW, warmup+cosine |
| Tracking | CSV logs + matplotlib (`woodpusher/plot.py`) |
| Data | [Lichess open database](https://database.lichess.org) monthly dumps |

## ♖ Architecture

woodpusher is a decoder-only autoregressive transformer in the modern Llama-style
idiom, kept to a single file (`model.py`). The rest of the repo depends on it
through one contract only: **token ids in, next-token logits out.** That interface
is what keeps the sequence-mixing core swappable.

Each layer runs on a pre-norm residual stream:

1. RMSNorm, then causal multi-head self-attention with rotary position embeddings
   (RoPE), added back to the stream.
2. RMSNorm, then a SwiGLU MLP (hidden width about 8/3 of the model width, rounded
   to a multiple of 64), added back.

Other specifics: attention is PyTorch `scaled_dot_product_attention` (flash when
available) with a causal mask and no biases; the token embedding is tied to the
output projection; dropout is off by default; residual-output projections use the
GPT-2 scaled init. Depth, width, and head count come from the ladder preset in
`configs.py`; the vocabulary (~4.2k) is fixed by the tokenizer.

**Held fixed (the controlled spine).** Constant across runs so that ladder and
conditioning comparisons stay clean:

- Move-level UCI tokenization (one move is one token, ~4.2k deterministic vocab)
- Elo-bucket prefix conditioning (`<welo:B> <belo:B>`)
- Decoder-only next-token objective, trained single-GPU with AdamW, bf16, and a warmup-then-cosine schedule
- The eval harness (`illegal`, `mate_in_one`, `probes`, `elo_vs_stockfish`)

**Open to experiment.** Knobs we expect to turn; the ids-to-logits interface keeps
each isolated from the rest of the code:

- Sequence-mixing block: a transformer today, but Mamba/SSM, RWKV, or MoE can drop in behind the same interface for architecture comparisons
- Attention variant: multi-head now, group-query (GQA) later to cut KV-cache cost
- Conditioning mechanism: discrete tokens vs continuous embeddings vs latent (no conditioning)
- Tokenization granularity: move-level UCI vs character-level SAN
- Scale: the smoke / 5m / 25m / 100m ladder
- Post-training: RL on top of the imitation base

## ♞ Quickstart

```powershell
uv sync

# 1. data: 2013-01 is tiny (~17 MB, ~120k games), perfect for smoke tests
uv run python -m woodpusher.data.download --month 2013-01
uv run python -m woodpusher.data.prepare --pgn data/raw/lichess_db_standard_rated_2013-01.pgn.zst --out data/smoke

# 2. smoke-test the loop
uv run python -m woodpusher.train --preset smoke --data-dir data/smoke

# 3. evals + play
uv run python -m woodpusher.evals.illegal --ckpt runs/smoke/ckpt.pt --data-dir data/smoke
uv run python -m woodpusher.evals.mate_in_one --ckpt runs/smoke/ckpt.pt --data-dir data/smoke
uv run python -m woodpusher.evals.probes --ckpt runs/smoke/best.pt
uv run python -m woodpusher.play --ckpt runs/smoke/ckpt.pt --selfplay
uv run python -m woodpusher.play --ckpt runs/smoke/ckpt.pt --color white --model-elo 1500

# 4. plots
uv run python -m woodpusher.plot

# 5. web UI: play any checkpoint and watch its per-move candidate distribution
uv run python -m woodpusher.web   # http://localhost:8000
```

The Elo-vs-Stockfish eval needs a [Stockfish binary](https://stockfishchess.org/download/):

```powershell
uv run python -m woodpusher.evals.elo_vs_stockfish --ckpt runs/25m/best.pt --stockfish path\to\stockfish.exe --engine-elo 1400
```

## ♛ Project plan

1. [x] Data pipeline + tokenizer
2. [ ] 5M smoke-test model (`--preset 5m`, needs ~120M tokens from the shared corpus)
3. [ ] Eval harness validated on the 5M model; play against it
4. [ ] Ladder runs + scaling plot (`5m` to `25m`, rematch at each rung)
5. [ ] 100M Chinchilla run (~1.8B tokens, ~a day on one GPU)
6. [ ] Conditioning experiments: does `<welo:2400>` beat `<welo:1200>`?

## ♚ Scale & budget notes

| Preset | Params | Chinchilla tokens | Hardware |
|---|---|---|---|
| smoke | ~1M | (3M, not Chinchilla) | local (RTX 4070) |
| 5m | ~6M | 120M | local (RTX 4070, ~minutes) |
| 25m | ~28M | 560M | local (RTX 4070, ~hours) |
| 100m | ~90M | 1.8B | local overnight, or rented A100 for speed (~$30–60) |

The 5m/25m/100m rungs all read **one shared corpus**, prepared once from a single
recent Lichess month (BOT-filtered, with a held-out val/test split). Each rung
draws its Chinchilla token budget from the same pool via the trainer's step count,
so scaling comparisons hold the data distribution fixed and the held-out set is
common across rungs. `smoke` uses the tiny 2013-01 dump for quick pipeline checks.

Rules of thumb: training compute ≈ 6·params·tokens FLOPs; compute-optimal runtime
scales with params² (each rung is roughly 4× the last); a game averages ~71 tokens
(measured on recent data). A recent month is ~90–100M games; filtered to blitz+rapid
it yields ~4B tokens, comfortably above the 100m rung's 1.8B, so one month covers the
whole ladder single-epoch and `prepare.py --max-games` caps it.

## ♝ Evals

- **illegal**: raw top-1 legality on held-out positions, the hallucination metric
- **mate_in_one**: tactics accuracy vs a random-legal baseline, no engine needed
- **probes**: fixed-position belief check, reporting top-k candidates and legal mass at telltale positions
- **elo_vs_stockfish**: match play vs strength-limited Stockfish for an Elo estimate
- **play --selfplay**: eyeball test; prints a full game and raw-illegal count

## ♟ Prior work & reading

woodpusher sits in a small but active line of work on learning chess from
games rather than search:

- **[Chess-GPT (Karvonen, 2024)](https://arxiv.org/abs/2403.15498)**: the closest prior art. A GPT trained on Lichess move text reaches ~1500 Elo, with board state and player skill linearly recoverable from its activations.
- **[Grandmaster-Level Chess Without Search (Ruoss et al., 2024)](https://arxiv.org/abs/2402.04494)**: a 270M transformer distilled from Stockfish annotations plays 2895 Lichess blitz with no search at all.
- **[Maia (McIlroy-Young et al., 2020)](https://maiachess.com)**: human-like move prediction targeted at specific rating bands, deployed as real Lichess bots.
- **[Chessformer (Monroe et al., 2026)](https://arxiv.org/abs/2605.19091)**: a recent preprint from the Maia lab (CSSLab), pairing an encoder-only board model with continuous soft rating embeddings; its Maia-3 family reaches 57.1% human move-match. The clearest current example of conditioning on skill through learned embeddings rather than input tokens.
- **[Transcendence (Zhang et al., 2024)](https://arxiv.org/abs/2406.11741)**: low-temperature sampling lets a game-trained model outplay every human in its own training data.
