# woodpusher

A language model that learns chess purely by reading games — built from
scratch to walk the **full foundation-model lifecycle at hobby scale**: data
pipeline, tokenizer, pretraining, evals, scaling ladder, and play.

The name is chess slang for an enthusiastic-but-weak amateur. It is also a
deliberate homage: this project is the spiritual successor to John Hamlen's
**Woodpusher (1989)** — a sub-64K experimental chess program born as a
university research project, campaigned at World Computer Chess Championships
for two decades, and retired in the 2012 paper *"Game Over for the Woodpusher
Experiment."* Same self-deprecating name, same scrappy-experiment spirit,
new paradigm: where the original searched, this one predicts.

## Design decisions

| Decision | Choice |
|---|---|
| Tokenizer | Move-level UCI (one token per move, ~4.2k vocab, generated in code) |
| Conditioning | Elo-bucket prefix tokens `<welo:B> <belo:B>` — a playing-strength dial |
| Architecture | Decoder-only transformer: RoPE, RMSNorm, SwiGLU (swappable — see `model.py`) |
| Training | Hand-rolled single-GPU PyTorch loop, bf16, AdamW, warmup+cosine |
| Tracking | CSV logs + matplotlib (`woodpusher/plot.py`) |
| Data | [Lichess open database](https://database.lichess.org) monthly dumps |

## Quickstart

```powershell
uv sync

# 1. data: 2013-01 is tiny (~17 MB, ~120k games) — perfect for smoke tests
uv run python -m woodpusher.data.download --month 2013-01
uv run python -m woodpusher.data.prepare --pgn data/raw/lichess_db_standard_rated_2013-01.pgn.zst --out data/smoke

# 2. smoke-test the loop
uv run python -m woodpusher.train --preset smoke --data-dir data/smoke

# 3. evals + play
uv run python -m woodpusher.evals.illegal --ckpt runs/smoke/ckpt.pt --data-dir data/smoke
uv run python -m woodpusher.evals.mate_in_one --ckpt runs/smoke/ckpt.pt --data-dir data/smoke
uv run python -m woodpusher.play --ckpt runs/smoke/ckpt.pt --selfplay
uv run python -m woodpusher.play --ckpt runs/smoke/ckpt.pt --color white --model-elo 1500

# 4. plots
uv run python -m woodpusher.plot

# 5. web UI: live training chart + play any checkpoint (hot-reloads mid-run)
uv run python -m woodpusher.web   # http://localhost:8000
```

The Elo-vs-Stockfish eval needs a [Stockfish binary](https://stockfishchess.org/download/):

```powershell
uv run python -m woodpusher.evals.elo_vs_stockfish --ckpt runs/25m/best.pt --stockfish path\to\stockfish.exe --engine-elo 1400
```

## The plan

1. [x] Data pipeline + tokenizer
2. [ ] 5M smoke-test model (`--preset 5m`, needs ~120M tokens — a 2015-era month)
3. [ ] Eval harness validated on the 5M model; play against it
4. [ ] Ladder runs + scaling plot (`5m` → `25m`, rematch at each rung)
5. [ ] 100M Chinchilla run (~1.8B tokens, rented 1×A100/H100, ~a day, ~$30–60)
6. [ ] Conditioning experiments: does `<welo:2400>` beat `<welo:1200>`?

## Scale & budget notes

| Preset | Params | Chinchilla tokens | Data needed | Hardware |
|---|---|---|---|---|
| smoke | ~1M | (3M, not Chinchilla) | 2013-01 dump | this laptop |
| 5m | ~6M | 120M | ~1 month of 2015 | this laptop (RTX 4070, hours) |
| 25m | ~28M | 560M | ~1 month of 2016-17 | this laptop (overnight) |
| 100m | ~90M | 1.8B | ~1 month of 2018+ | rented A100/H100, ~$30–60 |

Rules of thumb: training compute ≈ 6·params·tokens FLOPs; a game averages
~70 tokens; one recent monthly dump ≈ 8–10B tokens (far more than any rung
here needs — `prepare.py --max-games` caps it).

## Evals

- **illegal** — raw top-1 legality on held-out positions: the hallucination metric
- **mate_in_one** — tactics accuracy vs a random-legal baseline, no engine needed
- **elo_vs_stockfish** — match play vs strength-limited Stockfish → Elo estimate
- **play --selfplay** — eyeball test; prints a full game + raw-illegal count
