"""Illegal-move rate: the model's hallucination metric.

Feeds random game prefixes from the val set and checks whether the model's
raw (unmasked) top-1 token is a legal move in the position.

    uv run python -m woodpusher.evals.illegal --ckpt runs/smoke/ckpt.pt --data-dir data/smoke
"""

import argparse
import random

import chess
import torch

from ..tokenizer import Tokenizer
from .common import game_moves, load_model, next_logits, sample_games


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--positions", type=int, default=1000)
    ap.add_argument("--split", default="test", help="held-out split to sample (test/val)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = Tokenizer()
    model, _ = load_model(args.ckpt, args.device)
    games = sample_games(args.data_dir, args.positions, args.seed, args.split)
    rng = random.Random(args.seed)

    n = well_formed = legal = 0
    for ids in games:
        if n >= args.positions:
            break
        moves = game_moves(tok, ids)
        if not moves:
            continue
        cut = rng.randrange(len(moves))
        board = chess.Board()
        for m in moves[:cut]:
            board.push(chess.Move.from_uci(m))

        context = ids[: 3 + cut]  # <bos> <welo> <belo> + moves so far
        raw_id = int(next_logits(model, context, args.device).argmax())
        n += 1
        if not tok.is_move_id(raw_id):
            continue
        well_formed += 1
        try:
            if chess.Move.from_uci(tok.tokens[raw_id]) in board.legal_moves:
                legal += 1
        except ValueError:
            pass

    print(f"positions:        {n}")
    print(f"move-token top-1: {well_formed / n:.1%}")
    print(f"legal top-1:      {legal / n:.1%}")


if __name__ == "__main__":
    main()
