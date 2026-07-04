"""Mate-in-one accuracy: a tactics eval needing no external engine.

Scans val games for positions where the side to move has a mate in one,
then checks whether the model finds a mating move (raw top-1, and top-1
after legality masking). Baseline is the chance a random legal move mates.

    uv run python -m woodpusher.evals.mate_in_one --ckpt runs/5m/best.pt --data-dir data/main
"""

import argparse

import chess
import torch

from ..tokenizer import Tokenizer
from .common import game_moves, load_model, next_logits, pick_move, sample_val_games


def mating_moves(board):
    mates = []
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            mates.append(m)
        board.pop()
    return mates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-positions", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = Tokenizer()
    model, _ = load_model(args.ckpt, args.device)
    games = sample_val_games(args.data_dir, 100_000, args.seed)

    n = raw_hits = masked_hits = 0
    baseline_sum = 0.0
    for ids in games:
        if n >= args.max_positions:
            break
        moves = game_moves(tok, ids)
        board = chess.Board()
        for k, m in enumerate(moves):
            mates = mating_moves(board)
            if mates:
                context = ids[: 3 + k]
                raw_id = int(next_logits(model, context, args.device).argmax())
                if tok.is_move_id(raw_id):
                    try:
                        if chess.Move.from_uci(tok.tokens[raw_id]) in mates:
                            raw_hits += 1
                    except ValueError:
                        pass
                masked_move, _ = pick_move(model, tok, context, board, args.device)
                if masked_move in mates:
                    masked_hits += 1
                baseline_sum += len(mates) / board.legal_moves.count()
                n += 1
                if n >= args.max_positions:
                    break
            board.push(chess.Move.from_uci(m))

    if n == 0:
        print("no mate-in-one positions found — use more val games")
        return
    print(f"positions:            {n}")
    print(f"raw top-1 mates:      {raw_hits / n:.1%}")
    print(f"masked top-1 mates:   {masked_hits / n:.1%}")
    print(f"random-legal baseline: {baseline_sum / n:.1%}")


if __name__ == "__main__":
    main()
