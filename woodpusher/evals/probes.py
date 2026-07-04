"""Fixed-position probes: what a checkpoint believes at telltale positions.

    uv run python -m woodpusher.evals.probes --ckpt runs/smoke/best.pt

Importable for across-checkpoint comparisons (e.g. matplotlib curves of
legal mass per probe as a run progresses or across ladder rungs).
"""

import argparse

import chess
import torch

from ..tokenizer import Tokenizer
from .common import load_model, next_logits, topk_report

PROBES = [
    ("opening choice", []),
    ("reply to 1.e4", ["e2e4"]),
    ("defend e5 as black", ["e2e4", "e7e5", "g1f3"]),
    ("find Scholar's mate", ["e2e4", "e7e5", "f1c4", "f8c5", "d1h5", "g7g6"]),
]


def compute_probes(model, device, tok=None, elo=1800, k=5):
    tok = tok or Tokenizer()
    out = []
    for label, moves in PROBES:
        board = chess.Board()
        for u in moves:
            board.push(chess.Move.from_uci(u))
        ids = tok.prefix_ids(elo, elo) + [tok.move_id(u) for u in moves]
        logits = next_logits(model, ids, device)
        _, legal_mass, top = topk_report(logits, tok, board, k)
        out.append({"label": label, "legal_mass": legal_mass, "top": top})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/smoke/best.pt")
    ap.add_argument("--elo", type=int, default=1800)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    model, _ = load_model(args.ckpt, args.device)
    for p in compute_probes(model, args.device, elo=args.elo):
        print(f"{p['label']}  (legal mass {p['legal_mass'] * 100:.1f}%)")
        for t in p["top"]:
            mark = "" if t["legal"] else " x"
            print(f"  {t['move']:>8}{mark}  {t['prob'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
