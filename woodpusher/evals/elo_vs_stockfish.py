"""Estimate the model's Elo by playing matches against strength-limited Stockfish.

Requires a Stockfish binary (https://stockfishchess.org/download/). Stockfish's
UCI_Elo floor is 1320; below that this eval can only bound the model from above.

    uv run python -m woodpusher.evals.elo_vs_stockfish --ckpt runs/25m/best.pt ^
        --stockfish C:/tools/stockfish/stockfish.exe --games 20 --engine-elo 1400
"""

import argparse
import math

import chess
import chess.engine
import torch

from ..tokenizer import Tokenizer
from .common import load_model, pick_move, sampling_generator


def play_game(model, tok, engine, device, model_is_white, model_elo, engine_elo,
              temperature, limit, max_plies, generator=None):
    board = chess.Board()
    welo = model_elo if model_is_white else engine_elo
    belo = engine_elo if model_is_white else model_elo
    ids = tok.prefix_ids(welo, belo)

    while not board.is_game_over() and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == model_is_white:
            move, _ = pick_move(model, tok, ids, board, device, temperature, generator=generator)
        else:
            move = engine.play(board, limit).move
        ids.append(tok.move_id(move.uci()))
        board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0 if model_is_white else 0.0
    if result == "0-1":
        return 0.0 if model_is_white else 1.0
    return 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stockfish", required=True, help="path to stockfish executable")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--engine-elo", type=int, default=1400)
    ap.add_argument("--model-elo", type=int, default=2200, help="conditioning bucket for the model's side")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--movetime", type=float, default=0.05)
    ap.add_argument("--engine-nodes", type=int, default=0,
                    help="limit engine by node count instead of time (deterministic engine play)")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None, help="fix model sampling RNG for reproducibility")
    args = ap.parse_args()

    engine_elo = max(1320, args.engine_elo)
    if engine_elo != args.engine_elo:
        print(f"note: Stockfish UCI_Elo floor is 1320, using {engine_elo}")

    tok = Tokenizer()
    model, _ = load_model(args.ckpt, args.device)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": engine_elo})

    limit = (chess.engine.Limit(nodes=args.engine_nodes) if args.engine_nodes
             else chess.engine.Limit(time=args.movetime))
    gen = sampling_generator(args.device, args.seed)
    score = 0.0
    try:
        for g in range(args.games):
            s = play_game(model, tok, engine, args.device, g % 2 == 0,
                          args.model_elo, engine_elo, args.temperature,
                          limit, args.max_plies, generator=gen)
            score += s
            print(f"game {g + 1}/{args.games}: {'win' if s == 1 else 'draw' if s == 0.5 else 'loss'}"
                  f" ({'white' if g % 2 == 0 else 'black'}) running score {score}/{g + 1}")
    finally:
        engine.quit()

    frac = score / args.games
    print(f"\nscore vs Stockfish@{engine_elo}: {score}/{args.games} ({frac:.1%})")
    if 0 < frac < 1:
        diff = 400 * math.log10(frac / (1 - frac))
        print(f"estimated model Elo: {engine_elo + diff:.0f}")
    else:
        print("score at 0% or 100% — Elo difference outside measurable range; "
              "adjust --engine-elo and rerun")


if __name__ == "__main__":
    main()
