"""Estimate the model's Elo by playing matches against strength-limited Stockfish.

Requires a Stockfish binary (https://stockfishchess.org/download/). Stockfish's
UCI_Elo floor is 1320; below that, use --skill-level opponents or accept an
upper bound. --protocol raw replicates Karvonen's chess_gpt_eval rules
(unmasked sampling, 5 attempts per move, forfeit on failure).

    uv run python -m woodpusher.evals.elo_vs_stockfish --ckpt runs/25m/best.pt ^
        --stockfish C:/tools/stockfish/stockfish.exe --games 20 --engine-elo 1400
"""

import argparse
import math

import chess
import chess.engine
import torch

from ..tokenizer import Tokenizer
from .common import load_model, next_logits, pick_move, sampling_generator


def play_game(model, tok, engine, device, model_is_white, model_elo, engine_elo,
              temperature, limit, max_plies, generator=None, raw_attempts=0):
    """Returns (model_score, illegal_attempts, forfeited).

    With raw_attempts > 0, plays the raw-sampling protocol (as in Karvonen's
    chess_gpt_eval): unmasked sampling with a per-attempt temperature ramp
    min(attempt/max_attempts + 0.001, 0.5); exhausting the attempts on one
    move forfeits the game for the model.
    """
    board = chess.Board()
    # engine_elo=None would condition on <elo:unk>, which is nearly unseen in
    # training and garbles the model — fall back to the model's own bucket
    engine_bucket = engine_elo if engine_elo is not None else model_elo
    welo = model_elo if model_is_white else engine_bucket
    belo = engine_bucket if model_is_white else model_elo
    ids = tok.prefix_ids(welo, belo)
    illegal_attempts = 0

    while not board.is_game_over() and board.ply() < max_plies:
        if (board.turn == chess.WHITE) == model_is_white:
            if raw_attempts:
                logits = next_logits(model, ids, device)
                move = None
                for attempt in range(raw_attempts):
                    t = min(attempt / raw_attempts + 0.001, 0.5)
                    cand, _ = pick_move(model, tok, ids, board, device, t,
                                        mask_legal=False, generator=generator, logits=logits)
                    if cand is not None:
                        move = cand
                        break
                    illegal_attempts += 1
                if move is None:
                    return 0.0, illegal_attempts, True
            else:
                move, _ = pick_move(model, tok, ids, board, device, temperature, generator=generator)
        else:
            move = engine.play(board, limit).move
        ids.append(tok.move_id(move.uci()))
        board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0":
        return (1.0 if model_is_white else 0.0), illegal_attempts, False
    if result == "0-1":
        return (0.0 if model_is_white else 1.0), illegal_attempts, False
    return 0.5, illegal_attempts, False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stockfish", required=True, help="path to stockfish executable")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--engine-elo", type=int, default=0,
                    help="UCI_Elo strength limit (floor 1320; default 1400). With --skill-level, "
                         "used only as an assumed rating to anchor the Elo estimate")
    ap.add_argument("--skill-level", type=int, default=None,
                    help="limit strength via Skill Level (0-20) instead of UCI_Elo; reaches "
                         "far below the 1320 UCI_Elo floor (lichess levels / Chess-GPT plots)")
    ap.add_argument("--model-elo", type=int, default=2200, help="conditioning bucket for the model's side")
    ap.add_argument("--temperature", type=float, default=0.3,
                    help="model sampling temperature (ignored with --protocol raw, which ramps 0->0.5)")
    ap.add_argument("--movetime", type=float, default=None,
                    help="engine seconds/move (default 0.05; 0.1 with --protocol raw)")
    ap.add_argument("--protocol", choices=["masked", "raw"], default="masked",
                    help="masked: model samples among legal moves only. raw: unmasked sampling, "
                         "5 attempts/move with temp ramp then forfeit, model always White, "
                         "0.1s/move, 1000-ply cap (Karvonen's chess_gpt_eval protocol); "
                         "combine with --skill-level")
    ap.add_argument("--engine-nodes", type=int, default=0,
                    help="limit engine by node count instead of time (deterministic engine play)")
    ap.add_argument("--max-plies", type=int, default=None,
                    help="default 300 (1000 with --protocol raw)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None, help="fix model sampling RNG for reproducibility")
    args = ap.parse_args()

    if args.skill_level is not None:
        # engine strength set by Skill Level; --engine-elo (if given) is only the
        # assumed rating used for the Elo estimate and the engine side's Elo bucket
        engine_elo = args.engine_elo or None
        opponent = f"Stockfish skill {args.skill_level}"
    else:
        engine_elo = max(1320, args.engine_elo or 1400)
        if args.engine_elo and engine_elo != args.engine_elo:
            print(f"note: Stockfish UCI_Elo floor is 1320, using {engine_elo}")
        opponent = f"Stockfish@{engine_elo}"

    tok = Tokenizer()
    model, _ = load_model(args.ckpt, args.device)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    if args.skill_level is not None:
        engine.configure({"Skill Level": args.skill_level})
    else:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": engine_elo})

    raw = args.protocol == "raw"
    movetime = args.movetime or (0.1 if raw else 0.05)
    max_plies = args.max_plies or (1000 if raw else 300)
    raw_attempts = 5 if raw else 0
    limit = (chess.engine.Limit(nodes=args.engine_nodes) if args.engine_nodes
             else chess.engine.Limit(time=movetime))
    gen = sampling_generator(args.device, args.seed)
    score = 0.0
    illegal_total = 0
    forfeits = 0
    try:
        for g in range(args.games):
            model_is_white = True if raw else g % 2 == 0
            s, ill, forfeit = play_game(model, tok, engine, args.device, model_is_white,
                                        args.model_elo, engine_elo, args.temperature,
                                        limit, max_plies, generator=gen,
                                        raw_attempts=raw_attempts)
            score += s
            illegal_total += ill
            forfeits += forfeit
            note = " FORFEIT (illegal moves)" if forfeit else ""
            print(f"game {g + 1}/{args.games}: {'win' if s == 1 else 'draw' if s == 0.5 else 'loss'}"
                  f" ({'white' if model_is_white else 'black'}) running score {score}/{g + 1}{note}")
    finally:
        engine.quit()

    frac = score / args.games
    print(f"\nscore vs {opponent}: {score}/{args.games} ({frac:.1%})")
    if raw:
        print(f"illegal attempts: {illegal_total}; forfeits: {forfeits}/{args.games}")
    if engine_elo is None:
        print("(no Elo estimate: pass --engine-elo with an assumed rating for this skill level)")
    elif 0 < frac < 1:
        diff = 400 * math.log10(frac / (1 - frac))
        print(f"estimated model Elo: {engine_elo + diff:.0f}")
    else:
        print("score at 0% or 100% — Elo difference outside measurable range; "
              "adjust opponent strength and rerun")


if __name__ == "__main__":
    main()
