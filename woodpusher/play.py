"""Play against a checkpoint in the terminal.

The model's sampling is masked to legal moves, so any checkpoint is playable;
whenever its raw top-1 pick was illegal, that's flagged — a live view of the
illegal-move eval. --selfplay makes the model play itself (non-interactive).

    uv run python -m woodpusher.play --ckpt runs/5m/best.pt --color white --model-elo 1500
"""

import argparse
import random

import chess
import torch

from .tokenizer import Tokenizer
from .evals.common import load_model, pick_move, sampling_generator


def render(board, ascii_board):
    if ascii_board:
        return str(board)
    return board.unicode(empty_square="·")


def parse_user_move(board, text):
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text)
        except ValueError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--color", choices=["white", "black", "random"], default="white",
                    help="your color")
    ap.add_argument("--model-elo", type=int, default=1800, help="conditioning for the model's side")
    ap.add_argument("--your-elo", type=int, default=1600, help="conditioning for your side")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--ascii", action="store_true", help="plain-ASCII board")
    ap.add_argument("--selfplay", action="store_true", help="model plays itself once and exits")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None,
                    help="fix the sampling RNG: same seed + same checkpoint + same inputs = same game")
    args = ap.parse_args()

    tok = Tokenizer()
    model, _ = load_model(args.ckpt, args.device)
    gen = sampling_generator(args.device, args.seed)
    board = chess.Board()

    if args.selfplay:
        ids = tok.prefix_ids(args.model_elo, args.model_elo)
        sans, raw_illegal = [], 0
        while not board.is_game_over() and board.ply() < args.max_plies:
            move, raw_id = pick_move(model, tok, ids, board, args.device, args.temperature, generator=gen)
            if not (tok.is_move_id(raw_id) and chess.Move.from_uci(tok.tokens[raw_id]) in board.legal_moves):
                raw_illegal += 1
            sans.append(board.san(move))
            ids.append(tok.move_id(move.uci()))
            board.push(move)
        movetext = " ".join(
            (f"{i // 2 + 1}. {s}" if i % 2 == 0 else s) for i, s in enumerate(sans)
        )
        print(movetext)
        print(f"result: {board.result(claim_draw=True)}  "
              f"plies: {board.ply()}  raw-illegal: {raw_illegal}/{board.ply()}")
        return

    user_white = args.color == "white" or (args.color == "random" and random.random() < 0.5)
    welo = args.your_elo if user_white else args.model_elo
    belo = args.model_elo if user_white else args.your_elo
    ids = tok.prefix_ids(welo, belo)
    print(f"you are {'white' if user_white else 'black'}; model conditioned as "
          f"{args.model_elo} vs your {args.your_elo}")
    print("enter moves in SAN (Nf3) or UCI (g1f3); 'moves' lists legal moves, 'quit' exits\n")

    while not board.is_game_over():
        print(render(board, args.ascii), "\n")
        if board.turn == chess.WHITE and user_white or board.turn == chess.BLACK and not user_white:
            text = input("your move> ").strip()
            if text == "quit":
                return
            if text == "moves":
                print(", ".join(board.san(m) for m in board.legal_moves), "\n")
                continue
            move = parse_user_move(board, text)
            if move is None:
                print("could not parse that as a legal move\n")
                continue
        else:
            move, raw_id = pick_move(model, tok, ids, board, args.device, args.temperature, generator=gen)
            note = ""
            if not (tok.is_move_id(raw_id) and chess.Move.from_uci(tok.tokens[raw_id]) in board.legal_moves):
                note = f"  (raw top-1 was illegal: {tok.tokens[raw_id]})"
            print(f"model plays: {board.san(move)}{note}\n")
        ids.append(tok.move_id(move.uci()))
        board.push(move)

    print(render(board, args.ascii))
    print(f"\ngame over: {board.result(claim_draw=True)}")


if __name__ == "__main__":
    main()
