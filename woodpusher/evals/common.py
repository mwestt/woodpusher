"""Shared helpers for evals and play: checkpoint loading, val-set game
iteration, and move selection with optional legality masking."""

import random
from pathlib import Path

import chess
import numpy as np
import torch

from ..model import ModelConfig, Transformer
from ..tokenizer import Tokenizer


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ModelConfig(**ckpt["model_config"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def sample_val_games(data_dir, n, seed=0):
    """Return up to n token-id lists (<bos>..<eos>) from val.bin."""
    tok = Tokenizer()
    data = np.fromfile(Path(data_dir) / "val.bin", dtype=np.uint16)
    bos = np.flatnonzero(data == tok.bos_id)
    eos = np.flatnonzero(data == tok.eos_id)
    games, j = [], 0
    for b in bos:
        while j < len(eos) and eos[j] < b:
            j += 1
        if j >= len(eos):
            break
        games.append(data[b : eos[j] + 1].tolist())
    random.Random(seed).shuffle(games)
    return games[:n]


def game_moves(tok, ids):
    """UCI move strings from an encoded game (strips prefix and <eos>)."""
    return [tok.tokens[i] for i in ids if tok.is_move_id(i)]


@torch.no_grad()
def next_logits(model, ids, device):
    ids = ids[-model.cfg.block_size :]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, _ = model(x)
    return logits[0, -1]


def pick_move(model, tok, ids, board, device, temperature=0.0, mask_legal=True):
    """Choose the model's move for the current position.

    Returns (move, raw_argmax_token_id). `move` is a legal chess.Move when
    mask_legal, else whatever the raw choice decodes to (possibly None).
    """
    logits = next_logits(model, ids, device)
    raw_id = int(logits.argmax())

    if mask_legal:
        legal = {tok.move_id(m.uci()): m for m in board.legal_moves}
        masked = torch.full_like(logits, float("-inf"))
        idx = torch.tensor(list(legal.keys()), device=logits.device)
        masked[idx] = logits[idx]
        if temperature <= 0:
            chosen = int(masked.argmax())
        else:
            probs = torch.softmax(masked / temperature, dim=-1)
            chosen = int(torch.multinomial(probs, 1))
        return legal[chosen], raw_id

    if not tok.is_move_id(raw_id):
        return None, raw_id
    try:
        move = chess.Move.from_uci(tok.tokens[raw_id])
    except ValueError:
        return None, raw_id
    return (move if move in board.legal_moves else None), raw_id
