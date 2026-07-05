"""PGN (.pgn or .pgn.zst) -> tokenized uint16 shards train/val/test .bin + meta.json.

Games are encoded with the move-level tokenizer and concatenated back to back;
training samples random windows, so no padding is needed. One or more PGN files
can be streamed in sequence into a single shared corpus with a stable held-out
val/test split, so every ladder rung reads the same data and takes its own token
budget via the trainer's step count (no per-rung datasets).
"""

import argparse
import json
import random
from pathlib import Path

import chess.pgn
import numpy as np
import zstandard
from tqdm import tqdm

from ..tokenizer import Tokenizer

FLUSH_EVERY = 1_000_000  # token ids buffered per split before writing


def open_pgn(path: Path):
    if path.suffix == ".zst":
        return zstandard.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def parse_elo(headers, key):
    try:
        return int(headers.get(key, ""))
    except ValueError:
        return None


def speed_category(tc: str) -> str:
    """Lichess speed bucket from a TimeControl header ("base+increment" or "-").

    Uses Lichess's estimated-duration rule: base + 40 * increment seconds.
    """
    if not tc or tc == "-":
        return "correspondence"
    try:
        base, inc = tc.split("+")
        est = int(base) + 40 * int(inc)
    except (ValueError, AttributeError):
        return "unknown"
    if est < 30:
        return "ultrabullet"
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    if est < 1500:
        return "rapid"
    return "classical"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", nargs="+", required=True, help="one or more PGN files, streamed in order")
    ap.add_argument("--out", default="data/prepared")
    ap.add_argument("--max-games", type=int, default=0, help="stop after keeping this many (0 = all)")
    ap.add_argument("--min-elo", type=int, default=0, help="require both players at or above")
    ap.add_argument("--min-plies", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--test-frac", type=float, default=0.005)
    ap.add_argument("--keep-bots", action="store_true", help="keep games with a BOT-titled player (dropped by default)")
    ap.add_argument("--time-control", default="", help="comma-separated speed buckets to keep (e.g. blitz,rapid); empty = all")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    tok = Tokenizer()
    max_plies = 512 - 4  # game must fit one block alongside <bos> + 2 elo tokens + <eos>
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep_tc = {s.strip() for s in args.time_control.split(",") if s.strip()}

    splits = ("train", "val", "test")
    buffers = {s: [] for s in splits}
    files = {s: open(out / f"{s}.bin", "wb") for s in splits}
    tokens = {s: 0 for s in splits}
    games = {s: 0 for s in splits}
    dropped = {"filter": 0, "bot": 0, "time_control": 0}

    def flush(split):
        if buffers[split]:
            files[split].write(np.array(buffers[split], dtype=np.uint16).tobytes())
            buffers[split].clear()

    def kept_total():
        return games["train"] + games["val"] + games["test"]

    stop = False
    pbar = tqdm(unit=" games", desc="parsing")
    for pgn_path in args.pgn:
        if stop:
            break
        with open_pgn(Path(pgn_path)) as stream:
            while True:
                game = chess.pgn.read_game(stream)
                if game is None:
                    break
                pbar.update(1)

                h = game.headers
                if not args.keep_bots and (h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT"):
                    dropped["bot"] += 1
                    continue
                if keep_tc and speed_category(h.get("TimeControl", "")) not in keep_tc:
                    dropped["time_control"] += 1
                    continue

                moves = [m.uci() for m in game.mainline_moves()]
                welo = parse_elo(h, "WhiteElo")
                belo = parse_elo(h, "BlackElo")
                if (
                    game.errors
                    or len(moves) < args.min_plies
                    or len(moves) > max_plies
                    or (args.min_elo and (welo is None or belo is None
                                          or welo < args.min_elo or belo < args.min_elo))
                ):
                    dropped["filter"] += 1
                    continue

                r = rng.random()
                split = "test" if r < args.test_frac else "val" if r < args.test_frac + args.val_frac else "train"
                ids = tok.encode_game(moves, welo, belo)
                buffers[split].extend(ids)
                tokens[split] += len(ids)
                games[split] += 1
                if len(buffers[split]) >= FLUSH_EVERY:
                    flush(split)
                if args.max_games and kept_total() >= args.max_games:
                    stop = True
                    break
    pbar.close()

    for split in files:
        flush(split)
        files[split].close()

    meta = {
        "vocab_size": tok.vocab_size,
        "tokens": tokens,
        "games": games,
        "dropped": dropped,
        "args": vars(args),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
