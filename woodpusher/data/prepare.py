"""PGN (.pgn or .pgn.zst) -> tokenized uint16 shards train/val/test .bin + meta.json.

Games are encoded with the move-level tokenizer and concatenated back to back;
training samples random windows, so no padding is needed. One or more PGN files
can be streamed in sequence into a single shared corpus with a stable held-out
val/test split, so every ladder rung reads the same data and takes its own token
budget via the trainer's step count (no per-rung datasets).

A killed run resumes with --resume: it restores the counts from meta.json, skips
the already-read source games, and appends. Checkpoints (flush + meta) are written
every CHECKPOINT_EVERY source games, keeping the .bin shards and meta.json
consistent so a kill loses only the unflushed tail.
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

CHECKPOINT_EVERY = 25_000  # source games between consistent flush+meta checkpoints


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
    ap.add_argument("--max-games", type=int, default=0, help="stop after keeping this many, cumulative across resumes (0 = all)")
    ap.add_argument("--min-elo", type=int, default=0, help="require both players at or above")
    ap.add_argument("--min-plies", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--test-frac", type=float, default=0.005)
    ap.add_argument("--keep-bots", action="store_true", help="keep games with a BOT-titled player (dropped by default)")
    ap.add_argument("--time-control", default="", help="comma-separated speed buckets to keep (e.g. blitz,rapid); empty = all")
    ap.add_argument("--skip-games", type=int, default=0, help="skip this many source games before processing (manual chunking)")
    ap.add_argument("--resume", action="store_true", help="continue a killed run in --out: restore counts, skip already-read games, append")
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
    tokens = {s: 0 for s in splits}
    games = {s: 0 for s in splits}
    dropped = {"filter": 0, "bot": 0, "time_control": 0}
    source_read = 0
    skip = args.skip_games
    mode = "wb"

    meta_path = out / "meta.json"
    if args.resume and meta_path.exists():
        prev = json.loads(meta_path.read_text())
        tokens, games, dropped = prev["tokens"], prev["games"], prev["dropped"]
        source_read = prev.get("source_read", 0)
        skip = max(skip, source_read)  # jump back to the last consistent checkpoint
        mode = "ab"
        print(f"resuming from {meta_path}: {source_read:,} source games already read; appending")

    files = {s: open(out / f"{s}.bin", mode) for s in splits}

    def kept_total():
        return sum(games.values())

    def checkpoint():
        # flush every buffer then write meta: .bin and meta.json stay consistent,
        # so a kill loses only the unflushed tail and --resume picks up cleanly
        for s in splits:
            if buffers[s]:
                files[s].write(np.array(buffers[s], dtype=np.uint16).tobytes())
                buffers[s].clear()
            files[s].flush()
        meta_path.write_text(json.dumps({
            "vocab_size": tok.vocab_size,
            "tokens": tokens,
            "games": games,
            "dropped": dropped,
            "source_read": source_read,
            "args": vars(args),
        }, indent=2))

    to_skip = skip
    stop = bool(args.max_games) and kept_total() >= args.max_games
    pbar = tqdm(unit=" games", desc="parsing", initial=source_read)
    for pgn_path in args.pgn:
        if stop:
            break
        with open_pgn(Path(pgn_path)) as stream:
            while not stop:
                if to_skip > 0:
                    try:
                        found = chess.pgn.skip_game(stream)
                    except Exception:
                        found = False
                    if not found:
                        break  # file exhausted mid-skip; continue into the next file
                    to_skip -= 1
                    continue
                try:
                    game = chess.pgn.read_game(stream)
                except Exception as e:  # truncated/corrupt stream (e.g. partial download)
                    print(f"\nstream error in {pgn_path} ({type(e).__name__}); stopping this file")
                    break
                if game is None:
                    break
                source_read += 1
                pbar.update(1)

                h = game.headers
                if not args.keep_bots and (h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT"):
                    dropped["bot"] += 1
                elif keep_tc and speed_category(h.get("TimeControl", "")) not in keep_tc:
                    dropped["time_control"] += 1
                else:
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
                    else:
                        r = rng.random()
                        split = "test" if r < args.test_frac else "val" if r < args.test_frac + args.val_frac else "train"
                        ids = tok.encode_game(moves, welo, belo)
                        buffers[split].extend(ids)
                        tokens[split] += len(ids)
                        games[split] += 1
                        if args.max_games and kept_total() >= args.max_games:
                            stop = True

                if source_read % CHECKPOINT_EVERY == 0:
                    checkpoint()
    pbar.close()
    checkpoint()
    for f in files.values():
        f.close()
    print(json.dumps(json.loads(meta_path.read_text()), indent=2))


if __name__ == "__main__":
    main()
