"""Move-level UCI tokenizer.

The vocabulary is fully deterministic — generated in code, no vocab file:

    0..2      <pad> <bos> <eos>
    3..16     white-Elo bucket tokens  <welo:...>
    17..30    black-Elo bucket tokens  <belo:...>
    31..      every expressible UCI move:
              all from!=to square pairs (4032) + promotions (176)

A game encodes as:  <bos> <welo:B> <belo:B> move move ... <eos>

The Elo prefix tokens are the conditioning dial: at inference, setting them
steers the model toward the move distribution of that rating bracket.
"""

FILES = "abcdefgh"
RANKS = "12345678"

# Rating buckets: unknown, <800, 200-wide bins 800..2999, >=3000 (14 total)
BUCKETS = ["unk", "lt800"] + [str(lo) for lo in range(800, 3000, 200)] + ["ge3000"]


def elo_bucket(elo: int | None) -> str:
    if elo is None:
        return "unk"
    if elo < 800:
        return "lt800"
    if elo >= 3000:
        return "ge3000"
    return str(800 + 200 * ((elo - 800) // 200))


def _all_uci_moves() -> list[str]:
    squares = [f + r for r in RANKS for f in FILES]
    moves = [a + b for a in squares for b in squares if a != b]
    for i, f in enumerate(FILES):
        for j in (i - 1, i, i + 1):
            if 0 <= j < 8:
                for piece in "qrbn":
                    moves.append(f"{f}7{FILES[j]}8{piece}")  # white promotion
                    moves.append(f"{f}2{FILES[j]}1{piece}")  # black promotion
    return moves


class Tokenizer:
    def __init__(self):
        self.tokens = (
            ["<pad>", "<bos>", "<eos>"]
            + [f"<welo:{b}>" for b in BUCKETS]
            + [f"<belo:{b}>" for b in BUCKETS]
            + _all_uci_moves()
        )
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
        self.first_move_id = 3 + 2 * len(BUCKETS)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def prefix_ids(self, white_elo: int | None = None, black_elo: int | None = None) -> list[int]:
        return [
            self.bos_id,
            self.token_to_id[f"<welo:{elo_bucket(white_elo)}>"],
            self.token_to_id[f"<belo:{elo_bucket(black_elo)}>"],
        ]

    def encode_game(
        self,
        uci_moves: list[str],
        white_elo: int | None = None,
        black_elo: int | None = None,
    ) -> list[int]:
        ids = self.prefix_ids(white_elo, black_elo)
        ids.extend(self.token_to_id[m] for m in uci_moves)
        ids.append(self.eos_id)
        return ids

    def move_id(self, uci: str) -> int:
        return self.token_to_id[uci]

    def is_move_id(self, token_id: int) -> bool:
        return token_id >= self.first_move_id

    def decode(self, ids: list[int]) -> list[str]:
        return [self.tokens[i] for i in ids]
