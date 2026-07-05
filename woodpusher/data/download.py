"""Download monthly Lichess game dumps from database.lichess.org.

Sizes vary enormously by month: 2013-01 is ~17 MB (~120k games, ideal for
smoke tests); recent months exceed 30 GB (~90-100M games, ~8-10B tokens).
"""

import argparse
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", default="2013-01", help="YYYY-MM")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    url = URL.format(month=args.month)
    dest = out / url.rsplit("/", 1)[-1]
    if dest.exists():
        print(f"already downloaded: {dest}")
        return

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
    print(f"saved {dest}")


if __name__ == "__main__":
    main()
