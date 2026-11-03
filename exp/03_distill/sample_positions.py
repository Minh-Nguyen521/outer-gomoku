#!/usr/bin/env python3
"""Extract positions from harvest jsonl for teacher labelling.

Converts harvest records (moves prefix + stm) into the simple jsonl that
tools/label_positions.py expects: {"moves": [...], "stm": "B"} lines.
Position selection = every record (they were already sampled sensibly by
harvest_games.py: middlegame-dense, opening/endgame trimmed).
"""

import argparse
import json
import sys


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/harvest_top5.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    cfg = ap.parse_args(argv)

    n = 0
    with open(cfg.out, "w") as out:
        for i, line in enumerate(open(cfg.data)):
            if cfg.limit and i >= cfg.limit:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.write(json.dumps({"moves": r["moves"],
                                  "stm": r["stm"]}) + "\n")
            n += 1
    print(f"wrote {n} positions to {cfg.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
