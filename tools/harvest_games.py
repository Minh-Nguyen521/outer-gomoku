#!/usr/bin/env python3
"""Harvest training positions from real games between strong engines.

Replaces tools/gen_positions.py for training data. That script played
greedy/random moves, which produces blowouts: of its 50k positions, 80.2%
were already decided (teacher value >0.9 or <0.1) and only 3.6% were balanced
(0.4-0.6). A value head trained on that learns to output 0 or 1 and gets
almost no gradient where it matters -- close positions are exactly the ones an
evaluation function exists to judge.

Games between engines that can actually play produce the opposite
distribution: contested middlegames, real threat sequences, and openings that
occur in practice rather than uniformly at random.

Two targets are recorded per position, and they are complementary:

  * `result`   -- who actually won, from the side-to-move's view (+1/-1/0).
    Ground truth, but high variance: one blunder 30 plies later flips it.
  * teacher value/policy -- added afterwards by tools/label_positions.py.
    Low variance and gives a *policy* target, but caps the student at the
    teacher's judgement.

AlphaZero-style training blends both, which is why both are kept here rather
than choosing one now.

Output JSONL, one position per line:
    {"moves": [[x,y],...], "stm": "B"|"W", "ply": n,
     "result": 1|0|-1, "engines": [a, b], "winner": "B"|"W"|"draw"}

Usage:
    tools/harvest_games.py --engine-a ENGINE --engine-b ENGINE \\
        --games 50 --out data/harvest.jsonl [--every 2] [--min-ply 6]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import match  # noqa: E402


def play_and_record(idx, cmd_a, cmd_b, mcfg, cwd, cfg):
    """Play one game; return (positions, winner) or (None, reason)."""
    color_a = "B" if idx % 2 == 0 else "W"
    cmds = {"B": cmd_a if color_a == "B" else cmd_b,
            "W": cmd_a if color_a == "W" else cmd_b}
    brains = {s: match.Brain(cmds[s], cwd) for s in ("B", "W")}
    board = match.Board()
    played = []
    winner = None
    try:
        for side in ("B", "W"):
            b = brains[side]
            b.send("START 15")
            if (b.recv(120.0) or "").strip().upper() != "OK":
                return None, f"{side} init failed"
            b.send(f"INFO rule {cfg.rule}")
            b.send(f"INFO timeout_turn {cfg.timeout_turn}")
            b.send(f"INFO timeout_match {cfg.timeout_match}")

        mover, moves = "B", 0
        while True:
            mv, err = match.request_move(brains[mover], board, None, mcfg,
                                         moves == 0, cfg.timeout_turn, mover)
            if mv is None:
                return None, err or "no move"
            r, c = mv
            if not board.play(r, c, mover):
                return None, f"illegal move ({r},{c})"
            played.append((r, c, mover))
            moves += 1
            if board.win_at(r, c):
                winner = mover
                break
            if board.full():
                winner = "draw"
                break
            mover = "W" if mover == "B" else "B"
    finally:
        for b in brains.values():
            b.kill()

    # Sample positions along the game. Skip the opening -- near-empty boards
    # carry little signal and all look alike -- and skip the final plies, where
    # the win is already on the board and the label is trivially 0 or 1.
    out = []
    total = len(played)
    for ply in range(cfg.min_ply, max(cfg.min_ply, total - cfg.skip_last),
                     cfg.every):
        prefix = played[:ply]
        stm = "B" if ply % 2 == 0 else "W"
        if winner == "draw":
            res = 0
        else:
            res = 1 if winner == stm else -1
        out.append({
            "moves": [[x, y] for (x, y, _) in prefix],
            "stm": stm,
            "ply": ply,
            "result": res,
            "winner": winner,
            "engines": [" ".join(cmd_a), " ".join(cmd_b)],
        })
    return out, winner


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-a", required=True)
    ap.add_argument("--engine-b", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2,
                    help="sample every Nth ply (default 2: one per move pair)")
    ap.add_argument("--min-ply", type=int, default=6)
    ap.add_argument("--skip-last", type=int, default=4,
                    help="drop the final N plies: the win is already visible "
                         "there, so the label is trivial and teaches nothing")
    ap.add_argument("--timeout-turn", type=int, default=2000)
    ap.add_argument("--timeout-match", type=int, default=600000)
    ap.add_argument("--rule", type=int, default=0)
    cfg = ap.parse_args(argv)

    mcfg = match.parse_args([
        "--engine-a", cfg.engine_a, "--engine-b", cfg.engine_b,
        "--games", "1", "--rule", str(cfg.rule),
        "--timeout-turn", str(cfg.timeout_turn),
        "--timeout-match", str(cfg.timeout_match),
    ])
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd_a = match.build_engine_cmd(cfg.engine_a, mcfg)
    cmd_b = match.build_engine_cmd(cfg.engine_b, mcfg)

    n_pos = 0
    wins = {"B": 0, "W": 0, "draw": 0}
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    with open(cfg.out, "a") as fh:
        for i in range(cfg.games):
            positions, info = play_and_record(i, cmd_a, cmd_b, mcfg, cwd, cfg)
            if positions is None:
                print(f"  game {i}: SKIPPED ({info})", file=sys.stderr)
                continue
            wins[info] = wins.get(info, 0) + 1
            for rec in positions:
                fh.write(json.dumps(rec) + "\n")
            # Flush after every game. A harvest runs for hours, and without
            # this the positions sit in Python's buffer until the process
            # exits -- a run killed partway through leaves a 0-byte file
            # having played hundreds of games. (Termination happened to
            # flush ours, but that is luck, not a guarantee: SIGKILL, an OOM
            # or a crash would all lose the lot.)
            fh.flush()
            os.fsync(fh.fileno())
            n_pos += len(positions)
            print(f"  game {i}: {info} won, +{len(positions)} positions "
                  f"(total {n_pos})", file=sys.stderr, flush=True)
    print(f"wrote {n_pos} positions to {cfg.out}; results {wins}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
