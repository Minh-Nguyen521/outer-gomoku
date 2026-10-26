#!/usr/bin/env python3
"""Label positions with a strong teacher's value and policy (Phase 6 step 2).

Reads positions from tools/gen_positions.py and asks KataGomo's network for
its raw evaluation of each, via the GTP command `kata-raw-nn`. That returns a
single forward pass -- value and full policy -- with NO search, so labelling
is fast and the labels are the network's own judgement rather than a
search-blurred version of it.

Why a teacher rather than self-play: `PLAN.md` names rapfi as the Phase 6
reference, and rapfi's paper is titled "Distilling Efficient Neural Network
for the Game of Gomoku" -- distillation is the method. It also avoids the trap
that six measurements this session pointed at: our own evaluation is the weak
component, so training on our own labels would relearn our blind spots. The
teacher beat our engine 20-0 while doing zero search.

Value convention: `kata-raw-nn` reports whiteWin/whiteLoss. We store both the
raw white-relative value and `value_stm`, the win probability for the side to
move, since that is what an evaluation function actually needs to predict.

Caveats worth carrying into training:
  * The teacher plays FREESTYLE, not Outer-Open. Identical from move 2 on, so
    nearly all labels are sound, but it has no Outer-Open opening theory --
    early-ply labels are the least trustworthy.
  * The net (fs20x-b18c384n) was trained on 20x20 and we label 15x15. KataGo
    generalises across board sizes, but that is an assumption riding on label
    quality.

Output JSONL: {"moves": [...], "stm": ..., "ply": ...,
               "value_stm": float, "white_win": float, "policy": [225 floats]}
"""

import argparse
import json
import subprocess
import sys

N = 15
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"      # GTP skips 'I'


def to_gtp(x, y):
    return f"{LETTERS[x]}{N - y}"


class Teacher:
    def __init__(self, katago, model, config):
        self.p = subprocess.Popen(
            [katago, "gtp", "-model", model, "-config", config],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._drain(self._cmd("boardsize 15"))

    def _cmd(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        out = []
        while True:
            ln = self.p.stdout.readline()
            if ln == "":
                return None
            ln = ln.rstrip("\n")
            if ln.startswith("?"):
                return None
            if ln.startswith("=") :
                out.append(ln[2:] if len(ln) > 2 else "")
                break
        while True:
            ln = self.p.stdout.readline()
            if ln in ("\n", ""):
                break
            out.append(ln.rstrip("\n"))
        return out

    @staticmethod
    def _drain(x):
        return x

    def evaluate(self, moves, stm):
        """Set up the position and return (value_stm, white_win, policy[225])."""
        self._cmd("clear_board")
        for i, (x, y) in enumerate(moves):
            colour = "B" if i % 2 == 0 else "W"
            if self._cmd(f"play {colour} {to_gtp(x, y)}") is None:
                return None
        resp = self._cmd("kata-raw-nn 0")
        if not resp:
            return None

        white_win = None
        policy = []
        reading_policy = False
        for ln in resp:
            s = ln.strip()
            if s.startswith("whiteWin"):
                white_win = float(s.split()[1])
            elif s == "policy":
                reading_policy = True
            elif reading_policy:
                parts = s.split()
                if not parts or not _isfloat(parts[0]):
                    reading_policy = False
                    continue
                # KataGo prints NAN for occupied (illegal) squares. float()
                # accepts "NAN" happily, which would put NaN into every policy
                # target and make the argmax land on an occupied point.
                # Illegal moves have zero probability.
                for v in parts:
                    f = float(v)
                    policy.append(0.0 if f != f else f)   # f != f  <=> NaN
        if white_win is None or len(policy) < N * N:
            return None
        policy = policy[:N * N]           # trailing entry is the pass move
        # Renormalise: dropping the pass entry and zeroing illegal squares
        # leaves the distribution slightly short of 1.
        tot = sum(policy)
        if tot > 0:
            policy = [v / tot for v in policy]
        value_stm = white_win if stm == "W" else 1.0 - white_win
        return value_stm, white_win, policy

    def close(self):
        try:
            self.p.stdin.write("quit\n")
            self.p.stdin.flush()
        except Exception:
            pass
        self.p.kill()


def _isfloat(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", type=int, default=200)
    cfg = ap.parse_args(argv)

    t = Teacher(cfg.katago, cfg.model, cfg.config)
    done = failed = 0
    with open(cfg.inp) as fin, open(cfg.out, "w") as fout:
        for line in fin:
            if cfg.limit and done >= cfg.limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            moves = [tuple(m) for m in rec["moves"]]
            res = t.evaluate(moves, rec["stm"])
            if res is None:
                failed += 1
                continue
            v, ww, pol = res
            rec.update({"value_stm": v, "white_win": ww,
                        "policy": [round(p, 6) for p in pol]})
            fout.write(json.dumps(rec) + "\n")
            done += 1
            if cfg.progress and done % cfg.progress == 0:
                print(f"  labelled {done} (failed {failed})",
                      file=sys.stderr, flush=True)
    t.close()
    print(f"labelled {done} positions, {failed} failures", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
