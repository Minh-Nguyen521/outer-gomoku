#!/usr/bin/env python3
"""Fine-tune from WIN games: positions where strong engines beat weaker ones.

Sources (exp/06_finetune/win_games):
  - harvest games grouped by engine pair; a game is a "win game" for the
    winner side. We keep games whose winner is one of the TOP engines
    (vibefive/agmk26/yixin/rapfi) -- i.e. demonstrations of strong play --
    and keep only WINNER-side plies (early ones; the loser's blunders are
    excluded so we don't imitate getting into trouble).
  - policy target = played move (one-hot); value = +/-1 from game result.

Degeneration fixes vs exp/04_gumbel v2 run:
  - LR 1e-4 (was 1e-3): protect the distilled/ExIt features
  - no Gumbel pi-targets from noisy halving scores: plain played moves
  - short schedule (3 epochs) with checkpoint each epoch
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from tools.train_policy import Net, build_planes  # noqa: E402

N = 15
TOP = ("vibefive", "agmk26", "yixin18", "rapfi25", "rapfi")


def iter_games(path):
    cur = None
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cur is not None and (r["ply"] <= cur[-1]["ply"]
                                or r["engines"] != cur[-1]["engines"]):
            yield cur
            cur = None
        cur = (cur or []) + [r]
    if cur:
        yield cur


def strong(name):
    return any(t in name for t in TOP)


def movers_filter(is_winner, include_losers):
    return is_winner or include_losers


def load_win_positions(path, max_plies=40, stride=1, losers=False):
    """Positions from STRONG-engine WIN games, mover = the winner.

    losers=True additionally keeps LOSER-side plies (defensive play by the
    beaten engine -- teaches the net how to answer threats), with value -1.
    """
    X, P, V = [], [], []
    n_games = 0
    for g in iter_games(path):
        moves = g[-1]["moves"]
        winner = g[-1]["winner"]
        engs = g[-1]["engines"]
        widx = 0 if winner == "B" else 1
        if winner == "draw" or not strong(engs[widx]):
            continue
        n_games += 1
        for ply in range(2, min(len(moves), max_plies), stride):
            mover_is_winner = (ply % 2 == 0 and winner == "B") or \
                              (ply % 2 == 1 and winner == "W")
            if not movers_filter(mover_is_winner, losers):
                continue
            nxt = moves[ply]
            if not (0 <= nxt[0] < N and 0 <= nxt[1] < N):
                continue
            stm = "B" if ply % 2 == 0 else "W"
            onehot = np.zeros(N * N, dtype=np.float32)
            onehot[nxt[1] * N + nxt[0]] = 1.0
            X.append(build_planes([tuple(m) for m in moves[:ply]], stm))
            P.append(onehot)
            # winner moves: +1; loser defensive moves: value from their
            # (losing) perspective -- the ANSWER to a threat is what we
            # teach, not that they eventually lost
            V.append(1.0 if mover_is_winner else -1.0)
    print(f"win-games: {n_games}  positions: {len(X)}", file=sys.stderr)
    return (np.stack(X), np.stack(P), np.asarray(V, dtype=np.float32))


def load_labelled_positions(path):
    """Direct KataGomo-labelled positions: policy vector + value_stm."""
    X, P, V = [], [], []
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "policy" not in r or "value_stm" not in r:
            continue
        X.append(build_planes([tuple(m) for m in r["moves"]], r["stm"]))
        P.append(np.asarray(r["policy"], dtype=np.float32))
        V.append(float(r["value_stm"]))
    print(f"labelled positions: {len(X)}", file=sys.stderr)
    return (np.stack(X), np.stack(P), np.asarray(V, dtype=np.float32))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["win", "labelled"], default="win")
    ap.add_argument("--data", default="data/harvest_merged.jsonl")
    ap.add_argument("--init", default="exp/pipeline/exit_net.pt")
    ap.add_argument("--blocks", type=int, default=0,
                    help="if >0, train a FRESH net of this size instead of "
                         "warm-starting (architecture study)")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-plies", type=int, default=40)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--losers", action="store_true",
                    help="also learn loser-side defensive moves")
    ap.add_argument("--out", default="exp/06_finetune/finetuned.pt")
    cfg = ap.parse_args(argv)

    dev = ("mps" if torch.backends.mps.is_available() else "cpu")
    if cfg.mode == "labelled":
        X, P, V = load_labelled_positions(cfg.data)
    else:
        X, P, V = load_win_positions(cfg.data, cfg.max_plies, cfg.stride,
                                     cfg.losers)

    n = len(X)
    n_val = max(1, int(n * cfg.val_frac))
    perm = np.random.permutation(n)
    vi, ti = perm[:n_val], perm[n_val:]
    # keep the big pool on CPU; move only minibatches (full pool OOMs MPS)
    X_np, P_np, V_np = X, P, V
    Xv = torch.from_numpy(X_np[vi]).to(dev)
    Pv = torch.from_numpy(P_np[vi]).to(dev)
    Vv = torch.from_numpy(V_np[vi]).to(dev)

    def batch(idx_np):
        xb = torch.from_numpy(X_np[idx_np]).to(dev)
        pb = torch.from_numpy(P_np[idx_np]).to(dev)
        vb = torch.from_numpy(V_np[idx_np]).to(dev)
        return xb, pb, vb

    if cfg.blocks > 0:
        net = Net(cfg.blocks, cfg.channels).to(dev)
        ck = {"blocks": cfg.blocks, "channels": cfg.channels}
        print(f"fresh net: {cfg.blocks} blocks x {cfg.channels} ch", file=sys.stderr)
    else:
        ck = torch.load(cfg.init, map_location=dev)
        net = Net(ck.get("blocks", 4), ck.get("channels", 64)).to(dev)
        net.load_state_dict(ck["state_dict"])
        print(f"warm-started from {cfg.init}", file=sys.stderr)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out)), exist_ok=True)
    for ep in range(cfg.epochs):
        net.train()
        perm_t = np.random.permutation(len(ti))
        tot = 0.0
        for i in range(0, len(ti), cfg.batch):
            idx = perm_t[i:i + cfg.batch]
            xb, pb, vb = batch(ti[idx])
            logits, val = net(xb)
            loss = -(pb * F.log_softmax(logits, 1)).sum(1).mean() + \
                F.mse_loss(val, vb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        net.eval()
        with torch.no_grad():
            # evaluate val in chunks (val pool may also exceed memory)
            cor = 0
            vae = 0.0
            for i in range(0, len(vi), 2048):
                ii = vi[i:i + 2048]
                logits, val = net(torch.from_numpy(X_np[ii]).to(dev))
                cor += int((logits.argmax(1).cpu().numpy()
                            == P_np[ii].argmax(1)).sum())
                vae += float((val.squeeze(-1).cpu().numpy()
                              - V_np[ii]).__abs__().sum())
            top1 = cor / len(vi)
            vmae = vae / len(vi)
        print(f"ep {ep+1}: train {tot/len(ti):.4f}  "
              f"val top1 {float(top1)*100:.1f}%  vMAE {float(vmae):.3f}",
              file=sys.stderr, flush=True)
        torch.save({"state_dict": net.state_dict(),
                    "blocks": ck.get("blocks", 4),
                    "channels": ck.get("channels", 64),
                    "planes": 6, "board": N, "finetune": True},
                   f"{cfg.out}.{ep+1}")
    torch.save({"state_dict": net.state_dict(),
                "blocks": ck.get("blocks", 4),
                "channels": ck.get("channels", 64),
                "planes": 6, "board": N, "finetune": True}, cfg.out)
    print(f"saved {cfg.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
