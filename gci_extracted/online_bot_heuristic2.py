"""
Online Gomoku wrapper: full-heuristic bot v2 — port of the repo's proven
classical evaluation (src/eval.cpp M-mask line table + src/eval_v2.cpp fork
analysis + threat-restricted candidates), no model, no deep search.

Techniques (all measured in this repo before being ported):
  * M-mask 65536-entry line table (OOGiveMeFive style): classifyLine ports
    src/eval.cpp:210-237 including isJumpFour — broken/gap fours are promoted
    to four threats instead of miscounted as threes.
  * eval_v2 fork counting: four-threats deduped by stone SPAN not gain
    square (src/eval_v2.cpp:128-162 — open four _XXXX_ is one threat, not
    two), true open threes = filling an empty yields an open four
    (src/eval_v2.cpp:164-211). Fork bonuses: 44=90k, 43=60k, 33=30k, def x3/2.
  * Priority ladder per search_threatcand.cpp Level-1/2 restriction: if
    opponent has a five-completing square, candidates = only our five
    completions + blocks (provably lossless, shrinks candidates hugely).
  * Shallow VCF probe (depth 6, src/engines/search_vct.cpp style): chains
    of four-threats where the defender's reply is forced — finds most real
    Gomoku forced wins that a 1-ply scorer structurally cannot see.
  * Defense multiplier 3/2 per eval_params.hpp:37-38.

Weights (src/eval_params.hpp:30-38):
  Five=1e6, OpenFour=1e5, ClosedFour=1e4, OpenThree=5e3, ClosedThree=1e3,
  OpenTwo=5e2, ClosedTwo=1e2; Fork44=9e4, Fork43=6e4, Fork33=3e4.

Usage (headless):
    python online_bot_heuristic2.py --url http://host:8765 --code ABC123
Via GUI: client.py -> Client -> select this file as the bot.
"""
from __future__ import annotations

import argparse
import time

try:
    from client import BOARD_SIZE, is_outer, run_bot
except ImportError:
    from gomoku_online import BOARD_SIZE, is_outer, run_bot

BOT_NAME = "Heuristic-Full-v2"

N = BOARD_SIZE
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))

NONE, CLOSED_TWO, OPEN_TWO, CLOSED_THREE, OPEN_THREE, CLOSED_FOUR, OPEN_FOUR, FIVE = range(8)

WEIGHT = {FIVE: 1_000_000, OPEN_FOUR: 100_000, CLOSED_FOUR: 10_000,
          OPEN_THREE: 5_000, CLOSED_THREE: 1_000,
          OPEN_TWO: 500, CLOSED_TWO: 100, NONE: 0}
FORK_44, FORK_43, FORK_33 = 90_000, 60_000, 30_000
DEF_NUM, DEF_DEN = 3, 2  # defense weighting (own + opp*3/2)
POS_W, TEMPO = 6, 250   # eval_v2 centre proximity and tempo terms

VCF_DEPTH = 6


def _in(r, c):
    return 0 <= r < N and 0 <= c < N


# ---------------------------------------------------------------------------
# M-mask line table (port of src/eval.cpp classifyLine + isJumpFour)
# ---------------------------------------------------------------------------
def _cell_state(own8, opp8, off):
    """off in [-4, 4]; center 0 counts as own. Both bits set = off-board."""
    if off == 0:
        return 1
    bit = off + 4 if off < 0 else off + 3
    if ((own8 >> bit) & 1) and ((opp8 >> bit) & 1):
        return 2
    if (own8 >> bit) & 1:
        return 1
    if (opp8 >> bit) & 1:
        return 2
    return 0


def _is_jump_four(own8, opp8):
    for st in range(-4, 1):
        own = opp = 0
        for k in range(5):
            s = _cell_state(own8, opp8, st + k)
            own += s == 1
            opp += s == 2
        if opp == 0 and own >= 4:
            return True
    return False


def _classify_line(own8, opp8):
    """ThreatLevel for the line through a just-played center move."""
    left = 0
    while _cell_state(own8, opp8, -(left + 1)) == 1:
        left += 1
        if left == 4:
            break
    right = 0
    while _cell_state(own8, opp8, right + 1) == 1:
        right += 1
        if right == 4:
            break
    if 1 + left + right >= 5:
        return FIVE
    lopen = _cell_state(own8, opp8, -(left + 1)) == 0
    ropen = _cell_state(own8, opp8, right + 1) == 0
    run = 1 + left + right
    if run == 4:
        return OPEN_FOUR if (lopen and ropen) else CLOSED_FOUR
    if run == 3:
        return OPEN_THREE if (lopen and ropen) else CLOSED_THREE
    if run == 2:
        return OPEN_TWO if (lopen and ropen) else CLOSED_TWO
    lvl = NONE
    if _is_jump_four(own8, opp8):
        lvl = CLOSED_FOUR  # gap four completes five: a four threat
    return lvl


def _build_table():
    t = bytearray(65536)
    for f in range(65536):
        t[f] = _classify_line(f >> 8, f & 0xFF)
    return bytes(t)


_LINE_TABLE = _build_table()


def _line_masks(board, r, c, dr, dc, own, opp):
    """16-bit feature (own8<<8)|opp8 for the 8 cells around (r,c) with (r,c)
    counted as `own`. Off-board cells set both bits (blocked-for-both)."""
    own8 = opp8 = 0
    for off in range(-4, 5):
        if off == 0:
            continue
        rr, cc = r + dr * off, c + dc * off
        bit = off + 4 if off < 0 else off + 3
        if not _in(rr, cc):
            own8 |= 1 << bit
            opp8 |= 1 << bit
        elif board[rr][cc] == own:
            own8 |= 1 << bit
        elif board[rr][cc] == opp:
            opp8 |= 1 << bit
    return (own8 << 8) | opp8


def _threats_after(board, r, c, player):
    """(max_level, multiset of levels) the move creates for `player`."""
    opp = 3 - player
    levels = []
    for dr, dc in DIRS:
        f = _line_masks(board, r, c, dr, dc, player, opp)
        lvl = _LINE_TABLE[f]
        if lvl:
            levels.append(lvl)
    levels.sort(reverse=True)
    return levels


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------
def candidates(board, radius=1):
    occ = [(r, c) for r in range(N) for c in range(N) if board[r][c]]
    if not occ:
        return [(r, c) for r in range(N) for c in range(N) if is_outer(r, c)]
    out = set()
    for r, c in occ:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                p = (r + dr, c + dc)
                if _in(*p) and board[p[0]][p[1]] == 0:
                    out.add(p)
    return sorted(out)


def five_completes(board, r, c, player):
    """Playing (r,c) for `player` makes >= 5 in a row."""
    for dr, dc in DIRS:
        cnt = 1
        for s in (1, -1):
            rr, cc = r + dr * s, c + dc * s
            while _in(rr, cc) and board[rr][cc] == player:
                cnt += 1
                rr += dr * s
                cc += dc * s
        if cnt >= 5:
            return True
    return False


def winning_squares(board, player, cands=None):
    cands = cands if cands is not None else candidates(board)
    return [m for m in cands if five_completes(board, *m, player)]


# ---------------------------------------------------------------------------
# eval_v2 fork analysis: per-direction span-deduped threat counts
# ---------------------------------------------------------------------------
def _line_coords(r, c, dr, dc):
    """Full line through (r,c) as coordinates; (r,c) itself included."""
    pts = []
    rr, cc = r - dr * 9, c - dc * 9
    while not _in(rr, cc):
        rr += dr
        cc += dc
    while _in(rr, cc):
        pts.append((rr, cc))
        rr += dr
        cc += dc
    return pts


def fork_score(board, r, c, player):
    """(four_threats, open_threes, closed_threes) the move creates, with
    span dedup: _XXXX_ is ONE four, one open three spanning several windows
    is ONE three (src/eval_v2.cpp:128-211)."""
    fours = []
    othrees = []
    cthrees = []
    for dr, dc in DIRS:
        line = _line_coords(r, c, dr, dc)
        if len(line) < 5:
            continue
        # cell states: 1 own (incl. the new stone), 2 opp, 0 empty
        st = []
        for (rr, cc) in line:
            v = board[rr][cc]
            st.append(1 if (rr, cc) == (r, c) or v == player else (2 if v else 0))
        n = len(st)
        # four threats: 5-window of 4 own + 1 empty, dedup by stone span
        seen_spans = set()
        nf = 0
        for p in range(n - 4):
            own = opp = 0
            lo, hi = -1, -1
            empt = -1
            for k in range(5):
                s = st[p + k]
                if s == 1:
                    own += 1
                    if lo < 0:
                        lo = p + k
                    hi = p + k
                elif s == 2:
                    opp += 1
                else:
                    empt = p + k
            if opp or own != 4:
                continue
            if (lo, hi) in seen_spans:
                continue
            seen_spans.add((lo, hi))
            nf += 1
        fours.append(nf)
        # open threes: 5-window of 3 own + 2 empty, no opp, where filling one
        # empty yields a CONTIGUOUS four flanked by empties on both ends
        seen3 = set()
        n3 = 0
        nc3 = 0
        for p in range(n - 4):
            own = opp = 0
            empties = []
            lo, hi = -1, -1
            for k in range(5):
                s = st[p + k]
                if s == 1:
                    own += 1
                    if lo < 0:
                        lo = p + k
                    hi = p + k
                elif s == 2:
                    opp += 1
                else:
                    empties.append(p + k)
            if opp or own != 3:
                continue
            if (lo, hi) in seen3:
                continue
            makes_open_four = False
            for e in empties:
                flo, fhi = min(lo, e), max(hi, e)
                if fhi - flo != 3:
                    continue
                contiguous = all(st[k] == 1 or k == e for k in range(flo, fhi + 1))
                if not contiguous:
                    continue
                L = flo - 1 >= 0 and st[flo - 1] == 0
                R = fhi + 1 < n and st[fhi + 1] == 0
                if L and R:
                    makes_open_four = True
            if makes_open_four:
                seen3.add((lo, hi))
                n3 += 1
            else:
                # closed three only if it can still become a four somewhere
                seen3.add((lo, hi))
                nc3 += 1
        othrees.append(n3)
        cthrees.append(nc3)
    return sum(fours), sum(othrees), sum(cthrees)


# ---------------------------------------------------------------------------
# Move evaluation: table levels + fork bonuses + defense + position
# ---------------------------------------------------------------------------
def move_score(board, r, c, player):
    """Full heuristic score of playing (r,c) as `player` (src/eval_v2-style)."""
    own_levels = _threats_after(board, r, c, player)
    s = sum(WEIGHT[l] for l in own_levels)
    # opponent view: threats we PREEMPT by occupying this square
    opp = 3 - player
    opp_levels = _threats_after(board, r, c, opp)
    s += sum(WEIGHT[l] for l in opp_levels) * DEF_NUM // DEF_DEN

    f4, o3, c3 = fork_score(board, r, c, player)
    if f4 >= 2:
        s += FORK_44
    elif f4 >= 1 and o3 >= 1:
        s += FORK_43
    elif o3 >= 2:
        s += FORK_33
    # defence forks: deny opponent's fork square
    of4, oo3, oc3 = fork_score(board, r, c, opp)
    if of4 >= 2 or (of4 >= 1 and oo3 >= 1) or (oo3 >= 2 and of4 >= 1):
        s += FORK_44 * DEF_NUM // DEF_DEN
    elif oo3 >= 2:
        s += FORK_33 * DEF_NUM // DEF_DEN

    s += POS_W * (max(0, 7 - max(abs(r - 7), abs(c - 7))))
    return s


def rank_moves(board, player, cands):
    return sorted(cands, key=lambda m: move_score(board, m[0], m[1], player), reverse=True)


# ---------------------------------------------------------------------------
# Shallow VCF probe (Victory by Continuous Fours, defender reply forced)
# ---------------------------------------------------------------------------
def vcf_win(board, player, attacker_moves, depth):
    """True if `player` can force a win with a chain of four-threat moves.
    fours(board, player) yields moves that create a five-threat (four+1 gap)."""
    if depth == 0:
        return False
    opp = 3 - player
    for m in candidates(board, 2):
        if not four_creating(board, m, player):
            continue
        if five_completes(board, *m, player):
            return True  # direct five
        r, c = m
        board[r][c] = player
        gain = winning_squares(board, player)
        if len(gain) == 0:
            board[r][c] = 0
            continue
        if len(gain) >= 2:
            # double four: opponent can block at most one
            board[r][c] = 0
            return True
        gr, gc = gain[0]
        board[gr][gc] = opp  # forced block
        won = vcf_win(board, player, attacker_moves + [m, gain[0]], depth - 1)
        board[gr][gc] = 0
        board[r][c] = 0
        if won:
            return True
    return False


def four_creating(board, m, player):
    r, c = m
    for dr, dc in DIRS:
        f = _line_masks(board, r, c, dr, dc, player, 3 - player)
        if _LINE_TABLE[f] >= CLOSED_FOUR:
            return True
    return False


# ---------------------------------------------------------------------------
# Top-level move choice
# ---------------------------------------------------------------------------
def choose_move(board, player: int, info: dict) -> tuple[int, int]:
    opp = 3 - player
    all_cands = candidates(board)

    # Level-1 restriction (provably lossless, search_threatcand.cpp:172-221):
    # if either side has a five-completing square, only fives+blocks matter.
    my_wins = [m for m in all_cands if five_completes(board, *m, player)]
    if my_wins:
        return my_wins[0]
    opp_wins = [m for m in all_cands if five_completes(board, *m, opp)]
    if opp_wins:
        # must block (multiple opp fives = lost, block the highest-scoring one)
        return max(opp_wins, key=lambda m: move_score(board, m[0], m[1], player))

    cands = all_cands
    # Level-2 restriction: opponent open-four-maker must be denied
    opp_o4 = [m for m in cands
              if any(l >= OPEN_FOUR for l in _threats_after(board, m[0], m[1], opp))]
    if opp_o4:
        pool = set(opp_o4)
        # our counter-fours and open-four-makers are also relevant
        for m in cands:
            lv = _threats_after(board, m[0], m[1], player)
            if any(l >= CLOSED_FOUR for l in lv):
                pool.add(m)
        cands = sorted(pool)

    ranked = rank_moves(board, player, cands)

    # VCF probe on the top few candidates: forced four-chains win now.
    if info["ply"] > 1:
        for m in ranked[:6]:
            r, c = m
            board[r][c] = player
            gain = winning_squares(board, player)
            if len(gain) >= 2:
                board[r][c] = 0
                return m
            if gain:
                gr, gc = gain[0]
                board[gr][gc] = opp
                # after forced block, do we still have another four?
                for m2 in candidates(board, 2)[:24]:
                    if five_completes(board, *m2, player):
                        board[gr][gc] = 0; board[r][c] = 0
                        return m
                    if four_creating(board, m2, player):
                        board[gr][gc] = 0; board[r][c] = 0
                        board[r][c] = player
                        ok = vcf_win(board, player, [], VCF_DEPTH)
                        board[r][c] = 0
                        if ok:
                            return m
                        break
                board[gr][gc] = 0
            board[r][c] = 0

    return ranked[0]


def load_bot(room) -> None:
    room.ready()  # table precomputed at import; nothing to load


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy bot heuristic v2 không giao diện")
    parser.add_argument("--url", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--name", default=BOT_NAME)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_bot(args.url, args.code, args.name, choose_move, margin=args.margin, once=args.once, load_bot=load_bot)


if __name__ == "__main__":
    main()
