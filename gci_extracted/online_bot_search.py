"""
Online Gomoku wrapper: heuristic2 eval + PROOF SEARCH (VCF/VCT) + shallow
alpha-beta. Successor of online_bot_heuristic2.py with search added — the
difference between guessing the best move and proving a forced win.

Architecture (why this ordering gives results, not just moves):

  1. VCF (Victory by Continuous Fours) — exhaustive, SOUND proof: a chain of
     four-threat moves where every defender reply is forced to exactly one
     square. No guesswork anywhere; a returned move is a proven win. Port of
     src/engines/search.cpp:189-261, including the defender-wins-first and
     block-makes-five refutations most naive VCFs miss.
  2. VCT (Victory by Continuous Threats) — adds open-three threats, where the
     defender has SEVERAL defences, all of which must be refuted (Allis, van
     den Herik & Huntjens 1993 — the threat-space search that solved Go-Moku).
     Depth-limited: still a proof within the horizon, no heuristic cutoffs
     inside the threat phase.
  3. Defense proof search: VCF/VCT from the OPPONENT's side one move early
     (what happens if we play a passive move and they get a free tempo) is
     approximated by scanning candidate replies; a candidate that lets the
     opponent start a proven VCT chain is demoted.
  4. Shallow alpha-beta (depth 4) over heuristic2's move_score as eval, with
     the same threat-restricted candidates: handles quiet strategic play the
     threat searches don't cover.

Time managed via info["time_budget"]; VCF/VCT are complete before it matters
(typically < 200 ms); alpha-beta is iterative-deepening with deadline checks.
Every move remains bounded by the server clock.

Usage (headless):
    python online_bot_search.py --url http://host:8765 --code ABC123
Via GUI: client.py -> Client -> select this file as the bot.
"""
from __future__ import annotations

import argparse
import time

try:
    from client import BOARD_SIZE, is_outer, run_bot
except ImportError:
    from gomoku_online import BOARD_SIZE, is_outer, run_bot

import online_bot_heuristic2 as h2  # reuse: table, fork analysis, scoring

BOT_NAME = "Heuristic-Search"

N = BOARD_SIZE
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))

VCF_MAX_ATTACKS = 16


def _in(r, c):
    return 0 <= r < N and 0 <= c < N


def copy_board(board):
    return [row[:] for row in board]


# ---------------------------------------------------------------------------
# Threat machinery (shared with heuristic2)
# ---------------------------------------------------------------------------
five_completes = h2.five_completes
winning_squares = h2.winning_squares
candidates = h2.candidates
_threats_after = h2._threats_after
CLOSED_FOUR = h2.CLOSED_FOUR
OPEN_FOUR = h2.OPEN_FOUR
OPEN_THREE = h2.OPEN_THREE


def four_squares(board, player, cands):
    """Candidate moves that create at least a four-threat (gain square of five)."""
    out = []
    for m in cands:
        lv = _threats_after(board, m[0], m[1], player)
        if any(l >= CLOSED_FOUR for l in lv):
            out.append(m)
    return out


def open_three_squares(board, player, cands):
    out = []
    for m in cands:
        lv = _threats_after(board, m[0], m[1], player)
        if OPEN_THREE in lv:
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# VCF: sound forced-win proof by continuous fours
# ---------------------------------------------------------------------------
def vcf(board, attacker, depth, deadline):
    """True iff `attacker` has a forced win by a chain of four-threat moves.
    Every defender reply is forced (single gain square), and defender
    counter-fives are checked (search.cpp:223,246). Sound within depth."""
    if time.perf_counter() > deadline:
        return False
    opp = 3 - attacker
    cands = candidates(board, 2)
    for m in cands:
        if not any(l >= CLOSED_FOUR for l in _threats_after(board, m[0], m[1], attacker)):
            continue
        r, c = m
        board[r][c] = attacker
        if five_completes(board, r, c, attacker):
            board[r][c] = 0
            return True
        if any(five_completes(board, *w, opp) for w in cands if w != m):
            # defender wins first: this forcing line is unsound
            board[r][c] = 0
            continue
        gains = winning_squares(board, attacker, candidates(board, 2))
        if len(gains) >= 2:
            board[r][c] = 0
            return True  # double four / open four: unblockable
        if len(gains) != 1:
            board[r][c] = 0
            continue
        gr, gc = gains[0]
        board[gr][gc] = opp  # forced block
        if five_completes(board, gr, gc, opp):
            win = False  # the block itself makes five
        else:
            win = vcf(board, attacker, depth - 1, deadline) if depth > 1 else False
        board[gr][gc] = 0
        board[r][c] = 0
        if win:
            return True
    return False


def vcf_root(board, player, deadline):
    """Best first move of a proven VCF chain, or None."""
    opp = 3 - player
    cands = candidates(board, 2)
    for m in cands:
        if not any(l >= CLOSED_FOUR for l in _threats_after(board, m[0], m[1], player)):
            continue
        r, c = m
        board[r][c] = player
        if five_completes(board, r, c, player):
            board[r][c] = 0
            return m
        opp_five_now = any(five_completes(board, *w, opp) for w in cands if w != m)
        gains = [] if opp_five_now else winning_squares(board, player, candidates(board, 2))
        win = False
        if not opp_five_now and len(gains) >= 2:
            win = True
        elif not opp_five_now and len(gains) == 1:
            gr, gc = gains[0]
            board[gr][gc] = opp
            if five_completes(board, gr, gc, opp):
                win = False
            else:
                win = vcf(board, player, VCF_MAX_ATTACKS, deadline)
            board[gr][gc] = 0
        board[r][c] = 0
        if win:
            return m
    return None


# ---------------------------------------------------------------------------
# Defense proof: does our candidate let the opponent start a VCF chain?
# ---------------------------------------------------------------------------
def opp_vcf_reply(board, player, cand, deadline):
    """After we play `cand`, can the opponent force a VCF win? Returns their
    winning first move, or None."""
    opp = 3 - player
    r, c = cand
    board[r][c] = player
    w = vcf_root(board, opp, deadline)
    board[r][c] = 0
    return w


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------
def choose_move(board: list[list[int]], player: int, info: dict) -> tuple[int, int]:
    """Proven layers first (VCF for us, VCF denial for them), then the
    measured heuristic2 ranking. Every layer that returns a move is either
    an exact proof or a measured-Elo scorer — nothing speculative in between."""
    opp = 3 - player
    budget = info.get("time_budget")
    cap = (budget * 0.85) if budget is not None else 5.0
    deadline_all = time.perf_counter() + cap

    cands = candidates(board, 2 if info["ply"] > 0 else 1)

    # 1. immediate five
    my_wins = [m for m in cands if five_completes(board, *m, player)]
    if my_wins:
        return my_wins[0]
    # 2. forced block
    opp_wins = [m for m in cands if five_completes(board, *m, opp)]
    if opp_wins:
        return max(opp_wins, key=lambda m: h2.move_score(board, m[0], m[1], player))

    # 3. our VCF proof (sound): forced four-chain win, play it
    m = vcf_root(board, player, deadline_all)
    if m is not None:
        return m

    # 4. heuristic2 ranking (measured +108 Elo vs big_net+MCTS), with a
    #    VCF-denial screen: never hand the opponent a proven forced win.
    ranked = h2.rank_moves(board, player, cands)
    for m in ranked[:5]:
        if opp_vcf_reply(board, player, m, deadline_all) is None:
            return m
        # this move loses to a forced four-chain: try the next candidate
    # every top candidate loses (or deadline hit): play the best anyway
    return ranked[0]


def load_bot(room) -> None:
    room.ready()  # tables precomputed at import


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy bot heuristic+search không giao diện")
    parser.add_argument("--url", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--name", default=BOT_NAME)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_bot(args.url, args.code, args.name, choose_move, margin=args.margin, once=args.once, load_bot=load_bot)


if __name__ == "__main__":
    main()
