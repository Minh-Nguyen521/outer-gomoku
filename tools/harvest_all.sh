#!/usr/bin/env bash
# Harvest training positions from games between the strong engines.
#
# WHY: the existing data/labeled_50k.jsonl is 80.2% already-decided (teacher
# value >0.9 or <0.1) and only 3.6% balanced, because tools/gen_positions.py
# played greedy/random moves and produced blowouts. A value head trained on
# that learns to emit 0 or 1 and gets almost no gradient where an evaluation
# function actually earns its keep. Games between engines that can play
# produce the opposite distribution.
#
# PAIRINGS: chosen for CONTESTED games, not for the strongest players. A
# lopsided matchup yields the same decided positions the old generator did.
# Measured head-to-head results against rapfi this session:
#     SKYZERO   12-8   (60%)  <- most balanced pairing we have
#     VIBEFIVE  17-3   (85%)
#     SlowRenju 18-2   (90%)
#     PentaZen   0-20  (0%)
# So rapfi vs SKYZERO is the best single source; SlowRenju vs VIBEFIVE pits
# two engines that both beat rapfi against each other; PentaZen and Rapfi-2018
# widen the style range at the weaker end.
#
# SAMPLING: --every 1, NOT 2. With a stride of 2 from an even min-ply, every
# sampled position has the same side to move (stm is derived from ply parity),
# so a whole harvest carries one colour only -- a dry run produced 17 positions
# all labelled +1. That teaches a colour prior rather than an evaluation. A
# stride of 1 samples both sides and doubles the yield.
#
# SPEED: 1s/move, not 5s. Both engines still play well (rapfi is strong at 1s)
# and games finish in ~1 minute, which is what makes tens of thousands of
# positions reachable overnight rather than in a week. Native engines are used
# for the bulk; the Wine-hosted ones (SKYZERO, VIBEFIVE) are slower per game
# because each spawns a container, so they get smaller allocations.
#
# PARALLELISM: harvest_games.py plays serially, so this runs several
# processes, each with its OWN output file. Appending from multiple processes
# to one file risks interleaved partial lines; separate files then `cat` is
# safe and costs nothing.
#
# Usage:
#   tools/harvest_all.sh [GAMES_PER_PAIR]      # default 60
#   cat data/harvest/*.jsonl > data/harvest_all.jsonl

set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
N="${1:-60}"
OUT=data/harvest
mkdir -p "$OUT"

TURN=1000
MATCH=300000

RAPFI=third_party/rapfi-bin/pbrain-rapfi
SLOW=third_party/SlowRenju/pbrain-slowrenju
PENTA=third_party/PentaZen/build/src/PentaZen
R2018=third_party/Rapfi-gomocup/pbrain-rapfi2018
CHIS=third_party/Chis/Chis2.0/pbrain-chis
WINE=third_party/wine-engine

harvest() {   # tag, engine-a, engine-b, games
    local tag="$1" a="$2" b="$3" g="$4"
    [ -x "${a%% *}" ] || [ -x "$WINE" ] || { echo "  skip $tag (missing $a)"; return; }
    echo "  starting $tag ($g games)"
    python3 tools/harvest_games.py --engine-a "$a" --engine-b "$b" \
        --games "$g" --every 1 --min-ply 6 --skip-last 4 \
        --timeout-turn "$TURN" --timeout-match "$MATCH" \
        --out "$OUT/$tag.jsonl" > "$OUT/$tag.log" 2>&1 &
}

# PAIRING CHOICE IS ABOUT GAME LENGTH, NOT WIN BALANCE.
# A first run made this mistake. Measured yield per pairing:
#     slow_rapfi   11 games -> 220 positions  (20/game)
#     penta_rapfi  21 games ->  30 positions  (1.4/game)
#     r2018_chis    4 games ->   6 positions
#     slow_penta   22 games ->   0 positions  <- zero, from 22 games
# PentaZen and Rapfi-2018 lose FAST, so after skipping the opening (min-ply)
# and the decided endgame (skip-last) there is nothing left in the middle.
# Only closely matched engines play the long, contested games that carry
# training signal. Dropped both weak engines; kept the pairs whose measured
# head-to-heads are close:
#     Chis vs rapfi       12-8   (60%)
#     SlowRenju vs rapfi  18-2, but games run long -> best yield observed
harvest slow_rapfi  "$SLOW" "$RAPFI" "$N"
harvest chis_rapfi  "$CHIS" "$RAPFI" "$N"
harvest slow_chis   "$SLOW" "$CHIS"  "$N"
wait
echo "  native pairs done"

# SKYZERO is the most balanced opponent rapfi has here (12-8) and would give
# excellent positions, but every game spawns a Wine container and it is far
# slower per move than the native engines. Run it last, alone, and only for a
# small share of the total.
harvest skyzero_rapfi "$WINE skyzero pbrain-SkyZero.exe" "$RAPFI" $((N / 4))
wait

echo "=== totals ==="
wc -l "$OUT"/*.jsonl 2>/dev/null | tail -8
