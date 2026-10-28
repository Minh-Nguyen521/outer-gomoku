#include "search_v5.hpp"

#include <algorithm>
#include <cmath>

#include "book.hpp"

namespace oog {

namespace {

int adjustToStore(int score, int ply) {
    if (score >= MATE_THRESHOLD) return score + ply;
    if (score <= -MATE_THRESHOLD) return score - ply;
    return score;
}

int adjustFromStore(int score, int ply) {
    if (score >= MATE_THRESHOLD) return score - ply;
    if (score <= -MATE_THRESHOLD) return score + ply;
    return score;
}

constexpr int LMR_TABLE_N = 64;
std::array<std::array<float, LMR_TABLE_N>, LMR_TABLE_N> buildReductionLUT() {
    std::array<std::array<float, LMR_TABLE_N>, LMR_TABLE_N> lut{};
    for (int d = 0; d < LMR_TABLE_N; ++d) {
        for (int mc = 0; mc < LMR_TABLE_N; ++mc) {
            lut[static_cast<std::size_t>(d)][static_cast<std::size_t>(mc)] =
                std::log(static_cast<float>(std::max(d, 1))) *
                std::log(static_cast<float>(std::max(mc, 1)));
        }
    }
    return lut;
}
const auto REDUCTION_LUT = buildReductionLUT();

// Untuned first-guess margins in this project's own evaluate() score units
// (S_CLOSED_TWO=100 .. S_FIVE=1'000'000, see eval.cpp). Motivated by rapfi's
// razorMargin/futilityMargin shape (small at shallow depth, growing with
// depth), not ported numbers -- real values belong in Phase 7 tuning.
// Four-threat extension: OFF, and measured, not assumed.
//
// The idea is sound -- a move creating a four forces the reply, so it is
// worth a ply. The *trigger* was not: it used `evaluateMove()`'s score as a
// cheap proxy for "this move makes a four", but that score combines threats
// the move creates for us with threats it pre-empts for the opponent, so an
// ordinary blocking move clears the ClosedFour threshold too. The extension
// fired almost everywhere and inflated the tree about 4x per depth level.
//
// Measured at 200 ms/move on the three fixed positions, ON vs OFF:
//   depth 5/6/4  ->  8/8/7   (SearchV4, for reference: 8/9/7)
// with near-identical node counts -- i.e. the same budget was buying far
// less real depth.
//
// Re-enabling this needs an *exact* trigger (after placing the move, test
// whether fivePointsMask() is non-empty for the mover), not a score proxy.
// That costs a mask per candidate, so it must be gated to stay affordable.
// The exact forced-reply extension below is unaffected and stays on.
constexpr bool EXT_FOUR_THREAT = false;

int razorMargin(int depth) { return 800 + 300 * depth; }
int futilityMargin(int depth, bool improving) {
    return std::max(700 * depth - (improving ? 300 : 0), 0);
}

// --- Bitboard-parallel five-point detection ------------------------------
// The primitive Round 5 was really measuring. See search_v5.hpp.

constexpr int MW = BitBoard::WORDS;  // 5 words covering the 272-bit grid
using Mask = std::array<std::uint64_t, MW>;

// out = in << n over the little-endian word layout.
void shl(Mask& out, const Mask& in, int n) {
    const int w = n >> 6;
    const int b = n & 63;
    for (int i = MW - 1; i >= 0; --i) {
        std::uint64_t v = 0;
        if (i - w >= 0) {
            v = in[static_cast<std::size_t>(i - w)] << b;
            if (b > 0 && i - w - 1 >= 0) {
                v |= in[static_cast<std::size_t>(i - w - 1)] >> (64 - b);
            }
        }
        out[static_cast<std::size_t>(i)] = v;
    }
}

// out = in >> n.
void shr(Mask& out, const Mask& in, int n) {
    const int w = n >> 6;
    const int b = n & 63;
    for (int i = 0; i < MW; ++i) {
        std::uint64_t v = 0;
        if (i + w < MW) {
            v = in[static_cast<std::size_t>(i + w)] >> b;
            if (b > 0 && i + w + 1 < MW) {
                v |= in[static_cast<std::size_t>(i + w + 1)] << (64 - b);
            }
        }
        out[static_cast<std::size_t>(i)] = v;
    }
}

// Playable squares only (rows 1..15, cols 0..14). Required: the sentinel
// column and pad rows read as "empty", so without this a five-window could
// nominate a sentinel cell as its gap.  Sentinels can never be *stones*, so
// a window that merely spans one is already rejected by the AND below.
const Mask PLAYABLE = [] {
    Mask m{};
    for (int r = 1; r <= 15; ++r) {
        for (int c = 0; c <= 14; ++c) {
            const int sq = r * BitBoard::ROW_W + c;
            m[static_cast<std::size_t>(sq >> 6)] |= 1ULL << (sq & 63);
        }
    }
    return m;
}();

// All squares where `color` would complete five, computed board-wide.
//
// A square s wins for `color` exactly when some 5-in-a-row window through s
// holds four `color` stones and has s as its only gap.  For each direction d
// and each position k the gap can take within the window, that is: s empty,
// and `color` present at s + t*d for every t in -k..4-k except t = 0.
void fivePointsMask(const BitBoard& bb, int color, Mask& out) {
    const Mask& own = bb.bits(color);
    const Mask& opp = bb.bits(color ^ 1);
    Mask empty;
    for (int i = 0; i < MW; ++i) {
        empty[static_cast<std::size_t>(i)] =
            PLAYABLE[static_cast<std::size_t>(i)] &
            ~(own[static_cast<std::size_t>(i)] | opp[static_cast<std::size_t>(i)]);
    }
    out.fill(0);

    constexpr int DIRS[4] = {1, BitBoard::ROW_W, BitBoard::ROW_W + 1,
                             BitBoard::ROW_W - 1};
    for (const int d : DIRS) {
        // sh[t + 4] has bit s set iff `own` has a stone at s + t*d.
        Mask sh[9];
        for (int t = -4; t <= 4; ++t) {
            if (t == 0) continue;
            if (t > 0) {
                shr(sh[static_cast<std::size_t>(t + 4)], own, t * d);
            } else {
                shl(sh[static_cast<std::size_t>(t + 4)], own, -t * d);
            }
        }
        for (int k = 0; k <= 4; ++k) {
            Mask acc = empty;
            for (int j = 0; j <= 4; ++j) {
                if (j == k) continue;
                const Mask& s = sh[static_cast<std::size_t>(j - k + 4)];
                for (int i = 0; i < MW; ++i) {
                    acc[static_cast<std::size_t>(i)] &= s[static_cast<std::size_t>(i)];
                }
            }
            for (int i = 0; i < MW; ++i) {
                out[static_cast<std::size_t>(i)] |= acc[static_cast<std::size_t>(i)];
            }
        }
    }
}

bool maskAny(const Mask& m) {
    for (int i = 0; i < MW; ++i) {
        if (m[static_cast<std::size_t>(i)] != 0) return true;
    }
    return false;
}

int maskCount(const Mask& m) {
    int c = 0;
    for (int i = 0; i < MW; ++i) {
        c += __builtin_popcountll(m[static_cast<std::size_t>(i)]);
    }
    return c;
}

bool maskTest(const Mask& m, int sq) {
    return (m[static_cast<std::size_t>(sq >> 6)] >> (sq & 63)) & 1;
}

void maskSquares(const Mask& m, std::vector<int>& out) {
    out.clear();
    for (int w = 0; w < MW; ++w) {
        std::uint64_t bits = m[static_cast<std::size_t>(w)];
        while (bits) {
            const int k = __builtin_ctzll(bits);
            bits &= bits - 1;
            out.push_back(w * 64 + k);
        }
    }
}

}  // namespace

void fivePointSquaresV5(const BitBoard& bb, int color, std::vector<int>& out) {
    Mask m;
    fivePointsMask(bb, color, m);
    maskSquares(m, out);
}

SearchV5::SearchV5()
    : history_(2 * BitBoard::SIZE, 0), counterMove_(2 * BitBoard::SIZE, -1) {
    for (auto& k : killers_) k = {-1, -1};
    rootUnsafe_.fill(false);
}

std::uint64_t SearchV5::keyWithSide(int color) const {
    return positionKey(*bb_, color);
}

bool SearchV5::outOfTime() {
    if (limits_.maxTimeMs > 0) {
        const auto now = std::chrono::steady_clock::now();
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - t0_);
        if (ms.count() >= limits_.maxTimeMs) {
            aborted_ = true;
            return true;
        }
    }
    if (limits_.maxNodes > 0 && nodes_ >= limits_.maxNodes) {
        aborted_ = true;
        return true;
    }
    return false;
}

void SearchV5::orderMoves(std::vector<int>& moves, int color, int ttMove,
                          int depth, int ply) {
    struct Scored {
        int sq;
        long long sc;
        int threat;
    };
    const bool cheap = depth <= 1;
    std::vector<Scored> v;
    v.reserve(moves.size());
    for (const int m : moves) {
        int threat = 0;
        long long sc = 0;
        if (!cheap) {
            threat = evaluateMove(*bb_, color, m);
            sc = static_cast<long long>(threat) * 10000;
        }
        sc += history_[static_cast<std::size_t>(color) * BitBoard::SIZE +
                       static_cast<std::size_t>(m)];
        if (ply == 0) sc += bookBonus_[static_cast<std::size_t>(m)];
        // Killers rank below the TT move but above history-ordered quiets:
        // try a sibling-local refutation before global statistics.
        if (ply < MAX_PLY) {
            const auto& k = killers_[static_cast<std::size_t>(ply)];
            if (m == k[0]) sc += 1LL << 38;
            else if (m == k[1]) sc += 1LL << 37;
        }
        if (lastSq_ >= 0 &&
            counterMove_[static_cast<std::size_t>(lastColor_) * BitBoard::SIZE +
                         static_cast<std::size_t>(lastSq_)] == m) {
            sc += 1LL << 36;
        }
        if (m == ttMove) sc += 1LL << 40;
        v.push_back({m, sc, threat});
    }
    std::sort(v.begin(), v.end(),
              [](const Scored& a, const Scored& b) { return a.sc > b.sc; });
    moveThreat_.resize(moves.size());
    for (std::size_t i = 0; i < moves.size(); ++i) {
        moves[i] = v[i].sq;
        moveThreat_[i] = v[i].threat;
    }
}

// Both of these keep SearchV3's signatures and semantics exactly; only the
// implementation changes, from a 272-square place/winsAt/remove loop to a
// board-wide shift/AND. They sit on the hot path of vcf(), of razoring and
// of futility pruning, so this alone speeds up the inherited search.
void SearchV5::collectWinningPoints(int color, std::vector<int>& out) {
    Mask m;
    fivePointsMask(*bb_, color, m);
    maskSquares(m, out);
}

bool SearchV5::hasImmediateWin(int color) {
    Mask m;
    fivePointsMask(*bb_, color, m);
    return maskAny(m);
}

// Masks are passed in rather than held as members: negamax recurses, and a
// child node would otherwise overwrite its parent's scratch state.
bool SearchV5::restrictToForcedReplies(const MaskArr& ourFive,
                                       const MaskArr& oppFive,
                                       std::vector<int>& moves) {
    if (!maskAny(oppFive)) return false;

    // The opponent threatens five. Anything that neither makes five for us
    // nor occupies one of their points loses immediately, so the union below
    // is the complete set of non-losing replies -- exact, not a heuristic.
    std::vector<int> out;
    out.reserve(8);
    for (const int m : moves) {
        if (maskTest(oppFive, m) || maskTest(ourFive, m)) out.push_back(m);
    }
    if (out.empty() || out.size() == moves.size()) return false;
    moves.swap(out);
    return true;
}

// Squares in `cand` where `color` would create an open four (two winning
// points) -- the gain squares of `color`'s open threes.
void SearchV5::openFourPointsIn(int color, const std::vector<int>& cand,
                                std::vector<int>& out) {
    out.clear();
    Mask m;
    for (const int s : cand) {
        if (!bb_->isEmpty(s)) continue;
        bb_->place(color, s);
        const bool five = bb_->winsAt(color, s);
        if (!five) {
            fivePointsMask(*bb_, color, m);
            if (maskCount(m) >= 2) out.push_back(s);
        }
        bb_->remove(color, s);
    }
}

// Squares in `cand` where `color` would create at least a four.
void SearchV5::fourPointsIn(int color, const std::vector<int>& cand,
                            std::vector<int>& out) {
    out.clear();
    Mask m;
    for (const int s : cand) {
        if (!bb_->isEmpty(s)) continue;
        bb_->place(color, s);
        if (bb_->winsAt(color, s)) {
            out.push_back(s);
        } else {
            fivePointsMask(*bb_, color, m);
            if (maskAny(m)) out.push_back(s);
        }
        bb_->remove(color, s);
    }
}

int SearchV5::computeLmrReduction(int color, int m, int depth, int ply,
                                  int moveIndex, bool nodeIsPv) const {
    const int d = std::min(depth, LMR_TABLE_N - 1);
    const int mc = std::min(moveIndex, LMR_TABLE_N - 1);
    float r = 0.5f * REDUCTION_LUT[static_cast<std::size_t>(d)]
                                  [static_cast<std::size_t>(mc)];

    if (nodeIsPv) r -= 0.75f;

    if (ply >= 2 &&
        evalAtPly_[static_cast<std::size_t>(ply)] <=
            evalAtPly_[static_cast<std::size_t>(ply - 2)]) {
        r += 0.5f;
    }

    const long long h =
        history_[static_cast<std::size_t>(color) * BitBoard::SIZE +
                 static_cast<std::size_t>(m)];
    if (h > 0) {
        r -= std::min(1.5f, static_cast<float>(h) /
                                (static_cast<float>(depth) * depth * 4.0f + 1.0f));
    }

    int ri = static_cast<int>(r + 0.5f);
    if (ri < 0) ri = 0;
    if (ri > depth - 1) ri = depth - 1;
    return ri;
}

int SearchV5::negamax(int color, int depth, int alpha, int beta, int ply) {
    ++nodes_;
    if ((nodes_ & 0x3FFu) == 0 && outOfTime()) return 0;

    if (lastSq_ >= 0 && bb_->winsAt(lastColor_, lastSq_)) {
        return -(MATE - ply);
    }
    if (bb_->count() >= N * N) return 0;

    // Mate distance pruning: no line from here can beat/lose by more than
    // the ply distance to a mate score, so tighten the window first.
    alpha = std::max(alpha, -(MATE - ply));
    beta = std::min(beta, MATE - ply - 1);
    if (alpha >= beta) return alpha;

    const std::uint64_t key = keyWithSide(color);
    const TT::Entry* e = tt_.probe(key);
    int ttMove = -1;
    if (e != nullptr) {
        ttMove = static_cast<int>(e->move);
        if (e->depth >= depth) {
            const int s = adjustFromStore(e->score, ply);
            if (e->flag == TT::Exact) return s;
            if (e->flag == TT::Lower && s >= beta) return s;
            if (e->flag == TT::Upper && s <= alpha) return s;
        }
    }

    if (ply < MAX_PLY) evalAtPly_[static_cast<std::size_t>(ply)] = eval_.value(color);
    if (depth <= 0) return eval_.value(color);

    const bool nodeIsPv = (beta - alpha) > 1;
    const bool rootNode = (ply == 0);
    const int eval = evalAtPly_[static_cast<std::size_t>(ply)];
    const bool improving =
        ply >= 2 && evalAtPly_[static_cast<std::size_t>(ply)] >
                        evalAtPly_[static_cast<std::size_t>(ply - 2)];

    // The two five-point masks for this node, computed at most once and
    // shared by every gate below plus the forced-reply restriction. The
    // board does not change between them, so SearchV4's up-to-six recomputes
    // were pure waste.
    Mask ourFive;
    Mask oppFive;
    bool masksReady = false;
    auto ensureMasks = [&]() {
        if (!masksReady) {
            fivePointsMask(*bb_, color, ourFive);
            fivePointsMask(*bb_, color ^ 1, oppFive);
            masksReady = true;
        }
    };

    // Node-level pruning: only away from the PV/root, only at shallow
    // remaining depth. The hasImmediateWin() forced-position guard is a full
    // board scan (~272 cells) -- deliberately checked *last*, only once a
    // cheap eval-margin condition has already decided to prune, not on
    // every eligible node.
    if (!nodeIsPv && !rootNode && depth <= 5) {
        // Razoring with VCF: eval is well below alpha -- rather than trust
        // a pessimistic eval outright, verify with the existing forced-win
        // search that there's no hidden win being pruned away.
        if (depth <= 3 && eval < MATE_THRESHOLD &&
            eval + razorMargin(depth) < alpha && (ensureMasks(), !maskAny(oppFive))) {
            if (!vcf(color, 6, ply)) {
                return eval;
            }
            // A forced win exists here; fall through to the normal search.
        }

        // Futility pruning: eval already clears beta by a comfortable
        // margin -- skip the node entirely, no moves generated.
        if (eval < MATE_THRESHOLD && beta < MATE_THRESHOLD &&
            eval - futilityMargin(depth, improving) >= beta &&
            (ensureMasks(), !maskAny(oppFive))) {
            return eval;
        }
    }

    // Null-move forward pruning. Pass the turn and search shallowly: if the
    // position still beats beta after conceding a free move, it beats beta
    // when we actually move. In Go-Moku that implication is exact -- there
    // is no zugzwang, so playing a stone is never worse than passing.  The
    // gate is now two mask computations rather than Round 5's two full-board
    // place/winsAt/remove scans, which is what made this unaffordable there.
    if (!nodeIsPv && !rootNode && depth >= 3 && nullPly_ < 0 &&
        eval >= beta && beta > -MATE_THRESHOLD && beta < MATE_THRESHOLD &&
        (ensureMasks(), !maskAny(ourFive) && !maskAny(oppFive))) {
        const int R = 2 + depth / 4;
        const int prevSq = lastSq_;
        const int prevColor = lastColor_;
        lastSq_ = -1;  // no stone placed, so the child must not win-check
        nullPly_ = ply;
        const int s =
            -negamax(color ^ 1, depth - 1 - R, -beta, -beta + 1, ply + 1);
        nullPly_ = -1;
        lastSq_ = prevSq;
        lastColor_ = prevColor;
        if (aborted_) return 0;
        if (s >= beta) return beta;
    }

    // Internal iterative deepening: ordering quality dominates alpha-beta's
    // node count, so at a high-depth node with no TT move it is worth a
    // cheap reduced-depth search purely to produce one.
    if (ttMove < 0 && !rootNode && depth >= 6) {
        negamax(color, depth - 4, alpha, beta, ply);
        if (aborted_) return 0;
        const TT::Entry* e2 = tt_.probe(key);
        if (e2 != nullptr) ttMove = static_cast<int>(e2->move);
    }

    std::vector<int> moves;
    bb_->genMovesNear(moves);
    if (moves.empty()) {
        for (int s = 0; s < BitBoard::SIZE; ++s) {
            if (BitBoard::isPlayableSq(s) && bb_->isEmpty(s)) moves.push_back(s);
        }
        if (moves.empty()) return 0;
    }
    // Exact forced-reply restriction: when the opponent threatens five, the
    // only non-losing moves are our five-points and theirs.
    ensureMasks();
    const bool restricted = restrictToForcedReplies(ourFive, oppFive, moves);

    // Forced-reply extension: a node with exactly one legal non-losing move
    // is forced, so spending a full ply of budget on it discards the forcing
    // information. Terminates because every extended ply places a stone.
    constexpr int MAX_EXT = 12;
    const int nodeExt =
        (restricted && moves.size() == 1 && extUsed_ < MAX_EXT) ? 1 : 0;

    orderMoves(moves, color, ttMove, depth, ply);
    constexpr std::size_t MAX_CAND = 24;
    // A restricted list is already minimal and every entry in it is forced,
    // so neither the candidate cap nor move-count pruning may discard any.
    if (!restricted && ply > 0 && moves.size() > MAX_CAND) {
        moves.resize(MAX_CAND);
    }

    const int openThreeScore = levelScore(ThreatLevel::OpenThree);
    const int moveCountLimit = 6 + depth * depth;

    int best = -INF;
    int bestMove = -1;
    const int alpha0 = alpha;
    for (std::size_t i = 0; i < moves.size(); ++i) {
        const int m = moves[i];
        const bool isThreat =
            i < moveThreat_.size() && moveThreat_[i] >= openThreeScore;

        // Move-count pruning: once enough moves have been tried at a
        // shallow node without a cutoff, skip the (already threat-ordered)
        // tail of the list instead of searching it -- never skips threats.
        if (!restricted && !rootNode && depth <= 4 && best > -MATE_THRESHOLD &&
            !isThreat && static_cast<int>(i) >= moveCountLimit) {
            continue;
        }

        // Root safety screen: skip moves that hand the opponent a forced
        // VCF win. Stands aside when every candidate is unsafe -- the
        // position is lost anyway and we still owe the caller a move.
        if (rootNode && rootHasSafe_ && rootUnsafe_[static_cast<std::size_t>(m)]) {
            continue;
        }

        // Four-threat extension: a move creating a four forces the reply.
        // Trigger reuses the threat score orderMoves() already computed, so
        // it costs nothing. An extended move is never also LMR-reduced.
        int mext = nodeExt;
        if (EXT_FOUR_THREAT && mext == 0 && extUsed_ < MAX_EXT && depth >= 2 &&
            i < moveThreat_.size() &&
            moveThreat_[i] >= levelScore(ThreatLevel::ClosedFour)) {
            mext = 1;
        }

        const int prevSq = lastSq_;
        const int prevColor = lastColor_;
        lastSq_ = m;
        lastColor_ = color;
        bb_->place(color, m);
        eval_.update(*bb_, m);
        const int nd = depth + mext;
        extUsed_ += mext;
        int s;
        if (i == 0) {
            s = -negamax(color ^ 1, nd - 1, -beta, -alpha, ply + 1);
        } else {
            const bool nodePv = nodeIsPv;
            int red = 0;
            if (!isThreat && mext == 0 && depth >= 3) {
                red = computeLmrReduction(color, m, depth, ply,
                                          static_cast<int>(i), nodePv);
            }
            const int rd = nd - red;
            s = -negamax(color ^ 1, rd - 1, -alpha - 1, -alpha, ply + 1);
            if (s > alpha && s < beta) {
                s = -negamax(color ^ 1, nd - 1, -beta, -alpha, ply + 1);
            }
        }
        extUsed_ -= mext;
        bb_->remove(color, m);
        eval_.update(*bb_, m);
        lastSq_ = prevSq;
        lastColor_ = prevColor;
        if (aborted_) return 0;
        if (s > best) {
            best = s;
            bestMove = m;
        }
        if (s > alpha) alpha = s;
        if (alpha >= beta) {
            history_[static_cast<std::size_t>(color) * BitBoard::SIZE +
                     static_cast<std::size_t>(m)] += depth * depth;
            // lastSq_/lastColor_ are already restored above, so they name
            // the opponent move this cutoff refutes.
            if (ply < MAX_PLY) {
                auto& k = killers_[static_cast<std::size_t>(ply)];
                if (k[0] != m) {
                    k[1] = k[0];
                    k[0] = m;
                }
            }
            if (lastSq_ >= 0) {
                counterMove_[static_cast<std::size_t>(lastColor_) *
                                 BitBoard::SIZE +
                             static_cast<std::size_t>(lastSq_)] = m;
            }
            break;
        }
    }

    if (bestMove < 0) {
        // Every move was move-count-pruned (degenerate at very shallow
        // depth with a large candidate set); fall back to the eval so we
        // never return a bogus -INF.
        return eval;
    }

    if (ply == 0) rootBestMove_ = bestMove;

    const TT::Flag flag =
        best >= beta ? TT::Lower : (best > alpha0 ? TT::Exact : TT::Upper);
    tt_.store(key, depth, flag, adjustToStore(best, ply), bestMove);
    return best;
}

bool SearchV5::vcf(int attacker, int depth, int ply) {
    ++nodes_;
    if ((nodes_ & 0x3FFu) == 0 && outOfTime()) return false;

    if (lastSq_ >= 0 && bb_->winsAt(lastColor_, lastSq_)) {
        return lastColor_ == attacker;
    }
    if (bb_->count() >= N * N || depth <= 0) return false;

    std::vector<int> moves;
    bb_->genMovesNear(moves);
    if (moves.empty()) return false;
    orderMoves(moves, attacker, -1, depth, ply);

    std::vector<int> wp;
    for (const int m : moves) {
        const int prevSq = lastSq_;
        const int prevColor = lastColor_;
        lastSq_ = m;
        lastColor_ = attacker;
        bb_->place(attacker, m);

        if (bb_->winsAt(attacker, m)) {
            bb_->remove(attacker, m);
            lastSq_ = prevSq;
            lastColor_ = prevColor;
            if (ply == 0) vcfFirstMove_ = m;
            return true;
        }

        collectWinningPoints(attacker, wp);
        const bool defenderWinsFirst = hasImmediateWin(attacker ^ 1);
        const bool forcing = !wp.empty();
        if (forcing && !defenderWinsFirst && wp.size() >= 2) {
            bb_->remove(attacker, m);
            lastSq_ = prevSq;
            lastColor_ = prevColor;
            if (ply == 0) vcfFirstMove_ = m;
            return true;
        }
        if (!forcing || defenderWinsFirst || wp.size() != 1) {
            bb_->remove(attacker, m);
            lastSq_ = prevSq;
            lastColor_ = prevColor;
            continue;
        }

        const int w = wp[0];
        bool win = true;
        lastSq_ = w;
        lastColor_ = attacker ^ 1;
        bb_->place(attacker ^ 1, w);
        if (bb_->winsAt(attacker ^ 1, w)) {
            win = false;
        } else {
            win = vcf(attacker, depth - 1, ply + 1);
        }
        bb_->remove(attacker ^ 1, w);
        lastSq_ = prevSq;
        lastColor_ = prevColor;
        bb_->remove(attacker, m);
        if (win) {
            if (ply == 0) vcfFirstMove_ = m;
            return true;
        }
    }
    return false;
}

int SearchV5::vcfWin(BitBoard& bb, int attacker, int maxAttacks) {
    bb_ = &bb;
    lastSq_ = -1;
    lastColor_ = attacker;
    vcfFirstMove_ = -1;
    return vcf(attacker, maxAttacks, 0) ? vcfFirstMove_ : -1;
}

// --- Threat-space search (Allis et al. 1993), VCF extended to VCT ---------

// Cheap precondition for "is a threat search worth starting here at all".
// Round 5's SearchVct ran its VCT every single move and spent up to a third
// of the per-move budget whether or not the position was forcing, which is
// most of why it lost on the clock. A three-threat needs at least an open
// two to build on, so requiring the attacker to already have a move worth an
// open three by the ordering heuristic is a cheap, strictly necessary-ish
// filter -- it costs one pass over the near-move list.
bool SearchV5::worthVct(int color) {
    std::vector<int> cand;
    bb_->genMovesNear(cand);
    if (cand.empty()) return false;
    const int openThree = levelScore(ThreatLevel::OpenThree);
    for (const int m : cand) {
        if (evaluateMove(*bb_, color, m) >= openThree) return true;
    }
    return false;
}

bool SearchV5::vct(int attacker, int depth, int ply) {
    ++nodes_;
    if (vctNodeCap_ > 0 && nodes_ >= vctNodeCap_) return false;
    if ((nodes_ & 0x3FFu) == 0 && outOfTime()) return false;

    if (lastSq_ >= 0 && bb_->winsAt(lastColor_, lastSq_)) {
        return lastColor_ == attacker;
    }
    if (bb_->count() >= N * N || depth <= 0) return false;

    const int defender = attacker ^ 1;

    // Two distinct lists, and they must not be conflated: `all` is the
    // domain for threat scans, `atk` the bounded set of moves tried as
    // attacks. (Conflating them was a real bug in Round 5's SearchVct --
    // winning points outside the top-ranked attacks were silently missed.)
    std::vector<int> all;
    bb_->genMovesNear(all);
    if (all.empty()) return false;

    std::vector<int> atk = all;
    orderMoves(atk, attacker, -1, depth, ply);
    constexpr std::size_t MAX_ATTACK = 12;
    constexpr std::size_t MAX_DEFENCE = 12;
    if (atk.size() > MAX_ATTACK) atk.resize(MAX_ATTACK);

    std::vector<int> wp;
    std::vector<int> gain;
    std::vector<int> counter;
    for (const int m : atk) {
        if (!bb_->isEmpty(m)) continue;
        const int prevSq = lastSq_;
        const int prevColor = lastColor_;
        lastSq_ = m;
        lastColor_ = attacker;
        bb_->place(attacker, m);

        if (bb_->winsAt(attacker, m)) {  // immediate five
            bb_->remove(attacker, m);
            lastSq_ = prevSq;
            lastColor_ = prevColor;
            if (ply == 0) vctFirstMove_ = m;
            return true;
        }

        collectWinningPoints(attacker, wp);
        // A defender who can make five right now simply wins the race, so
        // this line is not actually forcing.
        const bool defenderWinsFirst = hasImmediateWin(defender);

        if (!defenderWinsFirst && wp.size() >= 2) {  // open four: unstoppable
            bb_->remove(attacker, m);
            lastSq_ = prevSq;
            lastColor_ = prevColor;
            if (ply == 0) vctFirstMove_ = m;
            return true;
        }

        bool win = false;
        if (!defenderWinsFirst && wp.size() == 1) {
            // Four-threat: one winning point, so the reply is forced.
            const int w = wp[0];
            lastSq_ = w;
            lastColor_ = defender;
            bb_->place(defender, w);
            win = !bb_->winsAt(defender, w) && vct(attacker, depth - 1, ply + 1);
            bb_->remove(defender, w);
        } else if (!defenderWinsFirst && wp.empty() && depth >= 2) {
            // Three-threat: no four yet, but an open four is threatened next.
            // The defender has a choice, so every defence must be refuted.
            openFourPointsIn(attacker, all, gain);
            if (!gain.empty()) {
                std::vector<int> def = gain;
                fourPointsIn(defender, all, counter);
                for (const int c : counter) {
                    if (std::find(def.begin(), def.end(), c) == def.end()) {
                        def.push_back(c);
                    }
                }
                if (def.size() <= MAX_DEFENCE) {
                    win = true;
                    for (const int d : def) {
                        if (!bb_->isEmpty(d)) continue;
                        lastSq_ = d;
                        lastColor_ = defender;
                        bb_->place(defender, d);
                        const bool refuted =
                            bb_->winsAt(defender, d) ||
                            !vct(attacker, depth - 1, ply + 1);
                        bb_->remove(defender, d);
                        if (refuted) {
                            win = false;
                            break;
                        }
                    }
                }
            }
        }

        bb_->remove(attacker, m);
        lastSq_ = prevSq;
        lastColor_ = prevColor;
        if (win) {
            if (ply == 0) vctFirstMove_ = m;
            return true;
        }
    }
    return false;
}

int SearchV5::vctWin(BitBoard& bb, int attacker, int maxThreats) {
    bb_ = &bb;
    lastSq_ = -1;
    lastColor_ = attacker;
    vctFirstMove_ = -1;
    return vct(attacker, maxThreats, 0) ? vctFirstMove_ : -1;
}

// Pre-search phases (VCT probe, root safety screen) must not be able to
// consume the whole move budget. A node allowance is useless under a
// wall-clock budget -- limits_.maxNodes is 0 there -- so restrict the
// *deadline* instead: outOfTime() already measures elapsed time from t0_
// against limits_.maxTimeMs, so temporarily lowering it bounds the phase
// with the machinery that is already in place.
//
// Without this, a depth-8 vcf() in the screen runs until the real deadline,
// aborted_ is then cleared, and the main search starts with an expired clock
// -- which is why SearchV5 reached depth 4-6 where SearchV4 reached 7-9.
long long SearchV5::beginPhaseBudget(int denom) {
    const long long saved = limits_.maxTimeMs;
    if (saved > 0) {
        const long long elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0_)
                .count();
        limits_.maxTimeMs = std::min(saved, elapsed + saved / denom);
    }
    return saved;
}

void SearchV5::endPhaseBudget(long long saved) {
    limits_.maxTimeMs = saved;
    aborted_ = false;  // a phase budget stop must not abort the real search
}

// --- Root safety screening ------------------------------------------------

bool SearchV5::moveAllowsOpponentVcf(BitBoard& bb, int color, int sq,
                                     int vcfDepth) {
    bb_ = &bb;
    if (!bb.isEmpty(sq)) return false;
    bb.place(color, sq);
    bool lost = false;
    if (!bb.winsAt(color, sq)) {  // our own five ends the game first
        lastSq_ = sq;
        lastColor_ = color;
        vcfFirstMove_ = -1;
        lost = vcf(color ^ 1, vcfDepth, 0);
    }
    bb.remove(color, sq);
    lastSq_ = -1;
    lastColor_ = color;
    return lost;
}

// Play each of the top root candidates and give the *opponent* a VCF search
// from the resulting position. A candidate that hands over a forced win is
// marked unsafe. Bounded by its own node allowance so a screen that finds
// nothing cannot eat the budget the real search needs.
void SearchV5::screenRootMoves(BitBoard& bb, int color) {
    rootUnsafe_.fill(false);
    rootHasSafe_ = false;

    std::vector<int> cand;
    bb.genMovesNear(cand);
    if (cand.empty()) return;
    bb_ = &bb;
    orderMoves(cand, color, -1, 4, 1);  // ply 1: no root book bonus here

    constexpr std::size_t SCREEN_WIDTH = 10;
    const std::size_t k = std::min(SCREEN_WIDTH, cand.size());
    const std::uint64_t start = nodes_;
    const std::uint64_t allowance =
        limits_.maxNodes > 0 ? limits_.maxNodes / 4 : 30000;

    int safe = 0;
    for (std::size_t i = 0; i < k; ++i) {
        if (nodes_ - start > allowance) break;
        const int m = cand[i];
        if (moveAllowsOpponentVcf(bb, color, m, 8)) {
            rootUnsafe_[static_cast<std::size_t>(m)] = true;
        } else {
            ++safe;
        }
    }
    rootHasSafe_ = safe > 0;
}

SearchResult SearchV5::search(BitBoard& bb, int color, const SearchLimits& limits,
                              const OpeningBook* book) {
    bb_ = &bb;
    limits_ = limits;
    nodes_ = 0;
    aborted_ = false;
    rootColor_ = color;
    rootBestMove_ = -1;
    t0_ = std::chrono::steady_clock::now();
    tt_.newSearch();
    bookBonus_.fill(0);
    evalAtPly_.fill(0);
    for (auto& k : killers_) k = {-1, -1};
    nullPly_ = -1;
    extUsed_ = 0;
    rootUnsafe_.fill(false);
    rootHasSafe_ = false;

    SearchResult res;
    if (bb.hasFive(color) || bb.hasFive(color ^ 1)) {
        res.bestMove = -1;
        return res;
    }

    const int win = vcfWin(bb, color, 12);
    if (win >= 0) {
        res.bestMove = win;
        res.score = MATE;
        res.forcedWin = true;
        res.nodes = nodes_;
        return res;
    }

    // No forced win by fours. Widen the threat space to threes (VCT) and, if
    // a chain turns up, steer the search at it with a root ordering bonus.
    // The VCT defence set is a heuristic subset of all legal defences, so
    // the move is *suggested*, never returned as a mate -- the normal search
    // still has to confirm it.  Unlike Round 5's SearchVct this runs only
    // when a cheap precondition says the position is forcing, and is capped
    // at a quarter of the per-move budget.
    if (worthVct(color)) {
        // Node cap for node budgets, deadline cap for wall-clock budgets --
        // maxNodes is 0 in the latter, so the node cap alone bounds nothing.
        vctNodeCap_ = limits_.maxNodes > 0 ? nodes_ + limits_.maxNodes / 4 : 0;
        const long long saved = beginPhaseBudget(4);
        const int threatMove = vctWin(bb, color, 4);
        endPhaseBudget(saved);
        vctNodeCap_ = 0;
        if (threatMove >= 0) {
            bookBonus_[static_cast<std::size_t>(threatMove)] = 1LL << 42;
        }
    }

    if (book != nullptr) {
        std::vector<OpeningBook::Entry> entries;
        book->probeAll(bb, color, entries);
        if (!entries.empty()) {
            if (limits_.playBook) {
                res.bestMove = entries[0].move;
                res.score = 0;
                return res;
            }
            for (const auto& e : entries) {
                bookBonus_[static_cast<std::size_t>(e.move)] =
                    (1LL << 41) + e.weight;
            }
        }
    }

    int maxDepth = limits_.maxDepth;
    if (maxDepth == 0) {
        maxDepth = (limits_.maxTimeMs > 0 || limits_.maxNodes > 0) ? MAX_PLY
                                                                  : 10;
    }
    {
        // At most ~1/6 of the move budget may go to the safety screen.
        const long long saved = beginPhaseBudget(6);
        screenRootMoves(bb, color);
        endPhaseBudget(saved);
    }

    eval_.reset(bb);
    int prevScore = 0;
    for (int d = 1; d <= maxDepth; ++d) {
        lastSq_ = -1;
        lastColor_ = color;

        // Aspiration window around the previous iteration's score, widening
        // on a fail high/low. Disabled for the first iterations and inside
        // mate-scored windows, where the score can move too far to bracket.
        int alpha = -INF;
        int beta = INF;
        int delta = 0;
        if (d >= 4 && prevScore > -MATE_THRESHOLD && prevScore < MATE_THRESHOLD) {
            delta = 2 * levelScore(ThreatLevel::OpenThree);  // 10000
            alpha = prevScore - delta;
            beta = prevScore + delta;
        }

        int s = 0;
        for (;;) {
            s = negamax(color, d, alpha, beta, 0);
            if (aborted_ || delta == 0) break;
            if (s <= alpha) {
                beta = alpha / 2 + beta / 2;
                alpha = (s > -MATE_THRESHOLD) ? s - delta : -INF;
            } else if (s >= beta) {
                beta = (s < MATE_THRESHOLD) ? s + delta : INF;
            } else {
                break;
            }
            delta += delta / 2;
            if (delta > 20 * levelScore(ThreatLevel::OpenThree)) {
                alpha = -INF;
                beta = INF;
                delta = 0;
            }
        }
        if (aborted_) break;
        prevScore = s;
        res.depth = d;
        res.score = s;
        res.bestMove = rootBestMove_;
        if (s >= MATE_THRESHOLD) break;
    }
    res.nodes = nodes_;
    res.aborted = aborted_;
    return res;
}

}  // namespace oog
