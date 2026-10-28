#include "search_nullkiller.hpp"

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
int razorMargin(int depth) { return 800 + 300 * depth; }
int futilityMargin(int depth, bool improving) {
    return std::max(700 * depth - (improving ? 300 : 0), 0);
}

}  // namespace

SearchNullKiller::SearchNullKiller()
    : history_(2 * BitBoard::SIZE, 0), counterMove_(2 * BitBoard::SIZE, -1) {
    for (auto& k : killers_) k = {-1, -1};
}

std::uint64_t SearchNullKiller::keyWithSide(int color) const {
    return positionKey(*bb_, color);
}

bool SearchNullKiller::outOfTime() {
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

void SearchNullKiller::orderMoves(std::vector<int>& moves, int color, int ttMove,
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
        // Killers rank below the TT move but above every history-ordered
        // quiet move -- the point is to try a sibling-local refutation
        // before falling back on global statistics.
        if (ply < MAX_PLY) {
            const auto& k = killers_[static_cast<std::size_t>(ply)];
            if (m == k[0]) sc += 1LL << 38;
            else if (m == k[1]) sc += 1LL << 37;
        }
        // Counter-move: the reply that refuted the opponent's last stone.
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

void SearchNullKiller::collectWinningPoints(int color, std::vector<int>& out) {
    out.clear();
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || !bb_->isEmpty(s)) continue;
        bb_->place(color, s);
        const bool w = bb_->winsAt(color, s);
        bb_->remove(color, s);
        if (w) out.push_back(s);
    }
}

bool SearchNullKiller::hasImmediateWin(int color) {
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || !bb_->isEmpty(s)) continue;
        bb_->place(color, s);
        const bool w = bb_->winsAt(color, s);
        bb_->remove(color, s);
        if (w) return true;
    }
    return false;
}

int SearchNullKiller::computeLmrReduction(int color, int m, int depth, int ply,
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

int SearchNullKiller::negamax(int color, int depth, int alpha, int beta, int ply) {
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
            eval + razorMargin(depth) < alpha && !hasImmediateWin(color ^ 1)) {
            if (!vcf(color, 6, ply)) {
                return eval;
            }
            // A forced win exists here; fall through to the normal search.
        }

        // Futility pruning: eval already clears beta by a comfortable
        // margin -- skip the node entirely, no moves generated.
        if (eval < MATE_THRESHOLD && beta < MATE_THRESHOLD &&
            eval - futilityMargin(depth, improving) >= beta &&
            !hasImmediateWin(color ^ 1)) {
            return eval;
        }
    }

    // Null-move forward pruning.  Pass the turn and search shallowly: if the
    // position still beats beta after conceding a free move, it beats beta
    // when we actually move.  In Go-Moku that implication is exact -- there
    // is no zugzwang, so playing a stone is never worse than passing (see
    // the header).  The remaining guards are about the *eval* being
    // trustworthy here, not about the pass itself: never on the PV or at the
    // root, never inside a mate-scored window, never twice in a row, and
    // never while either side has a five on the board waiting to be played.
    if (!nodeIsPv && !rootNode && depth >= 3 && nullPly_ < 0 &&
        eval >= beta && beta > -MATE_THRESHOLD && beta < MATE_THRESHOLD &&
        !hasImmediateWin(color) && !hasImmediateWin(color ^ 1)) {
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

    std::vector<int> moves;
    bb_->genMovesNear(moves);
    if (moves.empty()) {
        for (int s = 0; s < BitBoard::SIZE; ++s) {
            if (BitBoard::isPlayableSq(s) && bb_->isEmpty(s)) moves.push_back(s);
        }
        if (moves.empty()) return 0;
    }
    orderMoves(moves, color, ttMove, depth, ply);
    constexpr std::size_t MAX_CAND = 24;
    if (ply > 0 && moves.size() > MAX_CAND) moves.resize(MAX_CAND);

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
        if (!rootNode && depth <= 4 && best > -MATE_THRESHOLD && !isThreat &&
            static_cast<int>(i) >= moveCountLimit) {
            continue;
        }

        const int prevSq = lastSq_;
        const int prevColor = lastColor_;
        lastSq_ = m;
        lastColor_ = color;
        bb_->place(color, m);
        eval_.update(*bb_, m);
        int s;
        if (i == 0) {
            s = -negamax(color ^ 1, depth - 1, -beta, -alpha, ply + 1);
        } else {
            const bool nodePv = nodeIsPv;
            int red = 0;
            if (!isThreat && depth >= 3) {
                red = computeLmrReduction(color, m, depth, ply,
                                          static_cast<int>(i), nodePv);
            }
            const int rd = depth - red;
            s = -negamax(color ^ 1, rd - 1, -alpha - 1, -alpha, ply + 1);
            if (s > alpha && s < beta) {
                s = -negamax(color ^ 1, depth - 1, -beta, -alpha, ply + 1);
            }
        }
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
            // Killer + counter-move bookkeeping.  lastSq_/lastColor_ have
            // already been restored above, so they name the opponent's move
            // that this cutoff refutes.
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

bool SearchNullKiller::vcf(int attacker, int depth, int ply) {
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

int SearchNullKiller::vcfWin(BitBoard& bb, int attacker, int maxAttacks) {
    bb_ = &bb;
    lastSq_ = -1;
    lastColor_ = attacker;
    vcfFirstMove_ = -1;
    return vcf(attacker, maxAttacks, 0) ? vcfFirstMove_ : -1;
}

SearchResult SearchNullKiller::search(BitBoard& bb, int color, const SearchLimits& limits,
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
    eval_.reset(bb);
    for (int d = 1; d <= maxDepth; ++d) {
        lastSq_ = -1;
        lastColor_ = color;
        const int s = negamax(color, d, -INF, INF, 0);
        if (aborted_) break;
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
