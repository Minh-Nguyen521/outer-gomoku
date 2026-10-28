#include "search_minibal.hpp"

#include <algorithm>

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

}  // namespace

SearchMinibal::SearchMinibal() : history_(2 * BitBoard::SIZE, 0) {}

std::uint64_t SearchMinibal::keyWithSide(int color) const {
    return positionKey(*bb_, color);
}

bool SearchMinibal::outOfTime() {
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

void SearchMinibal::orderMoves(std::vector<int>& moves, int color, int ttMove,
                               int depth, int ply) {
    struct Scored {
        int sq;
        long long sc;
    };
    const bool cheap = depth <= 1;
    std::vector<Scored> v;
    v.reserve(moves.size());
    for (const int m : moves) {
        long long sc = 0;
        if (!cheap) {
            sc = static_cast<long long>(evaluateMove(*bb_, color, m)) * 10000;
        }
        sc += history_[static_cast<std::size_t>(color) * BitBoard::SIZE +
                       static_cast<std::size_t>(m)];
        if (ply == 0) sc += bookBonus_[static_cast<std::size_t>(m)];
        if (m == ttMove) sc += 1LL << 40;
        v.push_back({m, sc});
    }
    std::sort(v.begin(), v.end(),
              [](const Scored& a, const Scored& b) { return a.sc > b.sc; });
    for (std::size_t i = 0; i < moves.size(); ++i) moves[i] = v[i].sq;
}

void SearchMinibal::collectWinningPoints(int color, std::vector<int>& out) {
    out.clear();
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || !bb_->isEmpty(s)) continue;
        bb_->place(color, s);
        const bool w = bb_->winsAt(color, s);
        bb_->remove(color, s);
        if (w) out.push_back(s);
    }
}

bool SearchMinibal::hasImmediateWin(int color) {
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || !bb_->isEmpty(s)) continue;
        bb_->place(color, s);
        const bool w = bb_->winsAt(color, s);
        bb_->remove(color, s);
        if (w) return true;
    }
    return false;
}

// Identical to Search::negamax (baseline): the "opponent plays rationally"
// half of Minibal's design is ordinary alpha-beta, unmodified.
int SearchMinibal::negamax(int color, int depth, int alpha, int beta, int ply) {
    ++nodes_;
    if ((nodes_ & 0x3FFu) == 0 && outOfTime()) return 0;

    if (lastSq_ >= 0 && bb_->winsAt(lastColor_, lastSq_)) {
        return -(MATE - ply);
    }
    if (bb_->count() >= N * N) return 0;

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

    if (depth <= 0) return eval_.value(color);

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

    int best = -INF;
    int bestMove = -1;
    const int alpha0 = alpha;
    for (std::size_t i = 0; i < moves.size(); ++i) {
        const int m = moves[i];
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
            const bool lmr = i >= 4 && depth >= 3;
            const int rd = lmr ? depth - 1 : depth;
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
            break;
        }
    }

    if (ply == 0) rootBestMove_ = bestMove;

    const TT::Flag flag =
        best >= beta ? TT::Lower : (best > alpha0 ? TT::Exact : TT::Upper);
    tt_.store(key, depth, flag, adjustToStore(best, ply), bestMove);
    return best;
}

// The only place this class actually differs from Search: the root move
// choice. Every candidate is searched at a full window (no pruning across
// root siblings) so each move's true value is available for the Minibal_+
// comparison, not just whichever value ordinary alpha-beta happened to
// prove was >= the best found so far.
int SearchMinibal::rootMinibalPass(int color, int depth) {
    std::vector<int> moves;
    bb_->genMovesNear(moves);
    if (moves.empty()) {
        for (int s = 0; s < BitBoard::SIZE; ++s) {
            if (BitBoard::isPlayableSq(s) && bb_->isEmpty(s)) moves.push_back(s);
        }
    }
    if (moves.empty()) {
        rootBestMove_ = -1;
        return 0;
    }
    orderMoves(moves, color, -1, depth, 0);

    int bestPositiveMove = -1;
    int bestPositiveScore = INF;   // smallest positive score wins
    int bestOtherMove = -1;
    int bestOtherScore = -INF;     // largest (least negative) score wins

    for (const int m : moves) {
        const int prevSq = lastSq_;
        const int prevColor = lastColor_;
        lastSq_ = m;
        lastColor_ = color;
        bb_->place(color, m);
        eval_.update(*bb_, m);
        const int s = -negamax(color ^ 1, depth - 1, -INF, INF, 1);
        bb_->remove(color, m);
        eval_.update(*bb_, m);
        lastSq_ = prevSq;
        lastColor_ = prevColor;
        if (aborted_) return 0;

        if (s > 0) {
            if (s < bestPositiveScore) {
                bestPositiveScore = s;
                bestPositiveMove = m;
            }
        } else if (s > bestOtherScore) {
            bestOtherScore = s;
            bestOtherMove = m;
        }
    }

    if (bestPositiveMove >= 0) {
        rootBestMove_ = bestPositiveMove;
        return bestPositiveScore;
    }
    rootBestMove_ = bestOtherMove;
    return bestOtherScore;
}

bool SearchMinibal::vcf(int attacker, int depth, int ply) {
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

int SearchMinibal::vcfWin(BitBoard& bb, int attacker, int maxAttacks) {
    bb_ = &bb;
    lastSq_ = -1;
    lastColor_ = attacker;
    vcfFirstMove_ = -1;
    return vcf(attacker, maxAttacks, 0) ? vcfFirstMove_ : -1;
}

SearchResult SearchMinibal::search(BitBoard& bb, int color,
                                   const SearchLimits& limits,
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

    SearchResult res;
    if (bb.hasFive(color) || bb.hasFive(color ^ 1)) {
        res.bestMove = -1;
        return res;
    }

    // Forced-win precheck: if Minibal_+ can win outright, it should -- the
    // paper's decision rule only governs the choice among the moves that
    // remain once "can I win" and "must I not lose" are already settled by
    // the search values below; a proven mate is as decisive as it gets.
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
        const int s = rootMinibalPass(color, d);
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
