#include "search.hpp"

#include <algorithm>

#include "book.hpp"

namespace oog {

namespace {

// Mate scores are relative to the distance from the root.  When storing in
// the TT, shift them so they stay valid when the position is hit at a
// different ply; undo the shift on retrieval.
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

Search::Search() : history_(2 * BitBoard::SIZE, 0) {}

std::uint64_t Search::keyWithSide(int color) const {
    return positionKey(*bb_, color);
}

bool Search::outOfTime() {
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

void Search::orderMoves(std::vector<int>& moves, int color, int ttMove,
                        int depth, int ply) {
    struct Scored {
        int sq;
        long long sc;
    };
    // Threat scoring is expensive (~180 ns/move); near the leaves ordering has
    // no cut-off value, so order by history only there.
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

void Search::collectWinningPoints(int color, std::vector<int>& out) {
    out.clear();
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || !bb_->isEmpty(s)) continue;
        bb_->place(color, s);
        const bool w = bb_->winsAt(color, s);
        bb_->remove(color, s);
        if (w) out.push_back(s);
    }
}

bool Search::hasImmediateWin(int color) {
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || !bb_->isEmpty(s)) continue;
        bb_->place(color, s);
        const bool w = bb_->winsAt(color, s);
        bb_->remove(color, s);
        if (w) return true;
    }
    return false;
}

int Search::negamax(int color, int depth, int alpha, int beta, int ply) {
    ++nodes_;
    if ((nodes_ & 0x3FFu) == 0 && outOfTime()) return 0;

    // The player who just moved completed five -> side to move loses.
    if (lastSq_ >= 0 && bb_->winsAt(lastColor_, lastSq_)) {
        return -(MATE - ply);
    }
    if (bb_->count() >= N * N) return 0;  // draw

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
    if (moves.empty()) {  // degenerate: not full but nothing near stones
        for (int s = 0; s < BitBoard::SIZE; ++s) {
            if (BitBoard::isPlayableSq(s) && bb_->isEmpty(s)) moves.push_back(s);
        }
        if (moves.empty()) return 0;
    }
    orderMoves(moves, color, ttMove, depth, ply);
    // Bound branching: beyond the top candidates a quiet move almost never
    // matters, and the search stays correct because every forcing threat is
    // scored at the front.  The root is never capped.
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
            // Late moves are tried at reduced depth (LMR), then re-searched
            // at full depth only if they beat alpha.
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

bool Search::vcf(int attacker, int depth, int ply) {
    ++nodes_;
    if ((nodes_ & 0x3FFu) == 0 && outOfTime()) return false;

    // The player who just moved won: the attacker wins iff that was them.
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

        if (bb_->winsAt(attacker, m)) {  // immediate five
            bb_->remove(attacker, m);
            lastSq_ = prevSq;
            lastColor_ = prevColor;
            if (ply == 0) vcfFirstMove_ = m;
            return true;
        }

        // Where could the attacker complete five on the next move?
        collectWinningPoints(attacker, wp);
        // If the defender can win right now, this forcing line is unsound.
        const bool defenderWinsFirst = hasImmediateWin(attacker ^ 1);
        const bool forcing = !wp.empty();
        if (forcing && !defenderWinsFirst && wp.size() >= 2) {
            // Open four: two winning points, the defender can block only one.
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

        // Exactly one winning point: the defender's reply is forced.
        const int w = wp[0];
        bool win = true;
        lastSq_ = w;
        lastColor_ = attacker ^ 1;
        bb_->place(attacker ^ 1, w);
        if (bb_->winsAt(attacker ^ 1, w)) {
            win = false;  // the forced block also makes five for the defender
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

int Search::vcfWin(BitBoard& bb, int attacker, int maxAttacks) {
    bb_ = &bb;
    lastSq_ = -1;
    lastColor_ = attacker;
    vcfFirstMove_ = -1;
    return vcf(attacker, maxAttacks, 0) ? vcfFirstMove_ : -1;
}

SearchResult Search::search(BitBoard& bb, int color, const SearchLimits& limits,
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
        res.bestMove = -1;  // game already decided
        return res;
    }

    // Threat-space precheck: a forcing win is definitive and fast to find.
    const int win = vcfWin(bb, color, 12);
    if (win >= 0) {
        res.bestMove = win;
        res.score = MATE;
        res.forcedWin = true;
        res.nodes = nodes_;
        return res;
    }

    // Opening book: play the best line directly, or order book moves first.
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
        if (s >= MATE_THRESHOLD) break;  // forced mate confirmed by search
    }
    res.nodes = nodes_;
    res.aborted = aborted_;
    return res;
}

}  // namespace oog
