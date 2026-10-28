#include "search_uct_pb.hpp"

#include <algorithm>
#include <cmath>

#include "book.hpp"

namespace oog {

namespace {

// Progressive Bias UCB constants. k1 = sqrt(2) matches the paper directly;
// k2 is described there only as "a tunable parameter" with no value given
// in the abstract, so this project picks 1.0 (comparable magnitude to the
// exploration term, since H/maxH is already normalized to [0, 1]) as an
// untuned first guess, same spirit as SearchV3's margin constants.
constexpr double K1 = 1.4142135623730951;
constexpr double K2 = 1.0;

// Squash evaluate()'s raw score (roughly S_CLOSED_TWO=100 .. S_FIVE=1e6,
// see eval.cpp) into MCTS's expected [-1, 1] value range. Scale chosen near
// the open-three threat magnitude (S_OPEN_THREE=5000); untuned.
double squash(int rawEval) {
    return std::tanh(static_cast<double>(rawEval) / 4000.0);
}

}  // namespace

SearchUctPb::SearchUctPb() = default;

bool SearchUctPb::outOfTime() {
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

void SearchUctPb::collectWinningPoints(int color, std::vector<int>& out) {
    out.clear();
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || !bb_->isEmpty(s)) continue;
        bb_->place(color, s);
        const bool w = bb_->winsAt(color, s);
        bb_->remove(color, s);
        if (w) out.push_back(s);
    }
}

bool SearchUctPb::hasImmediateWin(int color) {
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || !bb_->isEmpty(s)) continue;
        bb_->place(color, s);
        const bool w = bb_->winsAt(color, s);
        bb_->remove(color, s);
        if (w) return true;
    }
    return false;
}

bool SearchUctPb::vcf(int attacker, int depth, int ply) {
    ++nodes_;
    if ((nodes_ & 0x3FFu) == 0 && outOfTime()) return false;

    if (lastSq_ >= 0 && bb_->winsAt(lastColor_, lastSq_)) {
        return lastColor_ == attacker;
    }
    if (bb_->count() >= N * N || depth <= 0) return false;

    std::vector<int> moves;
    bb_->genMovesNear(moves);
    if (moves.empty()) return false;

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

int SearchUctPb::vcfWin(BitBoard& bb, int attacker, int maxAttacks) {
    bb_ = &bb;
    lastSq_ = -1;
    lastColor_ = attacker;
    vcfFirstMove_ = -1;
    return vcf(attacker, maxAttacks, 0) ? vcfFirstMove_ : -1;
}

void SearchUctPb::expand(Node* node, int color) {
    std::vector<int> moves;
    bb_->genMovesNear(moves);
    if (moves.empty()) {
        for (int s = 0; s < BitBoard::SIZE; ++s) {
            if (BitBoard::isPlayableSq(s) && bb_->isEmpty(s)) moves.push_back(s);
        }
    }

    // Rank by evaluateMove() and cap, same spirit as the alpha-beta
    // variants' MAX_CAND: keeps branching tractable near the root where
    // most iterations spend their budget.
    struct Scored {
        int sq;
        int h;
    };
    std::vector<Scored> scored;
    scored.reserve(moves.size());
    int maxH = 1;
    for (const int m : moves) {
        const int h = std::max(evaluateMove(*bb_, color, m), 0);
        maxH = std::max(maxH, h);
        scored.push_back({m, h});
    }
    std::sort(scored.begin(), scored.end(),
              [](const Scored& a, const Scored& b) { return a.h > b.h; });
    constexpr std::size_t MAX_CAND = 24;
    if (scored.size() > MAX_CAND) scored.resize(MAX_CAND);

    node->children.reserve(scored.size());
    for (const auto& sc : scored) {
        auto child = std::make_unique<Node>();
        child->move = sc.sq;
        child->heuristic = static_cast<double>(sc.h) / static_cast<double>(maxH);
        node->children.push_back(std::move(child));
    }
    node->expanded = true;
}

SearchUctPb::Node* SearchUctPb::select(Node* node) {
    Node* best = nullptr;
    double bestUcb = -1e300;
    const double logParent = std::log(static_cast<double>(node->visits + 1));
    for (auto& c : node->children) {
        const double q = c->visits > 0 ? c->valueSum / c->visits : 0.0;
        const double explore =
            K1 * std::sqrt(logParent / static_cast<double>(c->visits + 1));
        const double bias = K2 * c->heuristic;
        const double ucb = q + explore + bias;
        if (ucb > bestUcb) {
            bestUcb = ucb;
            best = c.get();
        }
    }
    return best;
}

double SearchUctPb::iterate(Node* node, int color, int ply) {
    ++nodes_;

    if (lastSq_ >= 0 && bb_->winsAt(lastColor_, lastSq_)) {
        // The move that led to `node` completed five for the opponent:
        // `color` (to move at `node`) has lost.
        node->visits++;
        node->valueSum += -1.0;
        return -1.0;
    }
    if (bb_->count() >= N * N) {
        node->visits++;
        return 0.0;
    }

    double v;
    if (!node->expanded) {
        expand(node, color);
        // ADP substitute: this project's own static eval, squashed to
        // [-1, 1], stands in for the paper's trained win-probability
        // network as the newly-expanded leaf's value estimate.
        v = squash(eval_.value(color));
    } else if (node->children.empty()) {
        v = 0.0;  // no legal moves left
    } else {
        Node* child = select(node);
        const int m = child->move;
        const int prevSq = lastSq_;
        const int prevColor = lastColor_;
        lastSq_ = m;
        lastColor_ = color;
        bb_->place(color, m);
        eval_.update(*bb_, m);
        const double childValue = iterate(child, color ^ 1, ply + 1);
        bb_->remove(color, m);
        eval_.update(*bb_, m);
        lastSq_ = prevSq;
        lastColor_ = prevColor;
        v = -childValue;
    }
    node->visits++;
    node->valueSum += v;
    return v;
}

SearchResult SearchUctPb::search(BitBoard& bb, int color,
                                 const SearchLimits& limits,
                                 const OpeningBook* /*book*/) {
    // Opening-book integration is intentionally out of scope for this
    // experimental variant -- it plays purely from the tree search.
    bb_ = &bb;
    limits_ = limits;
    nodes_ = 0;
    aborted_ = false;
    t0_ = std::chrono::steady_clock::now();

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

    eval_.reset(bb);
    Node root;
    lastSq_ = -1;
    lastColor_ = color;
    while (!outOfTime()) {
        lastSq_ = -1;
        lastColor_ = color;
        iterate(&root, color, 0);
    }

    // Final move: the most-visited child ("robust child"), standard MCTS
    // practice -- more robust to a single lucky/unlucky rollout than
    // picking by raw average value.
    Node* best = nullptr;
    for (auto& c : root.children) {
        if (best == nullptr || c->visits > best->visits) best = c.get();
    }
    if (best != nullptr) {
        res.bestMove = best->move;
        // Not a minimax score -- this is the squashed average value scaled
        // for rough eyeball comparison against the other engines' scores,
        // not an apples-to-apples number. depth reports the root's total
        // visit count (no single "search depth" concept applies to MCTS).
        res.score = static_cast<int>((best->valueSum / best->visits) * 100000.0);
        res.depth = static_cast<int>(std::min<std::uint64_t>(
            static_cast<std::uint64_t>(root.visits), std::uint64_t{1} << 30));
    }
    res.nodes = nodes_;
    res.aborted = aborted_;
    return res;
}

}  // namespace oog
