#pragma once

#include <memory>
#include <vector>

#include "search.hpp"  // reuses SearchLimits, SearchResult, MATE, MATE_THRESHOLD

namespace oog {

class OpeningBook;

// SearchUctPb implements the search side of Cao & Lin, "UCT-ADP Progressive
// Bias Algorithm for Solving Gomoku" (arXiv:1912.05407): genuine MCTS/UCT
// with a Progressive Bias term added to the selection formula,
//
//   UCB(child) = Q(child) + k1*sqrt(ln(N_parent)/N_child) + k2*(H_child/maxH)
//
// where H is a heuristic value (the paper's exponential threat sum; this
// project reuses its own evaluateMove() instead, since it already encodes
// the same threat-tier information via ThreatLevel/classifyFeature).
//
// One piece is NOT ported: the paper's "ADP" component trains a small
// neural network via self-play to replace random-playout simulation with a
// learned win-probability estimate. Training that network is a separate
// project (self-play data generation + a training loop), not something
// that belongs in a single search-strategy file. This class substitutes
// this codebase's existing static evaluation (EvalState, squashed into
// [-1, 1]) wherever the paper's ADP network would be queried. The tree
// search, selection formula, and expansion-without-full-rollout structure
// are otherwise a real MCTS, not a stand-in.
//
// A forced-win precheck (this project's own vcf()/vcfWin()) still runs
// first, matching the other search classes here, so the comparison isn't
// skewed by one engine lacking exact tactical detection the others have.
class SearchUctPb {
public:
    SearchUctPb();

    SearchResult search(BitBoard& bb, int color, const SearchLimits& limits,
                        const OpeningBook* book = nullptr);

    int vcfWin(BitBoard& bb, int attacker, int maxAttacks = 12);

    std::uint64_t nodes() const { return nodes_; }
    void clearTT() {}  // no transposition table in this variant

private:
    struct Node {
        int move = -1;              // move that led to this node
        int visits = 0;
        double valueSum = 0.0;      // backprop sum, from this node's own side-to-move perspective
        double heuristic = 0.0;     // normalized prior in [0, 1] (evaluateMove()/maxSiblingH)
        bool expanded = false;
        std::vector<std::unique_ptr<Node>> children;
    };

    BitBoard* bb_ = nullptr;
    EvalState eval_;
    SearchLimits limits_;
    std::uint64_t nodes_ = 0;
    int lastSq_ = -1;
    int lastColor_ = 0;
    bool aborted_ = false;
    std::chrono::steady_clock::time_point t0_;
    int vcfFirstMove_ = -1;

    // vcf()/collectWinningPoints()/hasImmediateWin() reuse the ordinary
    // move-ordering-free candidate generation (genMovesNear); no history
    // table or TT is needed for a tree-based search.
    bool outOfTime();
    bool vcf(int attacker, int depth, int ply);
    void collectWinningPoints(int color, std::vector<int>& out);
    bool hasImmediateWin(int color);

    // Builds `node`'s children (candidate moves + normalized heuristic
    // priors) without recursing into them.
    void expand(Node* node, int color);
    // Progressive-Bias UCB selection among `node`'s already-built children.
    Node* select(Node* node);
    // One MCTS iteration: select down to an unexpanded node, expand it,
    // back up. Returns the value from `color`'s (the mover-at-`node`'s)
    // perspective.
    double iterate(Node* node, int color, int ply);
};

}  // namespace oog
