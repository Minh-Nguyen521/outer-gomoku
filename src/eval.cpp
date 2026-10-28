#include "eval.hpp"

#include "eval_params.hpp"

#include <array>
#include <vector>

namespace oog {

namespace {

// --- scores ----------------------------------------------------------------
// Previously `constexpr`. Now aliases onto the runtime-tunable parameter
// block (src/eval_params.hpp) so a tuner can try values without a recompile.
// Defaults are exactly the old constants, so with no environment overrides
// set the behaviour -- and every measurement already recorded -- is
// unchanged. See eval_params.hpp for why this matters.
#define S_FIVE (evalParams().five)
#define S_OPEN_FOUR (evalParams().openFour)
#define S_CLOSED_FOUR (evalParams().closedFour)
#define S_OPEN_THREE (evalParams().openThree)
#define S_CLOSED_THREE (evalParams().closedThree)
#define S_OPEN_TWO (evalParams().openTwo)
#define S_CLOSED_TWO (evalParams().closedTwo)

// Opponent patterns are weighted higher than own (defense is urgent).
#define DEF_NUM (evalParams().defNum)
#define DEF_DEN (evalParams().defDen)

int runScore(int len, bool lopen, bool ropen) {
    if (len >= 5) return S_FIVE;
    if (len == 4) {
        if (lopen && ropen) return S_OPEN_FOUR;
        if (lopen || ropen) return S_CLOSED_FOUR;
        return S_CLOSED_FOUR;
    }
    if (len == 3) {
        if (lopen && ropen) return S_OPEN_THREE;
        if (lopen || ropen) return S_CLOSED_THREE;
        return S_CLOSED_THREE;
    }
    if (len == 2) {
        if (lopen && ropen) return S_OPEN_TWO;
        return S_CLOSED_TWO;
    }
    return 0;
}

// Value of one 5-cell window of a single color (mask: 5 bits) given whether
// the cells just beyond the window on each side are empty.
int windowValue(std::uint8_t mask, bool lEmpty, bool rEmpty) {
    int best = 0;
    int i = 0;
    while (i < 5) {
        if ((mask & (1u << i)) == 0) {
            ++i;
            continue;
        }
        int j = i;
        while (j < 5 && (mask & (1u << j))) ++j;
        const int len = j - i;
        const bool lopen = (i == 0) ? lEmpty : true;
        const bool ropen = (j == 5) ? rEmpty : true;
        const int v = runScore(len, lopen, ropen);
        if (v > best) best = v;
        i = j;
    }
    return best;
}

int cellState(std::uint8_t own8, std::uint8_t opp8, int off) {
    // off in [-4, 4]; center (0) is own after the move.
    if (off == 0) return 1;
    const int bit = off < 0 ? off + 4 : off + 3;
    // A cell set in both masks is out-of-board (blocked for both colors).
    if (((own8 >> bit) & 1) && ((opp8 >> bit) & 1)) return 2;
    if ((own8 >> bit) & 1) return 1;
    if ((opp8 >> bit) & 1) return 2;
    return 0;
}

ThreatLevel runLevel(int len, bool lopen, bool ropen) {
    if (len == 4) return (lopen && ropen) ? ThreatLevel::OpenFour
                                          : ThreatLevel::ClosedFour;
    if (len == 3) return (lopen && ropen) ? ThreatLevel::OpenThree
                                          : ThreatLevel::ClosedThree;
    if (len == 2) return (lopen && ropen) ? ThreatLevel::OpenTwo
                                          : ThreatLevel::ClosedTwo;
    return ThreatLevel::None;
}

// A "four with a gap": some 5-window (within the +/-4 line, including the
// center move) has 4 own stones + 1 empty.  Filling the empty completes five,
// so this is a closed-four threat for the mover.
bool isJumpFour(std::uint8_t own8, std::uint8_t opp8) {
    for (int st = -4; st <= 0; ++st) {
        int own = 0;
        int opp = 0;
        for (int k = 0; k < 5; ++k) {
            const int s = cellState(own8, opp8, st + k);
            own += s == 1;
            opp += s == 2;
        }
        if (opp == 0 && own >= 4) return true;
    }
    return false;
}

// Precomputed 16-bit line table (OOGiveMeFive M-mask style).
const std::array<std::uint8_t, 65536>& lineTable() {
    static const auto T = [] {
        std::array<std::uint8_t, 65536> t{};
        for (std::uint32_t f = 0; f < 65536; ++f) {
            const std::uint8_t own8 = static_cast<std::uint8_t>(f >> 8);
            const std::uint8_t opp8 = static_cast<std::uint8_t>(f & 0xFF);
            t[static_cast<std::size_t>(f)] =
                static_cast<std::uint8_t>(classifyLine(own8, opp8));
        }
        return t;
    }();
    return T;
}

// All straight lines (rows, columns, both diagonals) as sq sequences.
const std::vector<std::vector<int>>& lines() {
    static const auto L = [] {
        std::vector<std::vector<int>> res;
        for (int r = 0; r < N; ++r) {
            std::vector<int> l;
            for (int c = 0; c < N; ++c) l.push_back(sqOf(r, c));
            res.push_back(std::move(l));
        }
        for (int c = 0; c < N; ++c) {
            std::vector<int> l;
            for (int r = 0; r < N; ++r) l.push_back(sqOf(r, c));
            res.push_back(std::move(l));
        }
        // Diagonals down-right (r - c = const).
        for (int k = 0; k < N; ++k) {
            std::vector<int> l;
            for (int r = 0; r + k < N; ++r) l.push_back(sqOf(r, r + k));
            res.push_back(std::move(l));
        }
        for (int k = 1; k < N; ++k) {
            std::vector<int> l;
            for (int c = 0; c + k < N; ++c) l.push_back(sqOf(c + k, c));
            res.push_back(std::move(l));
        }
        // Diagonals down-left (r + c = const).
        for (int k = 0; k < N; ++k) {
            std::vector<int> l;
            for (int r = 0; r <= k; ++r) l.push_back(sqOf(r, k - r));
            res.push_back(std::move(l));
        }
        for (int k = N; k < 2 * N - 1; ++k) {
            std::vector<int> l;
            for (int r = k - N + 1; r < N; ++r)
                l.push_back(sqOf(r, k - r));
            res.push_back(std::move(l));
        }
        return res;
    }();
    return L;
}

// Extract the 8-cell line masks for a candidate move in one direction.
// Out-of-board cells are encoded as opponent stones so they act as blockers.
void extractLine(const BitBoard& bb, int color, int sq, int step,
                 std::uint8_t& own8, std::uint8_t& opp8) {
    own8 = 0;
    opp8 = 0;
    for (int k = 0; k < 8; ++k) {
        const int off = k < 4 ? k - 4 : k - 3;  // -4..-1, +1..+4
        const int s = sq + off * step;
        const std::uint8_t bit = static_cast<std::uint8_t>(1u << k);
        if (!BitBoard::isPlayableSq(s)) {
            // Out of board: blocked for both colors (encoded as both-set).
            own8 |= bit;
            opp8 |= bit;
            continue;
        }
        const Cell c = bb.at(s);
        if (c == Cell::Empty) continue;
        const bool own =
            color == 0 ? c == Cell::Black : c == Cell::White;
        if (own) {
            own8 |= bit;
        } else {
            opp8 |= bit;
        }
    }
}

}  // namespace

int levelScore(ThreatLevel t) {
    switch (t) {
        case ThreatLevel::Five: return S_FIVE;
        case ThreatLevel::OpenFour: return S_OPEN_FOUR;
        case ThreatLevel::ClosedFour: return S_CLOSED_FOUR;
        case ThreatLevel::OpenThree: return S_OPEN_THREE;
        case ThreatLevel::ClosedThree: return S_CLOSED_THREE;
        case ThreatLevel::OpenTwo: return S_OPEN_TWO;
        case ThreatLevel::ClosedTwo: return S_CLOSED_TWO;
        case ThreatLevel::None: return 0;
    }
    return 0;
}

ThreatLevel classifyLine(std::uint8_t own8, std::uint8_t opp8) {
    int left = 0;
    for (int o = -1; o >= -4; --o) {
        if (cellState(own8, opp8, o) == 1) {
            ++left;
        } else {
            break;
        }
    }
    int right = 0;
    for (int o = 1; o <= 4; ++o) {
        if (cellState(own8, opp8, o) == 1) {
            ++right;
        } else {
            break;
        }
    }
    const int run = 1 + left + right;
    if (run >= 5) return ThreatLevel::Five;

    const bool lopen = cellState(own8, opp8, -left - 1) == 0;
    const bool ropen = cellState(own8, opp8, right + 1) == 0;
    ThreatLevel t = runLevel(run, lopen, ropen);
    if (t < ThreatLevel::ClosedFour && isJumpFour(own8, opp8)) {
        t = ThreatLevel::ClosedFour;
    }
    return t;
}

ThreatLevel classifyFeature(std::uint16_t feature) {
    return static_cast<ThreatLevel>(
        lineTable()[static_cast<std::size_t>(feature)]);
}

int evaluate(const BitBoard& bb, int mover) {
    int sum[2] = {0, 0};
    for (const auto& line : lines()) {
        const int len = static_cast<int>(line.size());
        for (int start = 0; start + 5 <= len; ++start) {
            int nb = 0;
            int nw = 0;
            std::uint8_t mb = 0;
            std::uint8_t mw = 0;
            for (int k = 0; k < 5; ++k) {
                const Cell c = bb.at(line[static_cast<std::size_t>(start + k)]);
                if (c == Cell::Black) {
                    ++nb;
                    mb |= static_cast<std::uint8_t>(1u << k);
                } else if (c == Cell::White) {
                    ++nw;
                    mw |= static_cast<std::uint8_t>(1u << k);
                }
            }
            if (nb == 0 && nw == 0) continue;  // empty window
            if (nb > 0 && nw > 0) continue;    // blocked window
            const std::uint8_t mask = nb > 0 ? mb : mw;
            const int color = nb > 0 ? 0 : 1;
            const bool lEmpty =
                start - 1 >= 0 && bb.isEmpty(line[static_cast<std::size_t>(start - 1)]);
            const bool rEmpty =
                start + 5 < len &&
                bb.isEmpty(line[static_cast<std::size_t>(start + 5)]);
            sum[color] += windowValue(mask, lEmpty, rEmpty);
        }
    }
    const int own = sum[mover];
    const int opp = sum[mover ^ 1];
    return own - opp * DEF_NUM / DEF_DEN;
}

int evaluateMove(const BitBoard& bb, int color, int sq) {
    constexpr int DIRS[4] = {1, BitBoard::ROW_W, BitBoard::ROW_W + 1,
                             BitBoard::ROW_W - 1};
    int ownS = 0;
    int oppS = 0;
    for (const int d : DIRS) {
        std::uint8_t own8 = 0;
        std::uint8_t opp8 = 0;
        extractLine(bb, color, sq, d, own8, opp8);
        ownS += levelScore(classifyFeature(
            static_cast<std::uint16_t>((own8 << 8) | opp8)));
        oppS += levelScore(classifyFeature(
            static_cast<std::uint16_t>((opp8 << 8) | own8)));
    }
    // ownS = threats this move creates; oppS = threats it pre-empts by
    // occupying a square the opponent would otherwise use.
    return ownS + oppS * DEF_NUM / DEF_DEN;
}

// --- incremental evaluation ------------------------------------------------

namespace {

// Precomputed 5-cell windows over all lines, plus, for every square, the list
// of window ids whose value can change when that square changes (the windows
// containing it and the two windows for which it is the flanking cell that
// decides open/closed-ness).
struct WinTable {
    std::vector<std::array<int, 5>> cells;
    std::vector<int> left;   // flank sq to the left of a window, or -1
    std::vector<int> right;  // flank sq to the right of a window, or -1
    std::vector<std::vector<int>> affected;  // [sq] -> window ids
};

const WinTable& winTable() {
    static const WinTable T = [] {
        WinTable t;
        t.affected.resize(BitBoard::SIZE);
        const auto& L = lines();
        for (std::size_t li = 0; li < L.size(); ++li) {
            const auto& line = L[li];
            const int len = static_cast<int>(line.size());
            for (int start = 0; start + 5 <= len; ++start) {
                std::array<int, 5> c;
                for (int k = 0; k < 5; ++k) {
                    c[static_cast<std::size_t>(k)] =
                        line[static_cast<std::size_t>(start + k)];
                }
                const int wid = static_cast<int>(t.cells.size());
                t.cells.push_back(c);
                t.left.push_back(start - 1 >= 0
                                     ? line[static_cast<std::size_t>(start - 1)]
                                     : -1);
                t.right.push_back(start + 5 < len
                                      ? line[static_cast<std::size_t>(start + 5)]
                                      : -1);
                for (int k = 0; k < 5; ++k) {
                    t.affected[static_cast<std::size_t>(
                                   c[static_cast<std::size_t>(k)])]
                        .push_back(wid);
                }
                if (t.left.back() >= 0) {
                    t.affected[static_cast<std::size_t>(t.left.back())]
                        .push_back(wid);
                }
                if (t.right.back() >= 0) {
                    t.affected[static_cast<std::size_t>(t.right.back())]
                        .push_back(wid);
                }
            }
        }
        return t;
    }();
    return T;
}

// Score of one window for each color (0 if mixed/empty, per the evaluate()
// semantics).
void windowScores(const BitBoard& bb, const std::array<int, 5>& c, int left,
                  int right, int& v0, int& v1) {
    int nb = 0;
    int nw = 0;
    std::uint8_t mb = 0;
    std::uint8_t mw = 0;
    for (int k = 0; k < 5; ++k) {
        const Cell cc = bb.at(c[static_cast<std::size_t>(k)]);
        if (cc == Cell::Black) {
            ++nb;
            mb |= static_cast<std::uint8_t>(1u << k);
        } else if (cc == Cell::White) {
            ++nw;
            mw |= static_cast<std::uint8_t>(1u << k);
        }
    }
    v0 = 0;
    v1 = 0;
    if (nb > 0 && nw == 0) {
        const bool lEmpty = left >= 0 && bb.isEmpty(left);
        const bool rEmpty = right >= 0 && bb.isEmpty(right);
        v0 = windowValue(mb, lEmpty, rEmpty);
    } else if (nw > 0 && nb == 0) {
        const bool lEmpty = left >= 0 && bb.isEmpty(left);
        const bool rEmpty = right >= 0 && bb.isEmpty(right);
        v1 = windowValue(mw, lEmpty, rEmpty);
    }
}

}  // namespace

void EvalState::reset(const BitBoard& bb) {
    const auto& T = winTable();
    sum_[0] = 0;
    sum_[1] = 0;
    win_[0].assign(T.cells.size(), 0);
    win_[1].assign(T.cells.size(), 0);
    for (std::size_t w = 0; w < T.cells.size(); ++w) {
        int v0 = 0;
        int v1 = 0;
        windowScores(bb, T.cells[w], T.left[static_cast<std::size_t>(w)],
                     T.right[static_cast<std::size_t>(w)], v0, v1);
        win_[0][w] = v0;
        win_[1][w] = v1;
        sum_[0] += v0;
        sum_[1] += v1;
    }
}

void EvalState::update(const BitBoard& bb, int sq) {
    const auto& T = winTable();
    for (const int w : T.affected[static_cast<std::size_t>(sq)]) {
        const auto i = static_cast<std::size_t>(w);
        int v0 = 0;
        int v1 = 0;
        windowScores(bb, T.cells[i], T.left[i], T.right[i], v0, v1);
        sum_[0] += v0 - win_[0][i];
        win_[0][i] = v0;
        sum_[1] += v1 - win_[1][i];
        win_[1][i] = v1;
    }
}

int EvalState::value(int mover) const {
    return sum_[mover] - sum_[mover ^ 1] * DEF_NUM / DEF_DEN;
}

}  // namespace oog
