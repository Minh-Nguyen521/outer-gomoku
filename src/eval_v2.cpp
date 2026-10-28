#include "eval_v2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace oog {

namespace {

int envInt(const char* name, int dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return dflt;
    try { return std::stoi(v); } catch (...) { return dflt; }
}

constexpr int DIRS[4] = {1, BitBoard::ROW_W, BitBoard::ROW_W + 1,
                         BitBoard::ROW_W - 1};

// Fork bonuses, in the same score units as eval.cpp's constants
// (S_OPEN_THREE=5,000 .. S_OPEN_FOUR=100,000). A fork is as decisive as an
// open four, so the double-four bonus is sized near S_OPEN_FOUR; the weaker
// combinations sit below it. These are first guesses in the same sense the
// existing constants are -- fitting them is a separate job (PLAN.md Phase 2).
// Tunable alongside the base weights (OOG_FORK_44 / _43 / _33); defaults
// unchanged so existing measurements stay valid.
int forkDoubleFour() { static const int v = envInt("OOG_FORK_44", 90'000); return v; }
int forkFourThree()  { static const int v = envInt("OOG_FORK_43", 60'000); return v; }
int forkDoubleThree(){ static const int v = envInt("OOG_FORK_33", 30'000); return v; }

// Defence weighting, matching eval.cpp.
constexpr int DEF_NUM = 3;
constexpr int DEF_DEN = 2;

// All lines in each direction, and the line each square belongs to.
struct LineTables {
    std::vector<std::vector<int>> lines[4];
    // lineOf[dir][sq] = index into lines[dir], or -1 if sq is not playable.
    std::array<std::array<int, BitBoard::SIZE>, 4> lineOf{};
};

const LineTables& tables() {
    static const LineTables T = [] {
        LineTables t;
        for (auto& a : t.lineOf) a.fill(-1);
        for (int d = 0; d < 4; ++d) {
            const int step = DIRS[d];
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < N; ++c) {
                    const int sq = BitBoard::sqOf(r, c);
                    // Start a line only where the previous cell in this
                    // direction is off-board, so each line is walked once.
                    const int prev = sq - step;
                    if (BitBoard::isPlayableSq(prev)) {
                        // Guard against wrap: a horizontal step must stay in
                        // the same row, a diagonal must change row by one.
                        const int pr = BitBoard::boardRow(prev);
                        const int pc = BitBoard::boardCol(prev);
                        const int dr = r - pr;
                        const int dc = c - pc;
                        const bool contiguous =
                            (step == 1 && dr == 0 && dc == 1) ||
                            (step == BitBoard::ROW_W && dr == 1 && dc == 0) ||
                            (step == BitBoard::ROW_W + 1 && dr == 1 && dc == 1) ||
                            (step == BitBoard::ROW_W - 1 && dr == 1 && dc == -1);
                        if (contiguous) continue;
                    }
                    std::vector<int> line;
                    int cur = sq;
                    int cr = r;
                    int cc = c;
                    while (cr >= 0 && cr < N && cc >= 0 && cc < N) {
                        line.push_back(cur);
                        cur += step;
                        if (step == 1) { cc += 1; }
                        else if (step == BitBoard::ROW_W) { cr += 1; }
                        else if (step == BitBoard::ROW_W + 1) { cr += 1; cc += 1; }
                        else { cr += 1; cc -= 1; }
                    }
                    if (line.size() < 5) continue;  // cannot hold five
                    const int idx = static_cast<int>(t.lines[d].size());
                    for (const int s : line) {
                        t.lineOf[static_cast<std::size_t>(d)]
                                [static_cast<std::size_t>(s)] = idx;
                    }
                    t.lines[d].push_back(std::move(line));
                }
            }
        }
        return t;
    }();
    return T;
}

int tempoBonus()  { static const int v = envInt("OOG_TEMPO", 250); return v; }
int posWeight()   { static const int v = envInt("OOG_POS_W", 6); return v; }

// Centre proximity, 0 at the board edge up to 7 at the centre. Cheap table.
const std::array<int, BitBoard::SIZE>& posTable() {
    static const auto T = [] {
        std::array<int, BitBoard::SIZE> t{};
        t.fill(0);
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                const int d = std::max(std::abs(r - N / 2), std::abs(c - N / 2));
                t[static_cast<std::size_t>(BitBoard::sqOf(r, c))] = (N / 2) - d;
            }
        }
        return t;
    }();
    return T;
}

int cellOf(const BitBoard& bb, int sq, int color) {
    if (bb.isEmpty(sq)) return 0;
    return (color == 0 ? bb.isBlack(sq) : bb.isWhite(sq)) ? 1 : 2;
}

// Count `color`'s distinct four-gain squares and open threes on one line.
void countLine(const BitBoard& bb, const std::vector<int>& line, int color,
               int& fourThreats, int& openThrees) {
    fourThreats = 0;
    openThrees = 0;
    const int len = static_cast<int>(line.size());
    if (len < 5) return;

    // --- four THREATS: a 5-window with 4 own + 1 empty. Deduped by the span
    // of the four own stones, NOT by the gain square.
    //
    // This distinction is the whole point. An open four `_XXXX_` has *two*
    // gain squares but is *one* threat, and the linear window term already
    // scores it S_OPEN_FOUR. Counting gain squares made every open four look
    // like a double four and stacked a fork bonus on top of a threat that
    // was already fully paid for -- which measured 14 points *worse* than
    // the unmodified baseline against rapfi. A fork is two threats on
    // different spans, not one threat with two completions.
    int spanFourLo[32];
    int spanFourHi[32];
    int nFour = 0;
    for (int p = 0; p + 5 <= len; ++p) {
        int own = 0;
        int opp = 0;
        int lo = 99;
        int hi = -1;
        for (int k = 0; k < 5; ++k) {
            const int st = cellOf(bb, line[static_cast<std::size_t>(p + k)], color);
            if (st == 1) { ++own; lo = std::min(lo, p + k); hi = std::max(hi, p + k); }
            else if (st == 2) ++opp;
        }
        if (opp != 0 || own != 4) continue;
        bool seen = false;
        for (int i = 0; i < nFour; ++i) {
            if (spanFourLo[i] == lo && spanFourHi[i] == hi) { seen = true; break; }
        }
        if (!seen && nFour < 32) {
            spanFourLo[nFour] = lo;
            spanFourHi[nFour] = hi;
            ++nFour;
        }
    }
    fourThreats = nFour;

    // --- open threes: a 5-window with 3 own + 2 empty and no opponent, where
    // filling one empty yields a four that is open at both ends (i.e. an open
    // four). Deduped by the span of the own stones, since one open three
    // appears in several overlapping windows.
    int spanLo[32];
    int spanHi[32];
    int nSpans = 0;
    for (int p = 0; p + 5 <= len; ++p) {
        int own = 0;
        int opp = 0;
        int emptyIdx[5];
        int nEmpty = 0;
        int lo = 99;
        int hi = -1;
        for (int k = 0; k < 5; ++k) {
            const int st = cellOf(bb, line[static_cast<std::size_t>(p + k)], color);
            if (st == 1) { ++own; lo = std::min(lo, p + k); hi = std::max(hi, p + k); }
            else if (st == 2) ++opp;
            else emptyIdx[nEmpty++] = p + k;
        }
        if (opp != 0 || own != 3) continue;

        bool makesOpenFour = false;
        for (int e = 0; e < nEmpty && !makesOpenFour; ++e) {
            // Hypothetically fill emptyIdx[e]; the four must be contiguous
            // and flanked by empties on both sides to be an *open* four.
            const int fill = emptyIdx[e];
            const int flo = std::min(lo, fill);
            const int fhi = std::max(hi, fill);
            if (fhi - flo != 3) continue;  // not four in a row
            bool contiguous = true;
            for (int k = flo; k <= fhi; ++k) {
                if (k == fill) continue;
                if (cellOf(bb, line[static_cast<std::size_t>(k)], color) != 1) {
                    contiguous = false;
                    break;
                }
            }
            if (!contiguous) continue;
            const bool lEmpty =
                flo - 1 >= 0 &&
                cellOf(bb, line[static_cast<std::size_t>(flo - 1)], color) == 0;
            const bool rEmpty =
                fhi + 1 < len &&
                cellOf(bb, line[static_cast<std::size_t>(fhi + 1)], color) == 0;
            if (lEmpty && rEmpty) makesOpenFour = true;
        }
        if (!makesOpenFour) continue;

        bool seen = false;
        for (int i = 0; i < nSpans; ++i) {
            if (spanLo[i] == lo && spanHi[i] == hi) { seen = true; break; }
        }
        if (!seen && nSpans < 32) {
            spanLo[nSpans] = lo;
            spanHi[nSpans] = hi;
            ++nSpans;
        }
    }
    openThrees = nSpans;
}

}  // namespace

void EvalStateV2::recomputeLine(const BitBoard& bb, int dir, int lineIdx) {
    const auto& line = tables().lines[dir][static_cast<std::size_t>(lineIdx)];
    auto& f = lineFour_[dir][static_cast<std::size_t>(lineIdx)];
    auto& o = lineOpen3_[dir][static_cast<std::size_t>(lineIdx)];
    for (int color = 0; color < 2; ++color) {
        int nf = 0;
        int no = 0;
        countLine(bb, line, color, nf, no);
        fourThreats_[color] += nf - f[static_cast<std::size_t>(color)];
        openThrees_[color] += no - o[static_cast<std::size_t>(color)];
        f[static_cast<std::size_t>(color)] = nf;
        o[static_cast<std::size_t>(color)] = no;
    }
}

void EvalStateV2::reset(const BitBoard& bb) {
    base_.reset(bb);
    fourThreats_[0] = fourThreats_[1] = 0;
    openThrees_[0] = openThrees_[1] = 0;
    positional_[0] = positional_[1] = 0;
    posOwner_.fill(-1);
    for (int s = 0; s < BitBoard::SIZE; ++s) {
        if (!BitBoard::isPlayableSq(s) || bb.isEmpty(s)) continue;
        const signed char col = bb.isBlack(s) ? 0 : 1;
        posOwner_[static_cast<std::size_t>(s)] = col;
        positional_[col] += posTable()[static_cast<std::size_t>(s)];
    }
    for (int d = 0; d < NDIR; ++d) {
        const std::size_t n = tables().lines[d].size();
        lineFour_[d].assign(n, {0, 0});
        lineOpen3_[d].assign(n, {0, 0});
        for (std::size_t i = 0; i < n; ++i) {
            recomputeLine(bb, d, static_cast<int>(i));
        }
    }
}

void EvalStateV2::update(const BitBoard& bb, int sq) {
    base_.update(bb, sq);
    // update() runs after both place() and remove(). A removal does not tell
    // us which colour left, so the square's previous credit is recorded
    // rather than inferred.
    const int pv = posTable()[static_cast<std::size_t>(sq)];
    const signed char now =
        bb.isEmpty(sq) ? static_cast<signed char>(-1)
                       : static_cast<signed char>(bb.isBlack(sq) ? 0 : 1);
    signed char& was = posOwner_[static_cast<std::size_t>(sq)];
    if (was != now) {
        if (was >= 0) positional_[was] -= pv;
        if (now >= 0) positional_[now] += pv;
        was = now;
    }
    // Only the four lines through `sq` can have changed.
    for (int d = 0; d < NDIR; ++d) {
        const int idx = tables().lineOf[static_cast<std::size_t>(d)]
                                      [static_cast<std::size_t>(sq)];
        if (idx >= 0) recomputeLine(bb, d, idx);
    }
}

int EvalStateV2::forkBonus(int color) const {
    const int n4 = fourThreats_[color];
    const int n3 = openThrees_[color];
    if (n4 >= 2) return forkDoubleFour();
    if (n4 >= 1 && n3 >= 1) return forkFourThree();
    if (n3 >= 2) return forkDoubleThree();
    return 0;
}

int EvalStateV2::value(int mover) const {
    const int base = base_.value(mover);
    const int own = forkBonus(mover);
    const int opp = forkBonus(mover ^ 1);
    const int pos = (positional_[mover] - positional_[mover ^ 1]) * posWeight();
    return base + own - opp * DEF_NUM / DEF_DEN + pos + tempoBonus();
}

int evaluateV2(const BitBoard& bb, int mover) {
    EvalStateV2 st;
    st.reset(bb);
    return st.value(mover);
}

}  // namespace oog
