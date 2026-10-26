// How much is better move ordering actually worth in SearchV5?
//
// WHY THIS EXISTS. The plan was to feed the distilled policy net's move
// prediction into the search as an ordering hint. The net agrees with its
// KataGomo teacher's best move 98.9% of the time, so the *hint* would be
// excellent. But before writing any net-to-engine plumbing, one number
// decides whether the whole idea is worth it: how much does SearchV5 stand
// to gain from perfect ordering in the first place?
//
// SearchV5 already orders with a TT move, killers, history, counter-moves,
// threat scores, and internal iterative deepening. If that stack is already
// near-optimal, a perfect hint saves nothing and the hybrid is dead.
//
// THE MEASUREMENT. Search each position twice without clearing the TT
// between runs. The second search finds a best move already stored for
// essentially every node it revisits, so its ordering is close to perfect
// everywhere. The cold/warm node ratio is therefore an UPPER BOUND on what
// any ordering scheme -- including a perfect oracle -- could buy.
//
// It is only an upper bound, and a loose one: the warm search also reuses
// TT *scores* and *bounds*, which cut nodes for reasons that have nothing to
// do with move ordering. So the true ordering-only gain is somewhere below
// this figure. That asymmetry is exactly what makes it a useful gate -- if
// even this inflated number is small, the idea cannot work, and we have
// spent no effort on inference plumbing to find out.
//
// Positions come from the harvested game records so they are real midgame
// positions from engine play, not synthetic ones.
//
// Usage:
//     oog_headroom <games.jsonl> [num_positions] [depth]

#include "bitboard.hpp"
#include "board.hpp"
#include "engines/search_v5.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace oog;

namespace {

// The harvest records are one JSON object per line with a "moves" array of
// [x, y] pairs. A full parser would be dead weight for two integer fields.
std::vector<std::pair<int, int>> parseMoves(const std::string& line) {
    std::vector<std::pair<int, int>> out;
    const std::size_t k = line.find("\"moves\"");
    if (k == std::string::npos) return out;
    std::size_t i = line.find('[', k);
    if (i == std::string::npos) return out;
    const std::size_t end = line.find("]]", i);
    while (i < line.size() && i != std::string::npos) {
        const std::size_t lb = line.find('[', i + 1);
        if (lb == std::string::npos || (end != std::string::npos && lb > end))
            break;
        int x = 0, y = 0;
        if (std::sscanf(line.c_str() + lb, "[%d,%d]", &x, &y) != 2) break;
        out.emplace_back(x, y);
        i = line.find(']', lb);
    }
    return out;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "data/harvest_all.jsonl";
    const int want = argc > 2 ? std::atoi(argv[2]) : 40;
    const int depth = argc > 3 ? std::atoi(argv[3]) : 8;

    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }

    std::printf("depth %d, positions from %s\n\n", depth, path);
    std::printf("%6s %12s %12s %8s\n", "pos", "cold", "warm", "ratio");

    std::vector<double> ratios;
    std::vector<std::string> seen;  // position signatures already measured
    std::string line;
    int done = 0;
    while (done < want && std::getline(in, line)) {
        const auto moves = parseMoves(line);
        // Take a midgame position: deep enough to be non-trivial, shallow
        // enough that the game is usually still undecided.
        if (moves.size() < 24) continue;
        // Engine-vs-engine games are deterministic and share long opening
        // prefixes, so a fixed ply would hand back the SAME position over and
        // over -- the first version of this printed one node count 13 times.
        // Vary the cut point, and skip any position already measured.
        const std::size_t take = 14 + static_cast<std::size_t>(done % 9) * 2;
        if (moves.size() < take + 2) continue;

        BitBoard bb;
        int color = 0;
        std::string sig;
        for (std::size_t i = 0; i < take; ++i) {
            const int sq = moves[i].second * N + moves[i].first;
            if (sq < 0 || sq >= BitBoard::SIZE) { color = -1; break; }
            bb.place(color, sq);  // (colour, square) -- not the other way round
            sig += std::to_string(sq);
            sig += ',';
            color ^= 1;
        }
        if (color < 0) continue;
        if (std::find(seen.begin(), seen.end(), sig) != seen.end()) continue;
        seen.push_back(sig);

        SearchLimits lim;
        lim.maxDepth = depth;

        SearchV5 s;
        // Cold: empty TT, ordering comes only from the heuristic stack.
        const SearchResult cold = s.search(bb, color, lim, nullptr);
        // Warm: same instance, TT retained, so almost every revisited node
        // already knows its best move.
        const SearchResult warm = s.search(bb, color, lim, nullptr);

        if (cold.nodes == 0 || warm.nodes == 0) continue;
        const double ratio = static_cast<double>(cold.nodes) /
                             static_cast<double>(warm.nodes);
        ratios.push_back(ratio);
        std::printf("%6d %12llu %12llu %7.2fx\n", done,
                    static_cast<unsigned long long>(cold.nodes),
                    static_cast<unsigned long long>(warm.nodes), ratio);
        ++done;
    }

    if (ratios.empty()) {
        std::fprintf(stderr, "no usable positions\n");
        return 1;
    }
    double sum = 0.0;
    for (double r : ratios) sum += r;
    std::printf("\npositions %zu   mean %.2fx   median %.2fx\n", ratios.size(),
                sum / static_cast<double>(ratios.size()), median(ratios));
    std::printf(
        "\nThis is an UPPER BOUND on the gain from perfect ordering (the warm\n"
        "search also reuses TT scores and bounds, which is not an ordering\n"
        "effect). Read it as: no hint scheme can do better than this.\n");
    return 0;
}
