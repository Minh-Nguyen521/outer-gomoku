// Head-to-head comparison across the project's search strategies:
//   Search          - baseline alpha-beta, flat LMR
//   SearchV2        - baseline + rapfi-style multi-signal LMR
//   SearchV3        - SearchV2 + futility/move-count pruning + razoring-VCF
//   SearchMinibal   - baseline alpha-beta + Minibal_+ root selection rule
//   SearchUctPb     - MCTS/UCT + Progressive Bias (ADP substituted by eval)
//   SearchVct       - SearchV3 + Allis threat-space search extended to VCT
//   SearchNullKiller- SearchV3 + null-move pruning + killer/counter-move
//   SearchThreatCand- SearchV3 + threat-driven candidate generation
//   SearchV4        - fast bitboard threat primitive + the Round 5 ideas
//   SearchV5        - V4 + root safety screen (opponent VCF) + extensions
//
// Node/depth efficiency on fixed positions at a matched node budget, plus a
// full round-robin self-play tournament with deterministic node budgets (no
// wall-clock, so results are reproducible), reported as a standings table.
#include "bitboard.hpp"
#include "board.hpp"
#include "search.hpp"
#include "search_minibal.hpp"
#include "search_nullkiller.hpp"
#include "search_threatcand.hpp"
#include "search_uct_pb.hpp"
#include "search_v2.hpp"
#include "search_v3.hpp"
#include "search_v4.hpp"
#include "search_evalv2.hpp"
#include "search_q.hpp"
#include "search_v5.hpp"
#include "search_vct.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace oog;

namespace {

BitBoard toBitBoard(const Board& b) {
    BitBoard bb;
    bb.fillBoard(b);
    return bb;
}

// Every legal Black-first-move square (the outer 2 rows/columns of the
// 15x15 board -- 104 cells total), then an evenly-spaced sample of `count`
// of them (deterministic, spread across corners/edges/sides rather than
// clustered) so a large game count doesn't just replay identical games
// between these deterministic engines.
std::vector<std::pair<int, int>> generateRingOpenings(int count) {
    std::vector<std::pair<int, int>> all;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (r < 2 || r >= N - 2 || c < 2 || c >= N - 2) all.emplace_back(r, c);
        }
    }
    if (count >= static_cast<int>(all.size())) return all;
    std::vector<std::pair<int, int>> sample;
    sample.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const std::size_t idx = static_cast<std::size_t>(
            (static_cast<long long>(i) * static_cast<long long>(all.size())) / count);
        sample.push_back(all[idx]);
    }
    return sample;
}

// --- Type-erased engine handle -------------------------------------------
// The Search* classes share an API by convention, not by a base class (each
// is deliberately standalone so the baseline stays untouched). This wrapper
// gives the tournament loop a uniform handle without changing any of them.
struct EngineIface {
    virtual ~EngineIface() = default;
    virtual SearchResult search(BitBoard& bb, int color,
                                const SearchLimits& lim) = 0;
    virtual void clearTT() = 0;
};

template <typename E>
struct EngineImpl final : EngineIface {
    E engine;
    SearchResult search(BitBoard& bb, int color, const SearchLimits& lim) override {
        return engine.search(bb, color, lim);
    }
    void clearTT() override { engine.clearTT(); }
};

struct EngineDef {
    const char* name;
    std::function<std::unique_ptr<EngineIface>()> make;
    bool comparableDepth;  // false for MCTS (depth/score aren't minimax units)
};

template <typename E>
EngineDef def(const char* name, bool comparableDepth = true) {
    return {name, [] { return std::unique_ptr<EngineIface>(new EngineImpl<E>()); },
            comparableDepth};
}

// Optional restriction of the tournament field, set from the command line.
// Empty = run every engine.
std::vector<std::string>& engineFilter() {
    static std::vector<std::string> f;
    return f;
}

// Optional worker-thread override. Wall-clock budgets are sensitive to CPU
// contention, so a time-matched run wants fewer workers than cores.
unsigned& threadOverride() {
    static unsigned t = 0;
    return t;
}

const std::vector<EngineDef>& allEngines() {
    static const std::vector<EngineDef> v = {
        def<Search>("Search"),
        def<SearchV2>("SearchV2"),
        def<SearchV3>("SearchV3"),
        def<SearchMinibal>("SearchMinibal"),
        def<SearchUctPb>("SearchUctPb", false),
        def<SearchVct>("SearchVct"),
        def<SearchNullKiller>("SearchNullKiller"),
        def<SearchThreatCand>("SearchThreatCand"),
        def<SearchV4>("SearchV4"),
        def<SearchV5>("SearchV5"),
        def<SearchEvalV2>("SearchEvalV2"),
        def<SearchQ>("SearchQ"),
    };
    return v;
}

const std::vector<EngineDef>& engines() {
    static std::vector<EngineDef> filtered;
    static bool built = false;
    if (!built) {
        built = true;
        if (engineFilter().empty()) return allEngines();
        for (const auto& e : allEngines()) {
            for (const auto& want : engineFilter()) {
                if (want == e.name) {
                    filtered.push_back(e);
                    break;
                }
            }
        }
    }
    if (engineFilter().empty()) return allEngines();
    return filtered;
}

struct Pos {
    const char* label;
    std::vector<std::pair<int, int>> moves;  // (r, c), Black first
};

const std::vector<Pos> POSITIONS = {
    {"ring-open + center contact",
     {{0, 7}, {7, 7}, {7, 8}, {6, 7}, {8, 8}, {5, 6}}},
    {"corner open + contact",
     {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {3, 3}, {2, 1}}},
    {"edge open + spread",
     {{7, 0}, {7, 7}, {6, 5}, {8, 8}, {5, 5}, {9, 9}}},
};

void nodeEfficiencyTest(std::uint64_t nodeBudget) {
    std::printf("=== Node efficiency @ %llu node budget/side ===\n",
                static_cast<unsigned long long>(nodeBudget));
    for (const auto& p : POSITIONS) {
        Board board;
        for (const auto& mv : p.moves) board.makeMove({mv.first, mv.second});
        const BitBoard bb0 = toBitBoard(board);
        const int color = board.isBlackToMove() ? 0 : 1;

        std::printf("%-28s\n", p.label);
        for (const auto& e : engines()) {
            BitBoard bb = bb0;
            SearchLimits lim;
            lim.maxNodes = nodeBudget;
            auto eng = e.make();
            const SearchResult r = eng->search(bb, color, lim);
            if (e.comparableDepth) {
                std::printf("  %-18s depth=%3d score=%9d nodes=%9llu\n", e.name,
                            r.depth, r.score,
                            static_cast<unsigned long long>(r.nodes));
            } else {
                std::printf(
                    "  %-18s visits=%3d value*1e5=%9d nodes=%9llu"
                    "  (not comparable -- see docs/search-comparison.md)\n",
                    e.name, r.depth, r.score,
                    static_cast<unsigned long long>(r.nodes));
            }
        }
    }
    std::printf("\n");
}

// Per-move budget: either a deterministic node count (reproducible, but it
// does not charge an engine for extra work done *inside* a node) or a
// wall-clock cap (charges everything, but is not bit-reproducible). Engines
// added in Round 5 do measurable per-node scanning that a node budget is
// blind to, so both modes matter -- see docs/search-comparison.md.
struct Budget {
    std::uint64_t nodes = 0;
    long long ms = 0;
    SearchLimits limits() const {
        SearchLimits lim;
        if (ms > 0) lim.maxTimeMs = ms;
        else lim.maxNodes = nodes;
        return lim;
    }
    std::string label() const {
        return ms > 0 ? std::to_string(ms) + " ms/move"
                      : std::to_string(nodes) + " nodes/move";
    }
};

char playGame(EngineIface& a, EngineIface& b, bool aIsBlack,
              const Budget& budget, std::pair<int, int> opening) {
    Board board;
    board.makeMove({opening.first, opening.second});
    a.clearTT();
    b.clearTT();
    const SearchLimits lim = budget.limits();
    int plies = 1;
    while (!board.isGameOver() && plies < N * N) {
        const int color = board.isBlackToMove() ? 0 : 1;
        BitBoard bb = toBitBoard(board);
        const bool useA = (color == 0) == aIsBlack;
        const int mv = (useA ? a : b).search(bb, color, lim).bestMove;
        if (mv < 0) break;
        const int br = BitBoard::boardRow(mv);
        const int bc = BitBoard::boardCol(mv);
        if (!board.makeMove({br, bc})) break;
        ++plies;
    }
    if (board.winner() == Cell::Black) return aIsBlack ? 'A' : 'B';
    if (board.winner() == Cell::White) return aIsBlack ? 'B' : 'A';
    return 'D';
}

struct MatchResult {
    int aWins = 0;
    int bWins = 0;
    int draws = 0;
};

// Each Search-family class is documented (search.hpp) as safe for one
// instance per game table with no shared mutable state, so games run
// concurrently as long as each game gets its own fresh engine instances
// (not shared/reused across threads) -- exactly what this does.
//
// Every game of every pairing goes into one global task pool rather than one
// pool per pairing: with only a handful of games per pairing, a per-pairing
// pool leaves most cores idle waiting for the slowest game in the pairing.
void roundRobin(const Budget& matchBudget, int numOpenings) {
    const auto& es = engines();
    const std::size_t n = es.size();
    const std::vector<std::pair<int, int>> openings =
        generateRingOpenings(numOpenings);

    struct Task {
        std::size_t i;
        std::size_t j;
        std::pair<int, int> opening;
        bool aIsBlack;
    };
    std::vector<Task> tasks;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            for (const auto& o : openings) {
                tasks.push_back({i, j, o, true});
                tasks.push_back({i, j, o, false});
            }
        }
    }

    std::printf("=== Round-robin (%d openings x 2 colors = %d games/pairing, "
                "%zu engines, %zu pairings, %zu games total) ===\n",
                numOpenings, numOpenings * 2, n, n * (n - 1) / 2, tasks.size());
    std::printf("(progress: one '.' per completed game)\n");

    std::vector<char> results(tasks.size(), '?');
    std::atomic<std::size_t> nextIdx{0};
    std::atomic<int> completed{0};
    std::mutex printMutex;
    const int totalGames = static_cast<int>(tasks.size());

    const unsigned numThreads =
        threadOverride() > 0 ? threadOverride()
                             : std::max(1u, std::thread::hardware_concurrency());
    std::printf("(workers: %u)\n", numThreads);
    auto worker = [&]() {
        for (;;) {
            const std::size_t k = nextIdx.fetch_add(1);
            if (k >= tasks.size()) break;
            const Task& t = tasks[k];
            auto a = es[t.i].make();  // fresh instances per game -- no state
            auto b = es[t.j].make();  // shared across threads or games
            results[k] = playGame(*a, *b, t.aIsBlack, matchBudget, t.opening);
            const int done = completed.fetch_add(1) + 1;
            std::lock_guard<std::mutex> lock(printMutex);
            // Flush every dot. stdout is line-buffered, so without this a
            // partial line of progress sits in the buffer until the newline
            // at game 50 -- making a run that is working normally look
            // completely stalled for its first 49 games.
            std::printf(".");
            std::fflush(stdout);
            if (done % 50 == 0 || done == totalGames) {
                std::printf(" [%d/%d]\n", done, totalGames);
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(numThreads);
    for (unsigned t = 0; t < numThreads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    std::printf("\n\n");

    std::vector<std::vector<MatchResult>> mrs(n, std::vector<MatchResult>(n));
    for (std::size_t k = 0; k < tasks.size(); ++k) {
        MatchResult& mr = mrs[tasks[k].i][tasks[k].j];
        if (results[k] == 'A') ++mr.aWins;
        else if (results[k] == 'B') ++mr.bWins;
        else ++mr.draws;
    }

    std::vector<double> points(n, 0.0);
    std::vector<int> played(n, 0);
    std::vector<std::vector<std::string>> grid(n, std::vector<std::string>(n, "-"));

    std::printf("=== Pairing results @ %s ===\n", matchBudget.label().c_str());
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const MatchResult& mr = mrs[i][j];
            const int games = mr.aWins + mr.bWins + mr.draws;
            std::printf("  %-18s %2d - %-2d %-18s (draws: %d)\n", es[i].name,
                        mr.aWins, mr.bWins, es[j].name, mr.draws);
            points[i] += mr.aWins + 0.5 * mr.draws;
            points[j] += mr.bWins + 0.5 * mr.draws;
            played[i] += games;
            played[j] += games;
            grid[i][j] = std::to_string(mr.aWins) + "-" + std::to_string(mr.bWins);
            grid[j][i] = std::to_string(mr.bWins) + "-" + std::to_string(mr.aWins);
        }
    }
    std::printf("\n");

    std::printf("\n=== Standings (win = 1, draw = 0.5) ===\n");
    std::vector<std::size_t> order(n);
    for (std::size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return points[a] > points[b];
    });
    std::printf("%-4s %-18s %8s %8s %8s\n", "rank", "engine", "points", "games",
                "score%");
    for (std::size_t r = 0; r < n; ++r) {
        const std::size_t i = order[r];
        const double pct = played[i] > 0 ? 100.0 * points[i] / played[i] : 0.0;
        std::printf("%-4zu %-18s %8.1f %8d %7.1f%%\n", r + 1, es[i].name,
                    points[i], played[i], pct);
    }

    std::printf("\n=== Cross-table (row engine's W-L vs column engine) ===\n");
    std::printf("%-18s", "");
    for (std::size_t j = 0; j < n; ++j) std::printf("%10.9s", es[j].name);
    std::printf("\n");
    for (std::size_t i = 0; i < n; ++i) {
        std::printf("%-18.17s", es[i].name);
        for (std::size_t j = 0; j < n; ++j) std::printf("%10s", grid[i][j].c_str());
        std::printf("\n");
    }
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 4096);  // line-buffered: flushes
                                                   // regularly without a
                                                   // write syscall per byte
    std::uint64_t effBudget = 20000;
    Budget matchBudget;
    matchBudget.nodes = 20000;
    int numOpenings = 5;  // 5 openings x 2 colors = 10 games/pairing
    // argv[4] > 0 switches the tournament to a wall-clock budget per move,
    // which (unlike a node budget) charges each engine for the per-node work
    // its heuristics add.
    if (argc > 1) effBudget = std::stoull(argv[1]);
    if (argc > 2) matchBudget.nodes = std::stoull(argv[2]);
    if (argc > 3) numOpenings = std::stoi(argv[3]);
    if (argc > 4) matchBudget.ms = std::stoll(argv[4]);
    if (argc > 5) {  // argv[5]: comma-separated engine names to include
        std::string spec = argv[5];
        std::size_t pos = 0;
        while (pos <= spec.size()) {
            const std::size_t comma = spec.find(',', pos);
            const std::size_t end = comma == std::string::npos ? spec.size() : comma;
            if (end > pos) engineFilter().push_back(spec.substr(pos, end - pos));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    if (argc > 6) threadOverride() = static_cast<unsigned>(std::stoul(argv[6]));

    nodeEfficiencyTest(effBudget);
    roundRobin(matchBudget, numOpenings);
    return 0;
}
