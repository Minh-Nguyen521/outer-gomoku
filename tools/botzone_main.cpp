// Botzone entry point: play Gomoku on botzone.org.cn using this engine.
//
// WHY: every external opponent we have beats this engine at its floor --
// KataGomo at ONE visit (a single network forward pass, no search) won 20-0.
// Four handicap mechanisms all produced cliffs rather than dials, so locally
// we cannot obtain a score strictly between 0% and 100%. Botzone's pool is a
// university course platform full of weak-to-moderate bots, which is exactly
// the graded opposition we cannot build here. It also serves as an INDEPENDENT
// instrument: six protocol bugs in our own match manager produced convincing
// but false results this session, and an outside referee shares none of that
// code.
//
// PROTOCOL (Botzone "simple interaction", one process run per turn):
//   stdin  : one JSON object
//              {"requests":[...], "responses":[...], "data":.., "globaldata":..}
//            For Gomoku each element is {"x":N,"y":N}; the request for the
//            very first move as Black is {"x":-1,"y":-1}.
//   stdout : {"response":{"x":N,"y":N}}
//
// requests[i] is the opponent's move for turn i and responses[i] is ours, so
// replaying them in the order r0, o0, r1, o1, ... reconstructs the position.
//
// RULES NOTE: Botzone's Gomoku is freestyle 15x15 with overlines winning, and
// has NO outer-open ring restriction on Black's first move. That differs from
// our variant in move 1 only -- the same assumption every external engine in
// our ladder already relies on -- so results transfer for everything after the
// opening, and we simply do not exercise the ring rule there.
//
// LIMITS: 1s per turn (doubled on the first), 256MB, C++ time multiplier 1x.
// A node budget is used rather than a wall clock because it is deterministic
// and cannot overrun on hardware slower than ours.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bitboard.hpp"
#include "piskvork.hpp"

using namespace oog;

namespace {

// Botzone's Gomoku payload is only ever {"x":N,"y":N}, so a full JSON parser
// would be dead weight in a single-file submission. This pulls the integer
// after each "x" / "y" key inside one bracketed array.
struct XY {
    int x, y;
};

std::vector<XY> parseArray(const std::string& s, const std::string& key)
{
    std::vector<XY> out;
    const std::size_t k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return out;
    const std::size_t lb = s.find('[', k);
    if (lb == std::string::npos) return out;
    // Match the closing bracket of THIS array (objects inside contain no
    // nested arrays for this game, but be defensive anyway).
    int depth = 0;
    std::size_t rb = lb;
    for (std::size_t i = lb; i < s.size(); ++i) {
        if (s[i] == '[') ++depth;
        else if (s[i] == ']') {
            if (--depth == 0) { rb = i; break; }
        }
    }
    const std::string body = s.substr(lb, rb - lb + 1);

    std::size_t p = 0;
    while (true) {
        const std::size_t xk = body.find("\"x\"", p);
        if (xk == std::string::npos) break;
        const std::size_t yk = body.find("\"y\"", xk);
        if (yk == std::string::npos) break;
        int x = 0, y = 0;
        if (std::sscanf(body.c_str() + xk, "\"x\"%*[^-0-9]%d", &x) != 1) break;
        if (std::sscanf(body.c_str() + yk, "\"y\"%*[^-0-9]%d", &y) != 1) break;
        out.push_back({x, y});
        p = yk + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv)
{
    // Node budget: Botzone allows 1s/turn and their hardware is likely slower
    // than the machine this was tuned on, so stay well inside it. Override
    // with argv[1] when testing locally.
    std::uint64_t nodes = 250000;
    if (argc > 1) nodes = std::strtoull(argv[1], nullptr, 10);

    std::ostringstream buf;
    buf << std::cin.rdbuf();
    const std::string input = buf.str();

    const std::vector<XY> reqs = parseArray(input, "requests");
    const std::vector<XY> resp = parseArray(input, "responses");

    // {"x":-1,"y":-1} as the first request means we are Black and move first.
    const bool weAreBlack = !reqs.empty() && reqs[0].x < 0;
    const int ourColor = weAreBlack ? 0 : 1;
    const int oppColor = weAreBlack ? 1 : 0;

    PiskvorkBrain brain(EngineKind::Baseline);
    brain.setNodeLimit(nodes);

    // Drive the engine through its public Gomocup interface rather than
    // reaching into placeRaw/computeMove, which are private. This is also the
    // exact code path every measured match uses, so behaviour here matches the
    // ladder results instead of being a second, untested entry point.
    std::istringstream empty;
    brain.handleLine("START 15", empty);
    brain.handleLine("INFO rule 0", empty);
    brain.handleLine("INFO timeout_turn 1000", empty);

    // Build the BOARD block. Fields are RELATIVE to the receiving engine
    // (1 = own stone, 2 = opponent's) -- the same convention tools/match.py
    // sends, so the engine sees precisely the input shape it was measured on.
    std::ostringstream board;
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        if (reqs[i].x >= 0)
            board << reqs[i].x << "," << reqs[i].y << ",2\n";   // opponent
        if (i < resp.size() && resp[i].x >= 0)
            board << resp[i].x << "," << resp[i].y << ",1\n";   // ours
    }
    board << "DONE\n";
    std::istringstream boardIn(board.str());
    const std::string reply = brain.handleLine("BOARD", boardIn);

    // handleLine returns "x,y" for a move. Anything else (an ERROR line, or
    // an empty string) must NOT be forwarded: Botzone treats a malformed
    // response as an invalid move and forfeits the game, so fall back to a
    // legal square instead.
    int x = -1, y = -1;
    if (std::sscanf(reply.c_str(), "%d,%d", &x, &y) != 2 ||
        x < 0 || x >= N || y < 0 || y >= N) {
        x = 7;
        y = 7;
    }

    std::cout << "{\"response\":{\"x\":" << x << ",\"y\":" << y << "}}"
              << std::endl;
    return 0;
}
