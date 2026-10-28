#include "doctest.h"

#include "bitboard.hpp"
#include "piskvork.hpp"

#include <sstream>
#include <string>

using namespace oog;

namespace {

// Run a full protocol session (lines separated by '\n') through the brain,
// collecting every reply in order.  `in` doubles as the stream the brain uses
// to consume multi-line commands, exactly like the real main loop.
std::string session(const std::string& input, PiskvorkBrain& brain) {
    std::istringstream in(input);
    std::string out;
    std::string line;
    while (std::getline(in, line)) {
        const std::string reply = brain.handleLine(line, in);
        if (!reply.empty()) out += reply + "\n";
        if (brain.ended()) break;
    }
    return out;
}

bool onRing(int r, int c) {
    return r < 2 || r >= N - 2 || c < 2 || c >= N - 2;
}

bool parseMove(const std::string& s, int& x, int& y) {
    const auto comma = s.find(',');
    if (comma == std::string::npos) return false;
    try {
        x = std::stoi(s.substr(0, comma));
        y = std::stoi(s.substr(comma + 1));
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace

TEST_CASE("piskvork: START accepts 15, rejects others") {
    PiskvorkBrain brain;
    CHECK(session("START 15\n", brain) == "OK\n");
    CHECK(brain.started());

    PiskvorkBrain b2;
    CHECK(session("START 19\n", b2) == "ERROR unsupported board size 19\n");
}

TEST_CASE("piskvork: RECTSTART 15,15 accepted") {
    PiskvorkBrain brain;
    CHECK(session("RECTSTART 15,15\n", brain) == "OK\n");
    PiskvorkBrain b2;
    CHECK(session("RECTSTART 15,10\n", b2) ==
          "ERROR rectangular board unsupported\n");
}

TEST_CASE("piskvork: BEGIN replies with an outer-ring first move") {
    PiskvorkBrain brain;
    const std::string out = session("START 15\nINFO timeout_turn 100\nBEGIN\n",
                                    brain);
    CHECK(out.rfind("OK\n", 0) == 0);
    const std::string move = out.substr(3);  // last reply line
    int x = -1;
    int y = -1;
    REQUIRE(parseMove(move, x, y));
    CHECK(onRing(y, x));  // (x,y) protocol coords, r=y, c=x
    CHECK(brain.stones() == 1);
}

TEST_CASE("piskvork: TURN places the opponent stone and replies") {
    PiskvorkBrain brain;
    const std::string out =
        session("START 15\nINFO timeout_turn 100\nTURN 0,7\n", brain);
    CHECK(out.rfind("OK\n", 0) == 0);
    const std::string move = out.substr(3);
    int x = -1;
    int y = -1;
    REQUIRE(parseMove(move, x, y));
    CHECK(x >= 0);
    CHECK(x < N);
    CHECK(y >= 0);
    CHECK(y < N);
    CHECK(brain.stones() == 2);  // opponent + our reply
}

TEST_CASE("piskvork: TURN on an occupied square is rejected") {
    PiskvorkBrain brain;
    const std::string out =
        session("START 15\nINFO timeout_turn 100\nTURN 0,7\nTURN 0,7\n", brain);
    CHECK(out.rfind("OK\n", 0) == 0);
    CHECK(out.find("ERROR illegal move\n") != std::string::npos);
}

TEST_CASE("piskvork: BOARD rebuilds any position and replies") {
    // 1 black + 1 white stone, Black to move (equal counts).
    PiskvorkBrain brain;
    const std::string out = session(
        "START 15\n"
        "INFO timeout_turn 100\n"
        "BOARD\n"
        "0,7,1\n"
        "7,7,2\n"
        "DONE\n",
        brain);
    CHECK(out.rfind("OK\n", 0) == 0);
    const std::string move = out.substr(3);
    int x = -1;
    int y = -1;
    REQUIRE(parseMove(move, x, y));
    CHECK(x >= 0);
    CHECK(x < N);
    CHECK(y >= 0);
    CHECK(y < N);
    CHECK(brain.stones() == 3);
}

TEST_CASE("piskvork: INFO is acknowledged silently") {
    PiskvorkBrain brain;
    const std::string out = session(
        "START 15\nINFO timeout_match 600000\nINFO timeout_turn 5000\n"
        "INFO max_memory 83886080\nINFO rule 2\nINFO time_left 600000\n",
        brain);
    CHECK(out == "OK\n");
}

TEST_CASE("piskvork: unknown command replies UNKNOWN") {
    PiskvorkBrain brain;
    CHECK(session("START 15\nFROB 1,2\n", brain) == "OK\nUNKNOWN FROB\n");
}

TEST_CASE("piskvork: ABOUT reports engine info") {
    PiskvorkBrain brain;
    const std::string out = session("ABOUT\n", brain);
    CHECK(out.find("name=") != std::string::npos);
    CHECK(out.find("Outer-Open") != std::string::npos);
}

TEST_CASE("piskvork: END stops the session") {
    PiskvorkBrain brain;
    session("START 15\nEND\nRESTART\n", brain);
    CHECK(brain.ended());
    // Nothing after END is processed, so RESTART must not produce a reply.
}

TEST_CASE("piskvork: RESTART and BACK reset the game") {
    PiskvorkBrain brain;
    const std::string out = session(
        "START 15\nINFO timeout_turn 100\nBEGIN\nBACK\nRESTART\nBEGIN\n", brain);
    CHECK(out.find("OK\n") != std::string::npos);
    // After RESTART the second BEGIN starts from an empty board.
    CHECK(brain.stones() == 1);
}
