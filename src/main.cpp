// Minimal CLI driver.
// Reads moves as "<r>,<c>" pairs on stdin and prints the board; also exposes
// the search via `best [time-ms]` and the opening book via `book [path]`.
#include "bitboard.hpp"
#include "board.hpp"
#include "book.hpp"
#include "search.hpp"

#include <cstdio>
#include <iostream>
#include <string>

namespace {
void printHelp() {
    std::cout << "outer-gomoku CLI (Phase 4: rules + search + opening book)\n"
                 "Commands:\n"
                 "  r,c        play move at row r, column c (0-indexed)\n"
                 "  undo       undo last move\n"
                 "  new        reset the board\n"
                 "  print      print the board + state\n"
                 "  book       show the book move for this position\n"
                 "  book <f>   load an opening book file\n"
                 "  best [ms]  compute the engine's best move (default 1000 ms)\n"
                 "  help       show this help\n"
                 "  quit       exit\n";
}
}  // namespace

int main() {
    oog::Board board;
    oog::Search search;
    oog::OpeningBook book;
    bool haveBook = false;
    if (book.loadFile("openings.txt") > 0) {
        haveBook = true;
        std::cout << "opening book loaded: " << book.size() << " positions\n";
    }
    std::cout << board.toString() << "\n";
    printHelp();

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line.empty()) break;
        if (line == "help") {
            printHelp();
        } else if (line == "undo") {
            board.undoMove();
            std::cout << board.toString() << "\n";
        } else if (line == "new") {
            board.reset();
            std::cout << board.toString() << "\n";
        } else if (line == "print") {
            std::cout << board.toString() << "\n";
        } else if (line == "book" || line.rfind("book ", 0) == 0) {
            if (line.size() > 5) {
                const std::size_t n = book.loadFile(line.substr(5));
                haveBook = n > 0;
                std::cout << (haveBook ? "book loaded: " : "book load failed: ")
                          << book.size() << " positions\n";
            } else if (!haveBook) {
                std::cout << "no book loaded\n";
            } else {
                oog::BitBoard bb;
                bb.fillBoard(board);
                const int mv =
                    book.probe(bb, board.isBlackToMove() ? 0 : 1);
                if (mv < 0) {
                    std::cout << "no book move\n";
                } else {
                    std::cout << "book: " << oog::BitBoard::boardRow(mv) << ","
                              << oog::BitBoard::boardCol(mv) << "\n";
                }
            }
        } else if (line == "best" || line.rfind("best ", 0) == 0) {
            long long ms = 1000;
            if (line.size() > 5) ms = std::stoll(line.substr(5));
            if (board.isGameOver()) {
                std::cout << "game is over\n";
                continue;
            }
            oog::BitBoard bb;
            bb.fillBoard(board);
            oog::SearchLimits lim;
            lim.maxTimeMs = ms;
            const oog::SearchResult r =
                search.search(bb, board.isBlackToMove() ? 0 : 1, lim,
                              haveBook ? &book : nullptr);
            if (r.bestMove < 0) {
                std::cout << "no legal move\n";
                continue;
            }
            // bestMove is a BitBoard square: convert via the BitBoard layout.
            const int br = oog::BitBoard::boardRow(r.bestMove);
            const int bc = oog::BitBoard::boardCol(r.bestMove);
            std::cout << "best: " << br << "," << bc
                      << "  score=" << r.score
                      << "  depth=" << r.depth
                      << "  nodes=" << r.nodes
                      << (r.forcedWin ? "  (forced win)" : "")
                      << "\n";
        } else {
            const auto comma = line.find(',');
            if (comma == std::string::npos) {
                std::cout << "invalid move (expected \"r,c\")\n";
                continue;
            }
            const int r = std::stoi(line.substr(0, comma));
            const int c = std::stoi(line.substr(comma + 1));
            if (board.makeMove({r, c})) {
                std::cout << board.toString() << "\n"
                          << (board.winner() == oog::Cell::Black ? "Black wins"
                              : board.winner() == oog::Cell::White ? "White wins"
                              : board.isDraw()                      ? "Draw"
                                                                    : "")
                          << "\n";
            } else {
                std::cout << "illegal move\n";
            }
        }
    }
    return 0;
}