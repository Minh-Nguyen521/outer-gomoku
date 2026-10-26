// Gomocup / Piskvork protocol entry point.
//
// Reads protocol lines on stdin and writes replies on stdout (flushed after
// every reply).  Optional args:
//   oog_piskvork [book-path] [--nodes N] [--engine search|v2|v3|vct|nullkiller|threatcand|v4|v5|evalv2|q]
// The book path defaults to "openings.txt" (next to the binary).  --nodes N
// applies a deterministic per-move node budget instead of time.  --engine
// picks the search (default: the baseline `Search`), so the same protocol
// binary can anchor any of our engines against an external opponent.
#include "piskvork.hpp"

#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    // The engine has to be known before the brain is constructed, so scan for
    // it first rather than folding it into the main argument loop.
    oog::EngineKind kind = oog::EngineKind::Baseline;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--engine" && i + 1 < argc) {
            if (!oog::engineKindFromName(argv[i + 1], kind)) {
                std::cerr << "unknown engine '" << argv[i + 1]
                          << "' (expected search|v2|v3|vct|nullkiller|threatcand|v4|v5|evalv2|q)\n";
                return 2;
            }
        }
    }

    oog::PiskvorkBrain brain(kind);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--nodes" && i + 1 < argc) {
            brain.setNodeLimit(static_cast<std::uint64_t>(
                std::stoull(argv[++i])));
        } else if (a == "--engine" && i + 1 < argc) {
            ++i;  // already handled above
        } else if (a.rfind("--", 0) != 0) {
            brain.loadBook(a);
        }
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        const std::string reply = brain.handleLine(line, std::cin);
        if (!reply.empty()) std::cout << reply << '\n' << std::flush;
        if (brain.ended()) break;
    }
    return 0;
}
