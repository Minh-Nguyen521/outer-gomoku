#include "eval_params.hpp"

#include <cstdlib>
#include <string>

namespace oog {

namespace {

// Read an integer override, leaving the default untouched if the variable is
// unset or unparseable. Silence on bad input is deliberate: a tuner spawns
// thousands of processes and a parse warning per spawn would drown the match
// output. A bad value simply means the default is used.
void readInt(const char* name, int& out) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return;
    try {
        out = std::stoi(v);
    } catch (...) {
        // keep default
    }
}

EvalParams loadParams() {
    EvalParams p;
    readInt("OOG_S_FIVE", p.five);
    readInt("OOG_S_OPEN_FOUR", p.openFour);
    readInt("OOG_S_CLOSED_FOUR", p.closedFour);
    readInt("OOG_S_OPEN_THREE", p.openThree);
    readInt("OOG_S_CLOSED_THREE", p.closedThree);
    readInt("OOG_S_OPEN_TWO", p.openTwo);
    readInt("OOG_S_CLOSED_TWO", p.closedTwo);
    readInt("OOG_DEF_NUM", p.defNum);
    readInt("OOG_DEF_DEN", p.defDen);
    if (p.defDen == 0) p.defDen = 1;  // never divide by zero
    return p;
}

}  // namespace

const EvalParams& evalParams() {
    static const EvalParams P = loadParams();
    return P;
}

}  // namespace oog
