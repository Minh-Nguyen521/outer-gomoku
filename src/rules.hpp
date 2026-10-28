#pragma once

#include <cstdlib>

namespace oog {

// Is the Outer-Open first-move restriction in force?
//
// Outer-Open Gomoku confines BLACK'S FIRST MOVE to the outer two rows and
// columns; from move two on it is identical to freestyle. That single
// difference is why every freestyle engine in third_party/ can be used as an
// opponent here at all.
//
// The restriction has to be switchable because we play on both rulesets:
//   * Outer-Open  -- our variant, and the ICGA Computer Olympiad event.
//   * Freestyle   -- Gomocup, and botzone.org.cn, which has no ring rule.
// With the rule hard-coded on, the engine opens on the edge in freestyle,
// which is simply a weak move and costs rating for no reason.
//
// Defaults to ON so every measurement taken before this existed remains
// reproducible; set OOG_OUTER_OPEN=0 to play plain freestyle. Read once into
// a function-local static: this sits on the move-generation path, and the
// value cannot change during a game.
// The compile-time default. 1 (Outer-Open) everywhere except builds that
// target a freestyle host: the Botzone bundle defines this to 0 because that
// sandbox runs plain freestyle and offers no way to set an environment
// variable. The env var still wins when present, so local testing can switch
// either way without rebuilding.
#ifndef OOG_DEFAULT_OUTER_OPEN
#define OOG_DEFAULT_OUTER_OPEN 1
#endif

inline bool outerOpenRule()
{
    static const bool enabled = [] {
        const char* e = std::getenv("OOG_OUTER_OPEN");
        if (e != nullptr && e[0] != '\0') return e[0] != '0';
        return OOG_DEFAULT_OUTER_OPEN != 0;
    }();
    return enabled;
}

}  // namespace oog
