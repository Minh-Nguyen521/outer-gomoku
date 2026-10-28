#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "bitboard.hpp"

namespace oog {

// Root move ordering / direct-play source for the opening phase.
//
// Positions are keyed by positionKey(bb, sideToMove); responses are squares
// with relative weights (higher = preferred).  The text loader expands every
// line across the 8 dihedral symmetries of the board, so the curated file
// only needs to describe one orientation of each line.
class OpeningBook {
public:
    struct Entry {
        int move;
        int weight;
    };

    // Load book text (see src/openings.txt for the format).  Returns the
    // number of distinct positions added (after symmetry expansion).
    std::size_t loadText(const std::string& text);

    // Read a file and load it.  Returns 0 if the file cannot be read.
    std::size_t loadFile(const std::string& path);

    // Best weighted move for the position, or -1.
    int probe(const BitBoard& bb, int color) const;

    // All weighted candidates for the position, highest weight first.
    void probeAll(const BitBoard& bb, int color, std::vector<Entry>& out) const;

    std::size_t size() const { return table_.size(); }

private:
    std::unordered_map<std::uint64_t, std::vector<Entry>> table_;
};

}  // namespace oog
