#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace oog {

// Bounded transposition table for the search.
//
// Keyed by the stones-only Zobrist hash with the side to move folded in by
// the search.  Scores are stored from the perspective of the side to move at
// the node that wrote them (negamax convention); mate scores are stored
// adjusted for the absolute distance from the root so they remain valid when
// the table is hit at a different ply.
//
// Replacement: prefer a deeper entry; otherwise overwrite if the slot holds an
// entry from an older search generation.  A generation-stale entry is always
// overwritten.
class TT {
public:
    static constexpr std::uint32_t SHIFT = 20;          // 2^20 slots
    static constexpr std::size_t SIZE = std::size_t{1} << SHIFT;

    enum Flag : std::uint8_t { Exact = 0, Lower = 1, Upper = 2 };

    struct Entry {
        std::uint64_t key = 0;
        std::uint32_t move = 0xFFFFFFFFu;  // best move sq, or sentinel
        std::int32_t score = 0;
        std::uint32_t generation = 0;      // search age for aging
        std::uint8_t depth = 0;
        std::uint8_t flag = Exact;
        std::uint16_t pad = 0;
    };
    static_assert(sizeof(Entry) == 24);

    TT();

    void clear();
    void newSearch();  // bump generation (call once per root search)

    // Returns the slot for key (may be empty or stale).  The caller verifies
    // `key == e->key && generation == e->generation`.
    const Entry* probe(std::uint64_t key) const;

    void store(std::uint64_t key, int depth, Flag flag, int score, int move);

    std::uint64_t hits() const { return hits_; }
    std::uint64_t probes() const { return probes_; }

private:
    std::uint32_t generation_ = 1;
    std::vector<Entry> slots_;  // SIZE entries, heap-allocated (~24 MB)
    mutable std::uint64_t hits_ = 0;
    mutable std::uint64_t probes_ = 0;
};

}  // namespace oog
