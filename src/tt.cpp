#include "tt.hpp"

#include <utility>

namespace oog {

TT::TT() : slots_(SIZE) {}

void TT::clear() {
    for (auto& e : slots_) e = Entry{};
    generation_ = 1;
}

void TT::newSearch() {
    ++generation_;
    if (generation_ == 0) {  // wrap-around: full clear
        clear();
        generation_ = 1;
    }
}

const TT::Entry* TT::probe(std::uint64_t key) const {
    ++probes_;
    const auto& e = slots_[static_cast<std::size_t>(key >> (64 - SHIFT))];
    if (e.key == key && e.generation == generation_) {
        ++hits_;
        return &e;
    }
    return nullptr;
}

void TT::store(std::uint64_t key, int depth, Flag flag, int score, int move) {
    auto& e = slots_[static_cast<std::size_t>(key >> (64 - SHIFT))];
    // Always replace entries from older generations or shallower entries.
    if (e.generation != generation_ || e.depth <= depth) {
        e.key = key;
        e.move = static_cast<std::uint32_t>(move);
        e.score = score;
        e.generation = generation_;
        e.depth = static_cast<std::uint8_t>(depth);
        e.flag = static_cast<std::uint8_t>(flag);
    }
}

}  // namespace oog
