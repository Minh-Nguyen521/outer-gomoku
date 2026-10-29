#include "piskvork.hpp"

#include <algorithm>
#include <cctype>
#include <istream>
#include <sstream>
#include <string>

#include "search_v4.hpp"
#include "search_evalv2.hpp"
#include "search_nullkiller.hpp"
#include "search_threatcand.hpp"
#include "search_v2.hpp"
#include "search_v3.hpp"
#include "search_vct.hpp"
#include "search_q.hpp"
#include "search_v5.hpp"

namespace oog {

namespace {

// Adapts any of the Search* classes (which share an API by convention, not by
// a base class) to the BrainEngine interface.
template <typename E>
class BrainEngineImpl final : public BrainEngine {
public:
    SearchResult search(BitBoard& bb, int color, const SearchLimits& limits,
                        const OpeningBook* book) override {
        return engine_.search(bb, color, limits, book);
    }
    void clearTT() override { engine_.clearTT(); }

private:
    E engine_;
};

}  // namespace

bool engineKindFromName(const std::string& name, EngineKind& out) {
    std::string s;
    s.reserve(name.size());
    for (const char ch : name) {
        s.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch))));
    }
    if (s == "search" || s == "baseline" || s == "v1") {
        out = EngineKind::Baseline;
        return true;
    }
    if (s == "v2" || s == "searchv2") {
        out = EngineKind::V2;
        return true;
    }
    if (s == "v3" || s == "searchv3") {
        out = EngineKind::V3;
        return true;
    }
    if (s == "vct" || s == "searchvct") {
        out = EngineKind::Vct;
        return true;
    }
    if (s == "nullkiller" || s == "searchnullkiller" || s == "nk") {
        out = EngineKind::NullKiller;
        return true;
    }
    if (s == "threatcand" || s == "searchthreatcand" || s == "tc") {
        out = EngineKind::ThreatCand;
        return true;
    }
    if (s == "v4" || s == "searchv4") {
        out = EngineKind::V4;
        return true;
    }
    if (s == "v5" || s == "searchv5") {
        out = EngineKind::V5;
        return true;
    }
    if (s == "evalv2" || s == "searchevalv2") {
        out = EngineKind::EvalV2;
        return true;
    }
    if (s == "q" || s == "searchq" || s == "quiesce") {
        out = EngineKind::Q;
        return true;
    }
    return false;
}

const char* engineKindName(EngineKind kind) {
    switch (kind) {
        case EngineKind::V2: return "SearchV2";
        case EngineKind::V3: return "SearchV3";
        case EngineKind::Vct: return "SearchVct";
        case EngineKind::NullKiller: return "SearchNullKiller";
        case EngineKind::ThreatCand: return "SearchThreatCand";
        case EngineKind::V4: return "SearchV4";
        case EngineKind::V5: return "SearchV5";
        case EngineKind::EvalV2: return "SearchEvalV2";
        case EngineKind::Q: return "SearchQ";
        case EngineKind::Baseline: break;
    }
    return "Search";
}

std::unique_ptr<BrainEngine> makeBrainEngine(EngineKind kind) {
    switch (kind) {
        case EngineKind::V2:
            return std::make_unique<BrainEngineImpl<SearchV2>>();
        case EngineKind::V3:
            return std::make_unique<BrainEngineImpl<SearchV3>>();
        case EngineKind::Vct:
            return std::make_unique<BrainEngineImpl<SearchVct>>();
        case EngineKind::NullKiller:
            return std::make_unique<BrainEngineImpl<SearchNullKiller>>();
        case EngineKind::ThreatCand:
            return std::make_unique<BrainEngineImpl<SearchThreatCand>>();
        case EngineKind::V4:
            return std::make_unique<BrainEngineImpl<SearchV4>>();
        case EngineKind::V5:
            return std::make_unique<BrainEngineImpl<SearchV5>>();
        case EngineKind::EvalV2:
            return std::make_unique<BrainEngineImpl<SearchEvalV2>>();
        case EngineKind::Q:
            return std::make_unique<BrainEngineImpl<SearchQ>>();
        case EngineKind::Baseline: break;
    }
    return std::make_unique<BrainEngineImpl<Search>>();
}

namespace {

// Parse a protocol coordinate token "x,y" (x = column, y = row) into board
// (r,c).  Returns true on success.
bool parseXY(const std::string& tok, int& r, int& c) {
    const auto comma = tok.find(',');
    if (comma == std::string::npos) return false;
    try {
        c = std::stoi(tok.substr(0, comma));
        r = std::stoi(tok.substr(comma + 1));
    } catch (...) {
        return false;
    }
    return r >= 0 && r < N && c >= 0 && c < N;
}

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

}  // namespace

PiskvorkBrain::PiskvorkBrain(EngineKind kind)
    : engineKind_(kind), engine_(makeBrainEngine(kind)) {
    book_.loadFile("openings.txt");
}

std::string PiskvorkBrain::handleLine(const std::string& line, std::istream& in) {
    if (ended_) return "";

    std::istringstream is(line);
    std::string cmd;
    is >> cmd;

    if (cmd == "START") {
        int size = 0;
        is >> size;
        if (size == N) {
            reset();
            started_ = true;
            return "OK";
        }
        return "ERROR unsupported board size " + std::to_string(size);
    }
    if (cmd == "RECTSTART") {
        // "RECTSTART <w>,<h>"
        std::string rest;
        std::getline(is, rest);
        int w = 0;
        int h = 0;
        const auto comma = rest.find(',');
        bool ok = comma != std::string::npos;
        if (ok) {
            try {
                w = std::stoi(rest.substr(0, comma));
                h = std::stoi(rest.substr(comma + 1));
            } catch (...) {
                ok = false;
            }
        }
        if (ok && w == N && h == N) {
            reset();
            started_ = true;
            return "OK";
        }
        return "ERROR rectangular board unsupported";
    }
    if (cmd == "RESTART") {
        reset();
        return "OK";
    }
    if (cmd == "BACK") {
        // Take back the engine's last move (and the opponent move that
        // preceded it, keeping side-to-move parity).
        if (history_.size() == 1) undo(1);
        else undo(2);
        return "OK";
    }
    if (cmd == "INFO") {
        std::string key;
        std::string value;
        is >> key >> value;
        std::int64_t v = 0;
        try {
            v = std::stoll(value);
        } catch (...) {
            return "";  // non-numeric value (e.g. folder path) -> ignore
        }
        if (key == "timeout_turn") {
            timeoutTurnMs_ = v;
            haveTimeoutTurn_ = true;
        } else if (key == "timeout_match") {
            timeoutMatchMs_ = v;
        } else if (key == "time_left") {
            timeLeftMs_ = v;
            haveTimeLeft_ = true;
        }
        // max_memory / game_type / rule / evaluate / folder are ignored.
        return "";
    }
    if (cmd == "TURN") {
        std::string rest;
        std::getline(is, rest);
        int r = 0;
        int c = 0;
        if (!parseXY(trim(rest), r, c)) return "ERROR bad move";
        if (!placeOpponent(r, c)) return "ERROR illegal move";
        return replyMove(computeMove());
    }
    if (cmd == "BEGIN") {
        return replyMove(computeMove());
    }
    if (cmd == "BOARD") {
        reset();
        std::string l;
        while (std::getline(in, l)) {
            const std::string t = trim(l);
            if (t.empty()) continue;
            if (t == "DONE") break;
            // "x,y,field" with field 1 = Black, 2 = White (3 = renju-only).
            const auto c1 = t.find(',');
            if (c1 == std::string::npos) return "ERROR bad board line";
            const auto c2 = t.find(',', c1 + 1);
            if (c2 == std::string::npos) return "ERROR bad board line";
            int c = 0;
            int r = 0;
            int f = 0;
            try {
                c = std::stoi(t.substr(0, c1));
                r = std::stoi(t.substr(c1 + 1, c2 - c1 - 1));
                f = std::stoi(t.substr(c2 + 1));
            } catch (...) {
                return "ERROR bad board line";
            }
            if (f < 1 || f > 2) return "ERROR bad board line";
            if (!placeRaw(r, c, f - 1)) return "ERROR bad board line";
        }
        return replyMove(computeMove());
    }
    if (cmd == "ABOUT") {
        return "name=\"Outer-Open Gomoku Engine\", version=\"0.4.0\", "
               "author=\"oog\", country=\"\"";
    }
    if (cmd == "END") {
        ended_ = true;
        return "";
    }
    return "UNKNOWN " + cmd;
}

void PiskvorkBrain::reset() {
    bb_.clear();
    history_.clear();
    stones_ = 0;
    lastMove_ = -1;
    lastScore_ = 0;
    engine_->clearTT();
}

bool PiskvorkBrain::placeOpponent(int r, int c) { return placeRaw(r, c, sideToMove()); }

bool PiskvorkBrain::placeRaw(int r, int c, int color) {
    if (r < 0 || r >= N || c < 0 || c >= N) return false;
    const int s = sqOf(r, c);
    if (!bb_.isEmpty(s)) return false;
    bb_.place(color, s);
    history_.push_back(s);
    ++stones_;
    return true;
}

int PiskvorkBrain::computeMove() {
    const int color = sideToMove();
    if (bb_.hasFive(0) || bb_.hasFive(1)) return -1;  // game already over
    SearchLimits lim;
    lim.maxTimeMs = moveBudgetMs();
    lim.maxNodes = nodeLimit_;
    const SearchResult r = engine_->search(bb_, color, lim, &book_);
    lastScore_ = r.score;
    if (r.bestMove < 0) return -1;
    const int mv = r.bestMove;
    if (!bb_.isEmpty(mv)) return -1;
    bb_.place(color, mv);
    history_.push_back(mv);
    ++stones_;
    lastMove_ = mv;
    return mv;
}

std::string PiskvorkBrain::replyMove(int mv) {
    if (mv < 0) return "ERROR no legal move";
    const int r = BitBoard::boardRow(mv);
    const int c = BitBoard::boardCol(mv);
    return std::to_string(c) + "," + std::to_string(r);
}

void PiskvorkBrain::undo(int n) {
    n = std::min(n, static_cast<int>(history_.size()));
    while (n-- > 0) {
        const int s = history_.back();
        history_.pop_back();
        const int color = stones_ % 2 == 0 ? 1 : 0;  // last placed color
        bb_.remove(color, s);
        --stones_;
    }
}

long long PiskvorkBrain::moveBudgetMs() const {
    constexpr long long kDefaultTurn = 5000;
    constexpr std::int64_t kMinBudget = 50;
    constexpr long long kMaxBudget = 60000;
    constexpr long long kUnlimited = 2147483647;

    long long budget = kDefaultTurn;
    if (haveTimeoutTurn_) {
        // Spec: timeout_turn == 0 means "play as fast as possible".
        budget = timeoutTurnMs_ > 0 ? timeoutTurnMs_ : kMinBudget;
    }
    if (haveTimeLeft_ && timeLeftMs_ > 0 && timeLeftMs_ < kUnlimited) {
        // Pace for a ~30-move game; never exceed the turn cap.
        const long long share = std::max(kMinBudget, timeLeftMs_ / 30);
        budget = std::min(budget, share);
    } else if (timeoutMatchMs_ > 0) {
        const long long share = std::max(kMinBudget, timeoutMatchMs_ / 30);
        budget = std::min(budget, share);
    }
    // Reserve a small margin so the reply lands inside the strict turn limit
    // even with pipe latency and clock skew (managers kill at the limit).
    budget = std::max(kMinBudget, budget - std::max<long long>(10, budget / 50));
    return std::clamp(budget, kMinBudget, kMaxBudget);
}

}  // namespace oog
