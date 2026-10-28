#include "book.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>

namespace oog {

namespace {

// Map (r,c) under the idx-th symmetry of the 15x15 board.
void applySym(int idx, int r, int c, int& nr, int& nc) {
    switch (idx) {
        case 0: nr = r; nc = c; break;             // identity
        case 1: nr = c; nc = 14 - r; break;        // rotate 90
        case 2: nr = 14 - r; nc = 14 - c; break;   // rotate 180
        case 3: nr = 14 - c; nc = r; break;        // rotate 270
        case 4: nr = 14 - r; nc = c; break;        // mirror horizontal
        case 5: nr = r; nc = 14 - c; break;        // mirror vertical
        case 6: nr = c; nc = r; break;             // mirror main diagonal
        default: nr = 14 - c; nc = 14 - r; break;  // mirror anti-diagonal
    }
}

void trim(std::string& s) {
    s.erase(0, s.find_first_not_of(" \t"));
    s.erase(s.find_last_not_of(" \t") + 1);
}

bool parseCoord(const std::string& tok, int& r, int& c) {
    const auto comma = tok.find(',');
    if (comma == std::string::npos) return false;
    r = std::stoi(tok.substr(0, comma));
    c = std::stoi(tok.substr(comma + 1));
    return r >= 0 && r < N && c >= 0 && c < N;
}

// Split a sequence string ("7,1 7,7 ...") into board coords.
bool parseSequence(const std::string& s, std::vector<std::pair<int, int>>& out) {
    out.clear();
    std::istringstream is(s);
    std::string tok;
    while (is >> tok) {
        int r = 0;
        int c = 0;
        if (!parseCoord(tok, r, c)) return false;
        out.emplace_back(r, c);
    }
    return true;
}

}  // namespace

std::size_t OpeningBook::loadText(const std::string& text) {
    std::size_t added = 0;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto sep = line.find(':');
        if (sep == std::string::npos) continue;

        std::string seqStr = line.substr(0, sep);
        std::string respStr = line.substr(sep + 1);
        trim(seqStr);
        trim(respStr);

        std::vector<std::pair<int, int>> seq;
        if (!parseSequence(seqStr, seq)) continue;

        // Parse responses "r,c:w r,c" (whitespace-separated, weight optional).
        std::vector<std::pair<std::pair<int, int>, int>> resp;
        std::istringstream rs(respStr);
        std::string tok;
        while (rs >> tok) {
            int weight = 1;
            const auto colon = tok.rfind(':');
            if (colon != std::string::npos) {
                weight = std::stoi(tok.substr(colon + 1));
                tok = tok.substr(0, colon);
            }
            int r = 0;
            int c = 0;
            if (parseCoord(tok, r, c))
                resp.emplace_back(std::make_pair(r, c), weight);
        }
        if (resp.empty()) continue;

        // Expand across the 8 symmetries.  Per line, a position reached from
        // two different symmetries of the same line is added only once.
        std::unordered_set<std::uint64_t> seen;
        for (int sym = 0; sym < 8; ++sym) {
            BitBoard bb;
            int color = 0;
            bool ok = true;
            for (const auto& mv : seq) {
                int r = 0;
                int c = 0;
                applySym(sym, mv.first, mv.second, r, c);
                const int s = sqOf(r, c);
                if (!bb.isEmpty(s)) {
                    ok = false;
                    break;
                }
                bb.place(color, s);
                color ^= 1;
            }
            if (!ok) continue;
            const std::uint64_t key = positionKey(bb, color);
            if (!seen.insert(key).second) continue;

            auto& entries = table_[key];
            for (const auto& mv : resp) {
                int r = 0;
                int c = 0;
                applySym(sym, mv.first.first, mv.first.second, r, c);
                const int s = sqOf(r, c);
                if (!bb.isEmpty(s)) continue;  // response must be legal
                entries.push_back({s, mv.second});
            }
            if (!entries.empty()) ++added;
        }
    }
    return added;
}

std::size_t OpeningBook::loadFile(const std::string& path) {
    std::ifstream f(path);
    if (!f) return 0;
    std::ostringstream ss;
    ss << f.rdbuf();
    return loadText(ss.str());
}

void OpeningBook::probeAll(const BitBoard& bb, int color,
                           std::vector<Entry>& out) const {
    out.clear();
    const auto it = table_.find(positionKey(bb, color));
    if (it == table_.end()) return;
    out = it->second;
    std::stable_sort(out.begin(), out.end(),
                     [](const Entry& a, const Entry& b) {
                         return a.weight > b.weight;
                     });
}

int OpeningBook::probe(const BitBoard& bb, int color) const {
    std::vector<Entry> out;
    probeAll(bb, color, out);
    return out.empty() ? -1 : out[0].move;
}

}  // namespace oog
