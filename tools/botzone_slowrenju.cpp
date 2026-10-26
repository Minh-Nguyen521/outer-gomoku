// Botzone entry point wrapping SlowRenju.
//
// SlowRenju 5.1.3 (Tianyi Hao, https://www.wind23.com/, GPL-3) is the
// strongest engine in this project that can actually be SUBMITTED to a
// sandbox: 18-2 against rapfi-nnue and 20-0 against Rapfi-2018, with no
// weight files and ~148 KB of source. Every stronger engine we measured is
// either binary-only (VIBEFIVE, STARPOINT, SKYZERO, AlphaGomoku) or needs a
// net far too large to paste into a source form (KataGomo 93 MB, rapfi
// 38 MB).
//
// GPL-3: this adapter and the engine are distributed together as source,
// which is what the licence asks for. Upstream attribution is preserved.
//
// THE ENGINE RUNS IN A CHILD PROCESS. Three properties make embedding it
// in-process unworkable, each found the hard way:
//   * its protocol loop is inline in main() and reads std::cin directly;
//   * it answers from a DETACHED search thread, so returning from main()
//     races the move;
//   * it prints with printf_s -- C stdio -- so swapping std::cout's rdbuf
//     captures nothing and the reply escapes to the real stdout.
// An in-process version segfaulted when the parent tore down the stringstream
// a live engine thread was still reading. A fork gives it its own stdio and
// lets us kill it cleanly.
//
// Botzone protocol: one JSON object in, one out.
//   in : {"requests":[{"x":N,"y":N},...], "responses":[...]}
//        the first request is {"x":-1,"y":-1} when we are Black and move first
//   out: {"response":{"x":N,"y":N}}
//
// Botzone's Gomoku is freestyle 15x15 with no outer-open restriction, which
// differs from our variant in move 1 only. SlowRenju plays freestyle natively.

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

// Signature must match exactly: SlowRenju's main takes (int, char**), so a
// no-arg declaration mangles to a different symbol and fails to link.
int slowrenju_main(int, char**);

namespace {

constexpr int BZ_N = 15;

struct XY {
    int x, y;
};

// Botzone's Gomoku payload is only ever {"x":BZ_N,"y":BZ_N}; a full JSON parser
// would be dead weight in a single-file submission.
std::vector<XY> parseArray(const std::string& s, const std::string& key)
{
    std::vector<XY> out;
    const std::size_t k = s.find("\"" + key + "\"");
    if (k == std::string::npos) return out;
    const std::size_t lb = s.find('[', k);
    if (lb == std::string::npos) return out;
    int depth = 0;
    std::size_t rb = lb;
    for (std::size_t i = lb; i < s.size(); ++i) {
        if (s[i] == '[') ++depth;
        else if (s[i] == ']' && --depth == 0) { rb = i; break; }
    }
    const std::string body = s.substr(lb, rb - lb + 1);
    std::size_t p = 0;
    while (true) {
        const std::size_t xk = body.find("\"x\"", p);
        if (xk == std::string::npos) break;
        const std::size_t yk = body.find("\"y\"", xk);
        if (yk == std::string::npos) break;
        int x = 0, y = 0;
        if (std::sscanf(body.c_str() + xk, "\"x\"%*[^-0-9]%d", &x) != 1) break;
        if (std::sscanf(body.c_str() + yk, "\"y\"%*[^-0-9]%d", &y) != 1) break;
        out.push_back({x, y});
        p = yk + 1;
    }
    return out;
}

}  // namespace

int main()
{
    std::ostringstream in;
    in << std::cin.rdbuf();
    const std::string input = in.str();

    const std::vector<XY> reqs = parseArray(input, "requests");
    const std::vector<XY> resp = parseArray(input, "responses");

    std::ostringstream cmd;
    cmd << "START 15\n"
        << "INFO rule 0\n"
        // Botzone allows 1s per turn (2s on the first) and that ceiling is a
        // property of the compiler, not something a submission can raise. So
        // the only lever is using more of the second we already have.
        //
        // This started at 800ms, was cut to 500ms out of caution about
        // Botzone's hardware being slower than the machine it was measured
        // on, and is now back at 800ms: five complete games there (three vs
        // rapfi/Jax, one vs a human, one reverse-colour) produced no timeout
        // and no illegal move at 500ms, so the margin was larger than needed.
        // A timeout is still an instant loss, hence the parent deadline below
        // stays under the 1s wall.
        << "INFO timeout_turn 800\n"
        << "INFO timeout_match 100000\n"
        << "INFO time_left 100000\n"
        << "BOARD\n";
    std::vector<XY> occupied;
    for (std::size_t i = 0; i < reqs.size(); ++i) {
        if (reqs[i].x >= 0) {
            cmd << reqs[i].x << "," << reqs[i].y << ",2\n";
            occupied.push_back(reqs[i]);
        }
        if (i < resp.size() && resp[i].x >= 0) {
            cmd << resp[i].x << "," << resp[i].y << ",1\n";
            occupied.push_back(resp[i]);
        }
    }
    cmd << "DONE\n";
    const std::string session = cmd.str();

    // Run the engine in a CHILD PROCESS rather than in-process. It keeps
    // global state, spawns detached search threads and writes with printf --
    // an in-process version segfaulted when the parent tore down the
    // stringstream that a still-running engine thread was reading. A child
    // owns its own stdio and dies cleanly when we are done with it.
    int to_child[2], from_child[2];
    if (pipe(to_child) != 0 || pipe(from_child) != 0) {
        std::printf("{\"response\":{\"x\":7,\"y\":7}}\n");
        return 0;
    }
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(to_child[0], STDIN_FILENO);
        dup2(from_child[1], STDOUT_FILENO);
        close(to_child[1]); close(from_child[0]);
        setvbuf(stdout, nullptr, _IONBF, 0);
        // The PARENT already read all of stdin to EOF before forking, and the
        // child inherits that stream state: both stdio's stdin and std::cin
        // still have their EOF flags set, so the engine's read loop would see
        // EOF immediately no matter what fd 0 now points at. Clear both.
        clearerr(stdin);
        std::cin.clear();
        static char arg0[] = "slowrenju";
        static char* argv_[] = {arg0, nullptr};
        slowrenju_main(1, argv_);
        _exit(0);
    }
    close(to_child[0]); close(from_child[1]);
    ssize_t ignored = write(to_child[1], session.data(), session.size());
    (void)ignored;
    // Leave the child's stdin OPEN: closing it makes the engine's read loop
    // hit EOF and return before its detached search thread has printed.

    fcntl(from_child[0], F_SETFL, O_NONBLOCK);
    std::string out;
    int mx = -1, my = -1;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(900);
    const std::regex movere(R"((\d+),(\d+))");
    while (std::chrono::steady_clock::now() < deadline) {
        char buf[4096];
        ssize_t n;
        while ((n = read(from_child[0], buf, sizeof buf)) > 0)
            out.append(buf, static_cast<std::size_t>(n));
        // Only lines that are JUST "x,y" are moves; MESSAGE lines are noise.
        std::istringstream ls(out);
        std::string line;
        while (std::getline(ls, line)) {
            std::smatch m;
            if (std::regex_match(line, m, movere)) {
                mx = std::stoi(m[1]); my = std::stoi(m[2]);
            }
        }
        if (mx >= 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    kill(pid, SIGKILL);
    close(to_child[1]); close(from_child[0]);

    bool bad = (mx < 0 || mx >= BZ_N || my < 0 || my >= BZ_N);
    if (!bad)
        for (const XY& o : occupied)
            if (o.x == mx && o.y == my) { bad = true; break; }
    if (bad) {
        for (int y = 0; y < BZ_N && bad; ++y)
            for (int x = 0; x < BZ_N && bad; ++x) {
                bool used = false;
                for (const XY& o : occupied)
                    if (o.x == x && o.y == y) { used = true; break; }
                if (!used) { mx = x; my = y; bad = false; }
            }
    }
    std::printf("{\"response\":{\"x\":%d,\"y\":%d}}\n", mx, my);
    return 0;
}
