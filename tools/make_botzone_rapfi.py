#!/usr/bin/env python3
"""Amalgamate rapfi's play path into ONE .cpp for a Botzone submission.

Botzone compiles a single uploaded source file, up to 4 MB. rapfi is 43
translation units (CMake's CORE_SOURCES) plus vendored libraries, so they have
to be inlined in dependency order.

Why rapfi and not AlphaGomoku: AlphaGomoku's inference path needs `minml`, a
separate library whose repository carries NO licence, so it cannot be
redistributed. rapfi is GPL-3 and every one of its eight vendored dependencies
is permissive -- cpptoml/cxxopts/flat.hpp/libnpy MIT, lz4 BSD, simde CC0,
thread-pool MIT, zip Unlicense. All of them live in Rapfi/external/, so there
is nothing to fetch.

Size budget (measured):
    core+game+search+eval+root   0.85 MB   <- the play path
    cpptoml + lz4                0.48 MB
    simde                       <=1.50 MB  <- only the headers actually used
    Botzone limit                4.00 MB

What this has to get right:

* Includes are RELATIVE ("../core/types.h"), so they resolve against the
  including file's own directory, not a single include root.
* Each header must be emitted exactly once even though dozens of files include
  it; `#pragma once` does not survive concatenation into one file.
* System includes (<vector>) must be hoisted above everything, because a
  header inlined at the point of use would otherwise appear after code.
* Vendored libraries are included as <lz4.h>, <cpptoml.h>, <simde/...>, so
  angle includes have to be checked against the external dirs BEFORE being
  treated as system headers.

GPL-3: rapfi is GPL-3 (Rapfi developers). The generated file carries upstream's
attribution, and submitting source is what the licence asks for.

Usage:
    tools/make_botzone_rapfi.py --out build/botzone_rapfi.cpp
    c++ -std=c++17 -O2 -o /tmp/bz build/botzone_rapfi.cpp   # verify
"""

import argparse
import os
import re
import sys

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^">]+)[">]')
PRAGMA_ONCE_RE = re.compile(r'^\s*#\s*pragma\s+once\s*$')

# CMake CORE_SOURCES, minus command modules a play-only build does not need.
#
# ORDER MATTERS. A .cpp that defines an explicit template specialization must
# be inlined BEFORE any .cpp that instantiates it -- otherwise the compiler
# reports "explicit specialization after instantiation". In separate
# translation units this never arises. command/benchmark.cpp runs a search and
# so instantiates MovePicker<ROOT>, which search/movepick.cpp specialises, so
# the whole engine core is emitted first and the command modules last.
CORE_SOURCES = """
core/compressor.cpp core/hash.cpp core/iohelper.cpp core/utils.cpp
core/platform.cpp core/version.cpp
game/board.cpp game/movegen.cpp game/pattern.cpp
eval/eval.cpp eval/evalconfig.cpp eval/evaluator.cpp
eval/mix9svqnnue.cpp
search/hashtable.cpp search/movepick.cpp search/opening.cpp
search/searchcommon.cpp search/searchconfig.cpp search/searchengine.cpp
search/searchoutput.cpp search/searchthread.cpp search/timecontrol.cpp
search/ab/history.cpp search/ab/search.cpp
search/mcts/node.cpp search/mcts/search.cpp
database/dbclient.cpp database/dbconfig.cpp database/dbutils.cpp
database/dbtypes.cpp database/renlib.cpp database/yxdbstorage.cpp
config.cpp internalConfig.cpp
command/argutils.cpp command/benchmark.cpp command/command.cpp
command/dbcommand.cpp command/gomocup.cpp
main.cpp
""".split()

# lz4 and xxhash are C libraries whose implementations live in .c files that
# nothing #includes, so amalgamating only the C++ leaves LZ4F_* undefined at
# link time. They are appended at the END of the bundle inside `extern "C"`,
# because the lz4 headers declare everything `extern "C"` and the definitions
# must match.
C_SOURCES = ["external/lz4/src/lz4_all.c", "external/lz4/src/xxhash.c"]


def drop_block(text, marker):
    """Remove a brace-balanced block that starts at `marker`."""
    i = text.find(marker)
    if i < 0:
        return text, False
    j = text.find("{", i)
    if j < 0:
        return text, False
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[:i] + text[k + 1:], True
        k += 1
    return text, False


# Source edits applied while inlining, keyed by path relative to Rapfi/.
#
# mix9svqnnue.cpp opens an anonymous namespace holding
# `using namespace Evaluation::mix9svq;`. That directive stays in effect for
# the REST of the translation unit, so concatenating mix10nnue.cpp after it
# makes Weight/FeatureDim/NumHeadBucket ambiguous between the two evaluators
# (20 compile errors). Separate translation units never saw it.
#
# We only ever load mix9svq (config.toml sets `type = "mix9svq"`), so mix10 is
# dropped entirely -- which also removes its weight-format code from the
# bundle.
_CFG_FIND_OLD = (
    "    else if (std::filesystem::exists(CommandLine::binaryDirectory / configPath))\n"
    "        resolvedConfigPath = CommandLine::binaryDirectory / configPath;\n"
)
_CFG_FIND_NEW = _CFG_FIND_OLD + (
    "    // Botzone: uploaded files land in ./data and there is no way to pass\n"
    "    // a --config flag, so look there before giving up to the internal\n"
    "    // config. Only the CONFIG needs redirecting -- getModelFullPath()\n"
    "    // already resolves weight files relative to the config file's own\n"
    "    // directory, so finding data/config.toml makes data/*.bin.lz4\n"
    "    // resolve for free.\n"
    "    else if (std::filesystem::exists(std::filesystem::path(\"data\") / configPath))\n"
    "        resolvedConfigPath = std::filesystem::path(\"data\") / configPath;\n"
    "    else if (std::filesystem::exists(CommandLine::binaryDirectory / \"data\" / configPath))\n"
    "        resolvedConfigPath = CommandLine::binaryDirectory / \"data\" / configPath;\n"
)

def _load_fs_shim():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "botzone_fs_shim.h")
    try:
        with open(p, "r") as fh:
            return fh.read()
    except OSError:
        print(f"  WARNING: missing {p}", file=sys.stderr)
        return ""


_FS_SHIM = _load_fs_shim()

PATCHES = {
    "command/command.cpp": [
        (_CFG_FIND_OLD, _CFG_FIND_NEW),
    ],
    # The adapter needs its own main(). rapfi's is renamed textually, in its
    # own file only -- a global #define would also rewrite the adapter's.
    "main.cpp": [
        ("int main(int argc, char *argv[])", "int rapfi_main(int argc, char *argv[])"),
        ("int main(int argc, char **argv)", "int rapfi_main(int argc, char **argv)"),
    ],
    # GCC 7: std::atomic<T>'s defaulted constructor is deleted unless T is
    # trivially default-constructible. Constructing from a value sidesteps it.
    "search/mcts/node.cpp": [
        (", bound()\n", ", bound(ValueBound {})\n"),
    ],
    # GCC 7 with _GLIBCXX_USE_CXX11_ABI=0 has no erase(iterator, const_iterator)
    # overload, so the two iterators must be the same constness.
    "database/dbtypes.cpp": [
        ("boardText.cend());", "boardText.end());"),
    ],
    "eval/evalconfig.cpp": [
        ('#include "mix10nnue.h"\n', ""),
        ("__DROP_BLOCK__", 'else if (cfg.type == "mix10") {'),
    ],
}


class Amalgamator:
    def __init__(self, root, extern_dirs):
        self.root = os.path.abspath(root)
        self.extern_dirs = [os.path.join(self.root, d) for d in extern_dirs]
        self.emitted = set()      # realpaths already inlined
        self.system = set()       # <vector> etc, hoisted to the top
        self.missing = set()      # angle includes we could not resolve
        self.out = []
        self.patched = []
        self.stats = {"files": 0, "bytes": 0}

    def resolve(self, name, quoted, origin_dir):
        """Return an absolute path for an include, or None if it is a system header."""
        if quoted:
            cand = os.path.normpath(os.path.join(origin_dir, name))
            if os.path.isfile(cand):
                return cand
        # Angle includes may still be vendored libraries (<lz4.h>, <simde/...>).
        for d in self.extern_dirs:
            cand = os.path.join(d, name)
            if os.path.isfile(cand):
                return cand
        # ...or project-root relative.
        cand = os.path.join(self.root, name)
        if os.path.isfile(cand):
            return cand
        return None

    def inline(self, path, depth=0):
        real = os.path.realpath(path)
        if real in self.emitted:
            return
        self.emitted.add(real)
        self.stats["files"] += 1

        rel = os.path.relpath(real, self.root)
        self.out.append(f"\n/* ==== {rel} ==== */\n")

        origin_dir = os.path.dirname(real)
        try:
            with open(real, "r", errors="replace") as fh:
                text = fh.read()
        except OSError as e:
            self.out.append(f"/* could not read {rel}: {e} */\n")
            return

        for find, repl in PATCHES.get(rel, []):
            if find == "__DROP_BLOCK__":
                text, ok = drop_block(text, repl)
                if ok:
                    self.patched.append(f"{rel}: dropped block `{repl[:40]}`")
            elif find in text:
                text = text.replace(find, repl)
                self.patched.append(f"{rel}: replaced `{find.strip()[:40]}`")
        lines = text.splitlines(keepends=True)

        for line in lines:
            if PRAGMA_ONCE_RE.match(line):
                # Meaningless once everything is one file, and emitting it
                # would be a hard error in some compilers.
                continue
            m = INCLUDE_RE.match(line)
            if not m:
                self.out.append(line)
                continue
            quoted = m.group(1) == '"'
            name = m.group(2)
            target = self.resolve(name, quoted, origin_dir)
            if target is None:
                if quoted:
                    self.missing.add(name)
                    self.out.append(f"/* UNRESOLVED: {line.strip()} */\n")
                else:
                    # Leave system includes exactly where they are. Hoisting
                    # them to the top strips them out of their platform
                    # guards: <Windows.h> sits inside `#if defined(_WIN32)`,
                    # and hoisted it becomes an unconditional include that
                    # fails to compile anywhere else. The same mistake broke
                    # the SlowRenju bundle with <mach-o/dyld.h>.
                    # Duplicates are harmless -- system headers self-guard.
                    if name == "filesystem":
                        # Replaced by the GCC 7 shim emitted at the top of the
                        # bundle. Botzone compiles with g++-7, which has no
                        # <filesystem> at all (GCC 8+), and its fixed link line
                        # has no -lstdc++fs so <experimental/filesystem> is out
                        # too.
                        continue
                    self.system.add(name)
                    self.out.append(line)
                continue
            self.inline(target, depth + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapfi", default="third_party/rapfi/Rapfi")
    ap.add_argument("--out", default="build/botzone_rapfi.cpp")
    a = ap.parse_args()

    root = os.path.abspath(a.rapfi)
    if not os.path.isdir(root):
        sys.exit(f"not found: {root}")

    externs = []
    ext_root = os.path.join(root, "external")
    if os.path.isdir(ext_root):
        for name in sorted(os.listdir(ext_root)):
            d = os.path.join(ext_root, name)
            if os.path.isdir(d):
                externs.append(os.path.join("external", name))
                # Most vendored libs put headers under include/ or src/.
                for sub in ("include", "src"):
                    if os.path.isdir(os.path.join(d, sub)):
                        externs.append(os.path.join("external", name, sub))

    am = Amalgamator(root, externs)
    for src in CORE_SOURCES:
        p = os.path.join(root, src)
        if os.path.isfile(p):
            am.inline(p)
        else:
            print(f"  missing source: {src}", file=sys.stderr)

    am.out.append('\n\nextern "C" {\n')
    # These may already have been "emitted" from inside a dead #if -- xxhash.h
    # carries `#if defined(XXH_INLINE_ALL) ... #include "xxhash.c"`, and this
    # bundler does not evaluate preprocessor conditionals, so it inlines the
    # body into a branch the compiler then discards. The emitted-set would
    # refuse to write it again here, where it is actually needed, leaving
    # _XXH32 undefined at link time. Forget them so they are emitted for real.
    for src in C_SOURCES:
        rp = os.path.realpath(os.path.join(root, src))
        am.emitted.discard(rp)
    for src in C_SOURCES:
        cp = os.path.join(root, src)
        if os.path.isfile(cp):
            am.inline(cp)
        else:
            print(f"  missing C source: {src}", file=sys.stderr)
    am.out.append('\n}  // extern "C"\n')

    # The Botzone adapter goes LAST and keeps its own main(). Botzone speaks
    # JSON ({"requests":[{"x":N,"y":N}]} -> {"response":{"x":N,"y":N}}), not
    # the Gomocup protocol, so without this the engine receives JSON it cannot
    # parse and answers nothing. Every other check -- compiling, loading
    # weights from ./data, playing real games -- exercises the Gomocup path,
    # which is exactly the interface Botzone does not use.
    adapter = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "botzone_rapfi_adapter.cpp")
    if os.path.isfile(adapter):
        am.out.append(f"\n/* ==== {os.path.basename(adapter)} ==== */\n")
        with open(adapter, "r", errors="replace") as fh:
            for line in fh:
                m = INCLUDE_RE.match(line)
                if m and m.group(1) == '"':
                    continue        # adapter's project includes already inlined
                am.out.append(line)
    else:
        print(f"  WARNING: adapter not found at {adapter}", file=sys.stderr)

    header = [
        "// Amalgamated rapfi for Botzone. Generated by tools/make_botzone_rapfi.py\n",
        "//\n",
        "// rapfi is free software licensed under the GNU General Public License\n",
        "// version 3 (see third_party/rapfi/Copying.txt). Copyright (C) Rapfi\n",
        "// developers. Upstream: https://github.com/dhbloo/rapfi\n",
        "//\n",
        "// Vendored dependencies inlined here and their licences:\n",
        "//   cpptoml, cxxopts, flat.hpp, libnpy, thread-pool  MIT\n",
        "//   lz4  BSD-2-Clause    simde  CC0-1.0    zip  Unlicense\n",
        "\n",
        "// lz4/xxhash expose their state structs only under these macros. In a\n",
        "// normal build each .c defines them before including its own header;\n",
        "// here the headers are inlined first, so they must be set up front or\n",
        "// every use of XXH32_state_t fails as an incomplete type.\n",
        "// Botzone compiles with -O2 -Wall -mavx -mavx2 -std=c++1z and NO\n",
        "// -DNDEBUG, so every assert() in rapfi stays live and any that fires\n",
        "// aborts the bot mid-game. rapfi is normally built Release; match\n",
        "// that here rather than shipping a debug engine to a 1s-per-turn\n",
        "// judge that treats a crash as a loss.\n",
        "#define NDEBUG 1\n\n",
        "// ---- std::filesystem shim for Botzone's g++-7 ----\n",
        _FS_SHIM,
        "// ---- end shim ----\n\n",
        "#define LZ4_STATIC_LINKING_ONLY\n",
        "#define LZ4F_STATIC_LINKING_ONLY\n",
        "#define XXH_STATIC_LINKING_ONLY\n",
        "\n",
    ]
    # System includes are emitted in place (see Amalgamator.inline), so the
    # header carries none. Keeping the set only for the stats line.

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.writelines(header)
        fh.writelines(am.out)

    size = os.path.getsize(a.out)
    print(f"wrote {a.out}")
    print(f"  files inlined : {am.stats['files']}")
    print(f"  system headers: {len(am.system)}")
    print(f"  size          : {size/1048576:.2f} MB  (Botzone limit 4.00 MB)")
    for p in am.patched:
        print(f"  patched: {p}")
    if am.missing:
        print(f"  UNRESOLVED quoted includes ({len(am.missing)}):")
        for n in sorted(am.missing)[:20]:
            print(f"    {n}")


if __name__ == "__main__":
    main()
