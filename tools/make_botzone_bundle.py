#!/usr/bin/env python3
"""Amalgamate the engine into ONE .cpp file for botzone.org.cn submission.

Botzone compiles a single source file uploaded through its web form, but this
engine is 21 translation units plus 21 headers. This inlines them in
dependency order and strips the local #include lines, leaving a file that
compiles standalone.

Rules followed:
  * every local header is emitted exactly once, before anything that needs it
    (topological order over `#include "..."` edges), with `#pragma once`
    removed since inclusion is now literal;
  * system includes (`#include <...>`) are hoisted, de-duplicated and sorted
    to the top, so ordering between translation units cannot matter;
  * `src/main.cpp` and `src/main_v3.cpp` are excluded -- they define their own
    `main()` and would collide with the Botzone entry point;
  * anonymous namespaces are left alone. They are file-local by definition,
    and concatenating two files that each declare `namespace { int x; }` is
    still legal: the merged file simply has one anonymous namespace. Symbols
    with the same name in two different files WOULD collide, so the script
    checks for that and reports rather than producing a file that fails to
    compile with a confusing error.

Usage:
    tools/make_botzone_bundle.py --out build/botzone_bundle.cpp
    c++ -std=c++17 -O2 -o /tmp/bundle build/botzone_bundle.cpp   # verify
"""

import argparse
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = "tools/botzone_main.cpp"
EXCLUDE_CPP = {"src/main.cpp", "src/main_v3.cpp"}

KEYWORDS = {'if', 'for', 'while', 'switch', 'return', 'else', 'do',
            'catch', 'sizeof', 'static_cast', 'const_cast', 'throw',
            'reinterpret_cast', 'dynamic_cast', 'operator', 'namespace'}

INC_LOCAL = re.compile(r'^\s*#\s*include\s+"([^"]+)"[^\n]*$', re.M)
INC_SYS = re.compile(r'^\s*#\s*include\s+<([^>]+)>[^\n]*$', re.M)
PRAGMA_ONCE = re.compile(r'^\s*#\s*pragma\s+once\s*$', re.M)


def find(name, searched):
    """Resolve a local include to a path under src/ or src/engines/."""
    for d in ("src", "src/engines", "tools", ""):
        p = os.path.join(ROOT, d, name)
        if os.path.isfile(p):
            return os.path.relpath(p, ROOT)
    searched.add(name)
    return None


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8", errors="replace") as fh:
        return fh.read()


def strip(text):
    """Remove local includes and pragma once; return (body, system_includes)."""
    sys_incs = set(INC_SYS.findall(text))
    text = INC_LOCAL.sub("", text)
    text = INC_SYS.sub("", text)
    text = PRAGMA_ONCE.sub("", text)
    return text, sys_incs


def header_order(headers, missing):
    """Topologically sort headers so dependencies come first."""
    deps = defaultdict(list)
    for h in headers:
        for inc in INC_LOCAL.findall(read(h)):
            r = find(inc, missing)
            if r and r.endswith((".h", ".hpp")):
                deps[h].append(r)
    out, seen, stack = [], set(), set()

    def visit(h):
        if h in seen:
            return
        if h in stack:                      # cycle: emit anyway, header guards
            return                          # in the originals made this work
        stack.add(h)
        for d in deps.get(h, []):
            visit(d)
        stack.discard(h)
        seen.add(h)
        out.append(h)

    for h in sorted(headers):
        visit(h)
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/botzone_bundle.cpp")
    ap.add_argument("--all-engines", action="store_true",
                    help="bundle all 12 engine variants. Off by default: they "
                         "each define a file-local adjustToStore(), and file-"
                         "local symbols from different translation units merge "
                         "into ONE anonymous namespace when amalgamated, so "
                         "they collide. The entry point only ever constructs "
                         "EngineKind::Baseline anyway.")
    cfg = ap.parse_args(argv)

    import glob
    cpps = [ENTRY] + [
        os.path.relpath(p, ROOT)
        for p in sorted(glob.glob(os.path.join(ROOT, "src", "*.cpp")) +
                        glob.glob(os.path.join(ROOT, "src", "engines", "*.cpp")))
        if os.path.relpath(p, ROOT) not in EXCLUDE_CPP
    ]
    if not cfg.all_engines:
        # Keep only the Baseline engine's translation unit.
        cpps = [c for c in cpps
                if not c.startswith("src/engines/") or c == "src/engines/search.cpp"]
    headers = [os.path.relpath(p, ROOT)
               for p in sorted(glob.glob(os.path.join(ROOT, "src", "*.hpp")) +
                               glob.glob(os.path.join(ROOT, "src", "engines", "*.hpp")))]

    missing = set()
    ordered_headers = header_order(headers, missing)

    # Find symbols defined inside an anonymous namespace in more than one
    # translation unit. Once amalgamated they all land in ONE anonymous
    # namespace, so same-named entities either redefine each other or -- more
    # insidiously -- become overloads and silently change which one a call
    # resolves to. book.cpp's `void trim(std::string&)` and piskvork.cpp's
    # `std::string trim(const std::string&)` did exactly that: `trim(rest)`
    # started resolving to the void overload.
    def anon_blocks(body):
        """Yield the contents of each anonymous namespace, brace-matched."""
        for m in re.finditer(r'namespace\s*\{', body):
            start = m.end() - 1
            depth, i = 0, start
            while i < len(body):
                if body[i] == '{':
                    depth += 1
                elif body[i] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            yield body[start + 1:i]

    local_syms = defaultdict(set)
    for c in cpps:
        for blk in anon_blocks(read(c)):
            # Namespace-scope FUNCTION definitions only. An earlier, looser
            # pattern also matched `if (`, loop variables and locals, and
            # renaming `if` produced a spectacularly broken file. Require a
            # return type, little indentation (so we stay at namespace scope
            # rather than inside a function body), and exclude keywords.
            for name in re.findall(
                    r'(?m)^[ \t]{0,4}(?:static\s+|inline\s+|constexpr\s+)*'
                    r'(?:[A-Za-z_][\w:<>]*\s*[*&]?\s+)+'
                    r'([a-zA-Z_]\w*)\s*\(', blk):
                if name in KEYWORDS:
                    continue
                local_syms[name].add(c)
    clashes = {k: sorted(v) for k, v in local_syms.items() if len(v) > 1}

    # Rename them per file so each keeps its own. Word-boundary replacement is
    # confined to the files that define the symbol locally, so it cannot touch
    # an unrelated identifier elsewhere.
    renames = {}
    for name, owners in clashes.items():
        for c in owners:
            renames.setdefault(c, {})[name] = f"{name}__{os.path.basename(c)[:-4]}"

    parts = ["// GENERATED by tools/make_botzone_bundle.py -- do not edit.\n"
             "// Amalgamation of the Outer-Open Gomoku engine for botzone.org.cn.\n"
             "//\n"
             "// Botzone plays PLAIN FREESTYLE: there is no Outer-Open ring\n"
             "// restriction on Black's first move, and its sandbox gives no way to\n"
             "// set an environment variable. So the rule is switched off at compile\n"
             "// time here. With it on, the engine opens on the board edge, which is\n"
             "// simply a weak move and gives away rating for nothing.\n"
             "//\n"
             "// The env var still overrides this if one is set, so the same file can\n"
             "// be tested locally under either ruleset.\n"
             "#define OOG_DEFAULT_OUTER_OPEN 0\n"]

    sys_incs = set()
    bodies = []
    DEFINE = re.compile(r'(?m)^\s*#\s*define\s+([A-Za-z_]\w*)')
    for f in ordered_headers + cpps:
        body, incs = strip(read(f))
        for old, new in renames.get(f, {}).items():
            body = re.sub(r'\b' + re.escape(old) + r'\b', new, body)
        sys_incs |= incs
        chunk = f"\n// ===== {f} =====\n{body}"
        # A #define in a .cpp is file-scoped in a normal build but leaks into
        # everything that follows once concatenated. eval.cpp's
        #     #define DEF_NUM (evalParams().defNum)
        # turned eval_v2.cpp's `constexpr int DEF_NUM = 3;` into a syntax
        # error this way. Undefine each file's macros at its end. Headers are
        # left alone: their macros are meant to be visible downstream.
        if f.endswith(".cpp"):
            macros = [m for m in DEFINE.findall(body)]
            if macros:
                chunk += ("\n// end of " + f + ": drop its file-local macros\n" +
                          "".join(f"#undef {m}\n" for m in sorted(set(macros))))
        bodies.append(chunk)

    if not cfg.all_engines:
        # makeBrainEngine switches over every EngineKind and references classes
        # whose definitions we just dropped, which would be a link error.
        # Replace its body: the Botzone entry point only asks for Baseline.
        for i, b in enumerate(bodies):
            key = "std::unique_ptr<BrainEngine> makeBrainEngine(EngineKind kind) {"
            j = b.find(key)
            if j < 0:
                continue
            start = j + len(key) - 1
            depth, k = 0, start
            while k < len(b):
                if b[k] == "{":
                    depth += 1
                elif b[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            bodies[i] = (b[:start] +
                         "{\n    // Bundle contains only the Baseline engine;\n"
                         "    // every other variant was dropped to avoid merging\n"
                         "    // their file-local helpers into one namespace.\n"
                         "    (void)kind;\n"
                         "    return std::make_unique<BrainEngineImpl<Search>>();\n}" +
                         b[k + 1:])
            break

    parts.append("".join(f"#include <{h}>\n" for h in sorted(sys_incs)))
    parts.extend(bodies)

    out = os.path.join(ROOT, cfg.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("".join(parts))

    size = os.path.getsize(out)
    print(f"wrote {cfg.out}: {size // 1024} KB, "
          f"{len(ordered_headers)} headers + {len(cpps)} sources, "
          f"{len(sys_incs)} system includes", file=sys.stderr)
    if missing:
        print(f"  WARNING unresolved includes: {sorted(missing)}", file=sys.stderr)
    if clashes:
        print(f"  renamed {len(clashes)} colliding file-local symbol(s): "
              f"{sorted(clashes)[:8]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
