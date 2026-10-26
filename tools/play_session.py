#!/usr/bin/env python3
"""One human-vs-engine game, driven over the Gomocup protocol.

Backs the "play" mode in tools/watch_match_web.py. The protocol handling is
imported from tools/match.py rather than rewritten: that file already knows the
two things that are easy to get wrong and have each cost real games here --

  * BOARD fields are RELATIVE (1 = the receiving engine's own stone, 2 = the
    opponent's), not absolute colours. Sending absolute colours hands a
    White-playing engine a colour-swapped position, which it either rejects or
    plays badly.
  * A stray MESSAGE / ERROR line is not a move. `Brain.recv` skips chatter and
    tolerates an ERROR followed by a legal move, which is what Yixin does.

Outer-Open is enforced here the same way the match runner does it: Black's
first stone must lie in the outer two rows or columns.
"""

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match import Board, Brain, parse_reply, REPO  # noqa: E402

N = 15
# Cold-start allowances. Generous on purpose: a false forfeit is far worse
# than a slow first move, and only the first move pays these.
START_TIMEOUT_S = 120.0
FIRST_MOVE_GRACE_S = 60.0


class PlaySession:
    """A single game: a human on one colour, an engine on the other."""

    def __init__(self, engine_cmd, human_color="B", think_ms=3000, label=""):
        self.lock = threading.Lock()
        self.board = Board()
        self.human = human_color            # "B" or "W"
        self.engine_color = "W" if human_color == "B" else "B"
        self.think_ms = int(think_ms)
        self.label = label or engine_cmd
        self.moves = []                     # [(colour, r, c, ms)]
        self.over = None                    # None | "B" | "W" | "draw"
        self.reason = ""
        self.error = ""
        self.to_move = "B"

        import shlex
        self.brain = Brain(shlex.split(engine_cmd), REPO)
        self.brain.send("INFO rule 0")
        self.brain.send(f"INFO timeout_turn {self.think_ms}")
        self.brain.send("INFO time_left 2147483647")
        self.brain.send("START 15")
        # Wine engines spin up a Docker container, boot a Wine prefix and load
        # an ONNX model before they can answer anything -- tens of seconds,
        # cold. 10s here produced "engine forfeited: no reply" before the
        # engine had even started. match.py gets away with less because
        # wine-engine warms the prefix before the manager times anything, and
        # because a 20-game match amortises startup; a single fresh game pays
        # it all up front.
        self.started_ok = self.brain.recv(START_TIMEOUT_S) is not None
        self.first_move = True

    # -- helpers ----------------------------------------------------------
    def _send_board(self):
        """Full position, RELATIVE to the engine (1 = engine's own stones)."""
        self.brain.send("BOARD")
        for r in range(N):
            for c in range(N):
                g = self.board.g[r][c]
                if g == ".":
                    continue
                self.brain.send(f"{c},{r},{1 if g == self.engine_color else 2}")
        self.brain.send("DONE")

    def _finish(self, r, c, colour):
        if self.board.win_at(r, c):
            self.over = colour
            self.reason = f"{self.board.win_len}-in-a-row at ({r},{c})"
            return True
        if self.board.full():
            self.over = "draw"
            self.reason = "board full"
            return True
        return False

    def _engine_move(self):
        if self.over:
            return
        budget = self.think_ms / 1000.0
        # The first reply may still be paying model-load cost even after START
        # was acknowledged, so grant it a much larger grace once.
        grace = budget * 0.25 + 0.5 + (FIRST_MOVE_GRACE_S if self.first_move else 0.0)
        import time
        t0 = time.monotonic()
        if not self.board.moves:
            self.brain.send("BEGIN")
        else:
            self._send_board()
        line = self.brain.recv(budget + grace)
        self.first_move = False
        dt = int((time.monotonic() - t0) * 1000)
        move, err = parse_reply(line)
        if move is None:
            self.over = self.human
            hint = ""
            if not getattr(self, "started_ok", True):
                hint = (" -- the engine never acknowledged START, so it most "
                        "likely failed to launch (missing binary, Docker not "
                        "running, or a 32-bit .exe our 64-bit Wine cannot run)")
            elif self.brain.warnings:
                hint = f" -- engine emitted {self.brain.warnings} ERROR/UNKNOWN line(s)"
            self.reason = f"engine forfeited: {err}{hint}"
            return
        r, c = move
        if not self.board.play(r, c, self.engine_color):
            self.over = self.human
            self.reason = f"engine played illegal move ({r},{c})"
            return
        self.moves.append((self.engine_color, r, c, dt))
        if not self._finish(r, c, self.engine_color):
            self.to_move = self.human

    # -- public API -------------------------------------------------------
    def start(self):
        """If the engine has Black it moves first."""
        with self.lock:
            if self.engine_color == "B":
                self._engine_move()
            return self.state()

    def human_move(self, r, c):
        with self.lock:
            self.error = ""
            if self.over:
                self.error = "game is over"
                return self.state()
            if self.to_move != self.human:
                self.error = "not your turn"
                return self.state()
            if not (0 <= r < N and 0 <= c < N):
                self.error = "off board"
                return self.state()
            # Outer-Open: Black's FIRST stone must be in the outer two rows
            # or columns. Only the first stone -- everything after is plain
            # freestyle.
            if (not self.board.moves and self.human == "B"
                    and not self.board.is_ring(r, c)):
                self.error = ("Outer-Open: Black's first stone must be in the "
                              "outer two rows or columns")
                return self.state()
            if not self.board.play(r, c, self.human):
                self.error = "square occupied"
                return self.state()
            self.moves.append((self.human, r, c, 0))
            if not self._finish(r, c, self.human):
                self.to_move = self.engine_color
                self._engine_move()
            return self.state()

    def state(self):
        return {
            "moves": [{"side": s, "r": r, "c": c, "ms": ms}
                      for (s, r, c, ms) in self.moves],
            "human": self.human,
            "engine": self.engine_color,
            "engine_label": self.label,
            "to_move": self.to_move,
            "over": self.over,
            "reason": self.reason,
            "error": self.error,
            "think_ms": self.think_ms,
        }

    def close(self):
        try:
            self.brain.send("END")
            self.brain.proc.terminate()
        except Exception:
            pass
        # A Wine engine runs inside a Docker container. `docker run --rm` only
        # cleans up when the client exits cleanly, so terminating the wrapper
        # can leave the container running -- each one holding a CPU share and a
        # Wine prefix. Six had accumulated before this was noticed.
        try:
            import subprocess
            subprocess.run([os.path.join(REPO, "tools", "reap_wine.sh")],
                           timeout=20, capture_output=True)
        except Exception:
            pass


# Engines offered in the UI. Only ones verified to start and answer here.
ENGINES = {
    "VIBEFIVE":    "third_party/wine-engine vibefive pbrain-vibefive.exe",
    "Barbakan":    "third_party/wine-engine barbakan pbrain-barbakan15x15_avx2.exe",
    "Yixin 2018":  "third_party/wine-engine yixin pbrain-Yixin2018.exe",
    "SKYZERO":     "third_party/wine-engine skyzero pbrain-SkyZero.exe",
    "SlowRenju":   "third_party/SlowRenju/pbrain-slowrenju",
    "PentaZen":    "third_party/PentaZen/build/src/PentaZen",
    "rapfi (1t)":  "third_party/rapfi-bin/pbrain-rapfi",
    "rapfi (8t)":  "third_party/rapfi-8t/pbrain-rapfi",
    "AlphaGomoku": "third_party/ag584/ag584-run",
    # JAX25 runs fine locally (wine-10.0 in our container, ~9s of a 10s
    # budget). Note the freestyle path resolves to the 128-channel net, NOT
    # the 320-net configuration that took Gomocup silver -- a different and
    # weaker engine, so do not read its results as "JAX 2025 as rated".
    "JAX25 (128)": "third_party/wine-engine jax25 pbrain-Jax.exe",
    "our engine":  "build/release/bin/oog_piskvork",
}
