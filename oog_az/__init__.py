"""AlphaZero for Outer-Open Gomoku.

Self-play reinforcement learning: the network learns from games it plays
against itself, and the external engines in third_party/ serve as a yardstick
(oog_az/arena.py), never as the learning signal. See selfplay.py for why that
distinction is the whole design, and README.md for how to run it.
"""

__all__ = ["env", "net", "mcts", "selfplay", "train", "arena", "player"]
