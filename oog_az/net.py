"""Residual policy + value network, and device selection.

Architecture is the standard AlphaZero shape -- conv stem, residual tower, two
heads -- deliberately, because the interesting variables here are the training
loop and the data, not the topology. It matches tools/nn_player.py's block
closely enough that the two are easy to compare.

DEVICE: detect, never assume. This host is an Apple M2 with no NVIDIA GPU, so
CUDA is absent and MPS is present; Colab is the reverse. A hard-coded device is
how a training script becomes unrunnable on the machine that has the GPU.

There is deliberately NO tensorboard import anywhere in this package. A missing
`tensorboard` is exactly what silently killed a MiniZero training run here: the
C++ self-play finished 2000 games, the Python learner died at
`from torch.utils.tensorboard import SummaryWriter` before any optimisation
step, and the only visible symptom was the message "Training failed to
complete" -- which was a grep check in a shell script, not an error from the
trainer. Metrics go to a JSONL file instead; that dependency cannot go missing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import N, NUM_ACTIONS, NUM_PLANES


def pick_device(requested: str | None = None) -> torch.device:
    """CUDA > MPS > CPU, unless the caller names one."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class AZNet(nn.Module):
    """Returns (policy_logits[B,225], value[B]) with value in [-1, 1].

    The policy head emits raw LOGITS, not probabilities. Callers mask illegal
    actions by adding -inf before the softmax; masking after a softmax would
    leave the illegal mass removed but the legal mass unnormalised.
    """

    def __init__(self, blocks: int = 6, channels: int = 96) -> None:
        super().__init__()
        self.blocks, self.channels = blocks, channels
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * N * N, NUM_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(),
            nn.Linear(N * N, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        h = self.body(self.stem(x))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def save_checkpoint(net: AZNet, path: str, **meta) -> None:
    """Store hyperparameters alongside weights so load needs no arguments."""
    torch.save(
        {
            "state_dict": net.state_dict(),
            "blocks": net.blocks,
            "channels": net.channels,
            **meta,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device | None = None):
    """Rebuild a net from a checkpoint. Returns (net, meta_dict).

    `weights_only=False` because the checkpoint carries plain metadata
    (iteration counts, config) alongside the tensors. Only load checkpoints you
    produced.
    """
    device = device or pick_device()
    ck = torch.load(path, map_location=device, weights_only=False)
    net = AZNet(ck.get("blocks", 6), ck.get("channels", 96)).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


__all__ = ["AZNet", "ResBlock", "pick_device", "save_checkpoint",
           "load_checkpoint"]
