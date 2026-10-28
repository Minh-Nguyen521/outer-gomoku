#!/usr/bin/env python3
"""Export a MiniZero model (.pt TorchScript or .pkl snapshot) to other formats.

Outputs (choose with --format, repeatable):
  onnx  <stem>.onnx          -- for onnxruntime (Botzone, C++, JS)
  npz   <stem>_weights.npz   -- raw float32 arrays, framework-free inference
  pth   <stem>.pth           -- plain state_dict for PyTorch

How it works:
  * .pkl  -- training snapshot; the state_dict is read directly.
  * .pt   -- TorchScript; state_dict is extracted from the scripted module.
  For ONNX the network is REBUILT in eager Python (via
  minizero.network.py.create_network) because torch.onnx cannot trace an
  already-scripted TorchScript module. Architecture parameters come from the
  run's .cfg file (--cfg), falling back to key-shape sniffing.

Usage:
    python3 tools/export_model.py <model.pt|model.pkl> [--cfg run.cfg]
                                  [--out-dir DIR]
                                  [--format onnx] [--format npz] [--format pth]
"""

import argparse
import os
import sys

import numpy as np
import torch

MZ_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "minizero")

# gomoku env constants (environment/gomoku/gomoku.h): 4 input channels,
# policy = board_size^2, discrete value size = 1 (base env default).
GOMOKU_INPUT_CHANNELS = 4
GOMOKU_DISCRETE_VALUE_SIZE = 1


def load_state_dict(path):
    if path.endswith(".pkl"):
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        return snapshot["network"], "pkl-snapshot"
    module = torch.jit.load(path, map_location="cpu")
    return module.state_dict(), "torchscript"


def parse_cfg(path):
    cfg = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    return cfg


def sniff_arch(sd, board_size):
    """Guess num_blocks/channels from state_dict key shapes (fallback path)."""
    blocks = {int(k.split(".")[2]) for k in sd
              if k.startswith("residual_blocks.")}
    num_blocks = (max(blocks) + 1) if blocks else 1
    ch = sd["policy.conv.weight"].shape[0]
    val_ch = sd["value.fc0.weight"].shape[0] \
        if "value.fc0.weight" in sd else 256
    return num_blocks, ch, val_ch


def build_network(sd, cfg, board_size):
    sys.path.insert(0, MZ_DEFAULT)
    from minizero.network.py.create_network import create_network

    def cfg_get(key, cast, default):
        try:
            return cast(cfg[key])
        except (KeyError, ValueError):
            return default

    board = board_size or cfg_get("env_board_size", int, 15)
    action_size = board * board
    nn_type = cfg.get("nn_type_name", "alphazero")
    # MuZero key layout is a superset of AlphaZero's; distinguish by keys.
    if any(k.startswith("dynamics_network") for k in sd):
        nn_type = "muzero"
    elif any(k.startswith("representation_network") for k in sd):
        nn_type = "muzero"  # MuZero-style key layout
    elif any(k.startswith("residual_blocks") for k in sd):
        nn_type = "alphazero"
    else:
        raise SystemExit("unrecognized state_dict layout")

    num_blocks = cfg_get("nn_num_blocks", int, None)
    num_hidden = cfg_get("nn_num_hidden_channels", int, None)
    num_val_hidden = cfg_get("nn_num_value_hidden_channels", int, None)
    if num_blocks is None or num_hidden is None or num_val_hidden is None:
        nb, ch, vch = sniff_arch(sd, board)
        num_blocks = num_blocks or nb
        num_hidden = num_hidden or ch
        num_val_hidden = num_val_hidden or vch
    value_size = cfg_get("nn_discrete_value_size", int,
                         GOMOKU_DISCRETE_VALUE_SIZE)
    if "atari" in cfg.get("env_game_name", ""):
        raise SystemExit("atari models are not supported by this script")

    net = create_network(
        game_name="gomoku_oo_%dx%d" % (board, board),
        num_input_channels=GOMOKU_INPUT_CHANNELS,
        input_channel_height=board,
        input_channel_width=board,
        num_hidden_channels=num_hidden,
        hidden_channel_height=board,
        hidden_channel_width=board,
        num_action_feature_channels=cfg_get(
            "nn_num_action_feature_channels", int, 1),
        num_blocks=num_blocks,
        action_size=action_size,
        num_value_hidden_channels=num_val_hidden,
        discrete_value_size=value_size,
        network_type_name=nn_type)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(
            "state_dict does not match rebuilt network:\n"
            f"  missing: {missing}\n  unexpected: {unexpected}\n"
            "Check --cfg (wrong nn_* values?) or the minizero checkout.")
    net.eval()
    return net, nn_type, board


def forward_initial(net, nn_type, board):
    state = torch.zeros(1, GOMOKU_INPUT_CHANNELS, board, board)
    out = net(state)
    return state, out


def export_onnx(net, nn_type, board, out_path):
    state = torch.randn(1, GOMOKU_INPUT_CHANNELS, board, board)

    if nn_type == "muzero":

        class FlatMu(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, state, action_plane):
                out = self.m(state, action_plane)
                return (out["policy_logit"], out["value"], out["hidden_state"])

        module = FlatMu(net)
        args = (state, torch.zeros(1, 1, board, board))
        names = ["state", "action_plane"]
        outs = ["policy_logit", "value", "hidden_state"]
        dyn = {"state": {0: "batch"}, "action_plane": {0: "batch"},
               "policy_logit": {0: "batch"}, "value": {0: "batch"},
               "hidden_state": {0: "batch"}}
    else:

        class FlatAZ(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, state):
                out = self.m(state)
                return (out["policy_logit"], out["value"])

        module = FlatAZ(net)
        args = (state,)
        names = ["state"]
        outs = ["policy_logit", "value"]
        dyn = {"state": {0: "batch"}, "policy_logit": {0: "batch"},
               "value": {0: "batch"}}

    torch.onnx.export(module, args, out_path,
                      input_names=names, output_names=outs,
                      opset_version=17, dynamic_axes=dyn, dynamo=False)
    # the legacy exporter leaves the wrapped network in training mode, which
    # would poison the numeric check below (BatchNorm batch statistics)
    module.train(False)
    print(f"  wrote {out_path}")
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cfg", default=None,
                    help="the run's .cfg (architecture source; guessed if omitted)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--format", action="append",
                    choices=["onnx", "npz", "pth"], default=None)
    args = ap.parse_args()

    formats = args.format or ["onnx", "npz"]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.model))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.model))[0]

    sd, kind = load_state_dict(args.model)
    print(f"loaded {args.model} ({kind}, {len(sd)} tensors)")

    cfg = {}
    if args.cfg:
        cfg = parse_cfg(args.cfg)
        print(f"using config {args.cfg}")
    if formats == ["pth"]:
        torch.save(sd, os.path.join(out_dir, f"{stem}.pth"))
        print(f"  wrote {os.path.join(out_dir, stem + '.pth')}")
        return

    net, nn_type, board = build_network(sd, cfg, board_size=None)
    print(f"rebuilt {nn_type} network, board {board}x{board}")

    if "pth" in formats:
        pth = os.path.join(out_dir, f"{stem}.pth")
        torch.save(net.state_dict(), pth)
        print(f"  wrote {pth}")
    if "npz" in formats:
        arrays = {k: v.numpy() for k, v in net.state_dict().items()}
        npz = os.path.join(out_dir, f"{stem}_weights.npz")
        np.savez(npz, **arrays)
        print(f"  wrote {npz}")
    if "onnx" in formats:
        onnx_path = os.path.join(out_dir, f"{stem}.onnx")
        state = export_onnx(net, nn_type, board, onnx_path)

        try:
            import onnx
            onnx.checker.check_model(onnx.load(onnx_path))
            print("  onnx.checker: OK")
        except ImportError:
            print("  (pip install onnx to enable the checker)")

        # numeric check against the eager network
        with torch.no_grad():
            ref = net(state)
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path,
                                        providers=["CPUExecutionProvider"])
            got = sess.run(None, {"state": state.numpy()})
            err = max(np.abs(got[0] - ref["policy_logit"].numpy()).max(),
                      np.abs(got[1] - ref["value"].numpy()).max())
            verdict = "OK" if err < 1e-4 else "MISMATCH"
            print(f"  onnxruntime max |diff| vs eager: {err:.2e} ({verdict})")
        except ImportError:
            print("  (pip install onnxruntime to enable the numeric check)")


if __name__ == "__main__":
    main()
