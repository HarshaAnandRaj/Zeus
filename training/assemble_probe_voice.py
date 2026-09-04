"""Assemble a probe-trained readout into an isolated Zeus checkpoint.

The live milestone is never overwritten.  This makes the clean-corpus probe
auditable before any deploy decision is considered.
"""

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.model import ZeusConfig, ZeusCore


def stamp_probe_step(base, probe_checkpoint):
    """Carry the trained mouth's step into its assembled evaluation artifact."""
    try:
        step = int(probe_checkpoint["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("probe checkpoint is missing a valid global_step") from exc
    if step < 0:
        raise ValueError("probe checkpoint global_step must be non-negative")
    # ``zsession.load_safe`` reports ``step`` while probe checkpoints use
    # ``global_step``.  Keep both representations coherent in the assembled
    # artifact so its filename, evaluator report, and weights name one run.
    base["global_step"] = step
    base["step"] = step
    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", default="zeus_sandbox/universe/shadow/milestone.pt")
    ap.add_argument("--run_dir", default="runs/probe_v2_primary")
    ap.add_argument("--out", default="runs/probe_v2_primary/milestone.pt")
    args = ap.parse_args()

    base_path, run_dir, out = pathlib.Path(args.base_checkpoint), pathlib.Path(args.run_dir), pathlib.Path(args.out)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    probe = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    step = stamp_probe_step(base, probe)
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    knobs = run_config["readout_config"]
    cfg_dict = dict(base["config"])
    cfg_dict.update(knobs)
    model = ZeusCore(ZeusConfig(**cfg_dict), tokenizer_path=base.get("tokenizer"))
    model.load_state_dict(base["model"], strict=False)
    readout = torch.load(run_dir / "readout.pt", map_location="cpu", weights_only=True)
    embedding = torch.load(run_dir / "emb.pt", map_location="cpu", weights_only=True)
    model.load_state_dict({"readout." + k: v for k, v in readout.items()}, strict=False)
    model.load_state_dict({"embed." + k: v for k, v in embedding.items()}, strict=False)
    base["config"] = cfg_dict
    base["model"] = model.state_dict()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    print(json.dumps({"kind": "probe_voice_assembled", "out": str(out), "run_dir": str(run_dir),
                      "global_step": step}))


if __name__ == "__main__":
    main()
