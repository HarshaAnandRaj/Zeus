import argparse, json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.model import ZeusCore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="zeus_sandbox/universe/shadow/milestone.pt")
    ap.add_argument("--pretrain_dir", default="runs/broca_pretrain")
    ap.add_argument("--out", default="runs/broca_voice/milestone.pt")
    ap.add_argument("--tokenizer", default="corpus/data/tokenizer/bpe_8192.json")
    args = ap.parse_args()

    payload = torch.load(args.src, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    cfg["cross_attn"] = True
    cfg["readout_layers"] = json.loads(
        open(f"{args.pretrain_dir}/run_config.json", encoding="utf-8").read()
    )["readout_layers"]

    model = ZeusCore.build_via_dict(cfg) if hasattr(ZeusCore, "build_via_dict") else None
    if model is None:
        from core.model import ZeusConfig
        model = ZeusCore(ZeusConfig(**cfg))

    model.load_state_dict(payload["model"], strict=False)
    model.eval()

    ro = torch.load(f"{args.pretrain_dir}/readout.pt", map_location="cpu", weights_only=True)
    em = torch.load(f"{args.pretrain_dir}/emb.pt", map_location="cpu", weights_only=True)
    missing, _ = model.load_state_dict({"readout." + k: v for k, v in ro.items()},
                                       strict=False)
    model.load_state_dict({"embed." + k: v for k, v in em.items()}, strict=False)
    print("unmatched by readout.pt:", missing)

    payload["config"] = cfg
    payload["model"] = model.state_dict()
    payload["tokenizer"] = args.tokenizer
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()