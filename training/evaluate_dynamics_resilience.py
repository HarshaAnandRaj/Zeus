"""Evaluate the pre-registered DYN1 learned-resilience protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from collections import deque
from dataclasses import asdict
from typing import Any

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model import ZeusConfig, ZeusCore  # noqa: E402


CONTRACT_VERSION = "dyn1-2026-09-05"
EXPECTED_ARTIFACT_SHA256 = "bd3c6ca0b8fe92153345d19ea4b30471832528d934e81d05e91530c59c49feb3"
SEEDS = [101, 103, 107, 109, 113, 127, 131, 137,
         139, 149, 151, 157, 163, 167, 173, 179]
RESET_NOISE = 0.12
WARM_STEPS = 128
RECOVERY_STEPS = 256
FINAL_WINDOW = 64
KICK_RATIO = 0.50
MAD_SCALE = 1.4826
ENVELOPE_MADS = 3.0
SATURATION_LIMIT = 0.01
FEATURES = ("mean_state_norm", "median_step_displacement",
            "mean_coordinate_variance", "effective_rank_90")


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator("cpu")
    generator.manual_seed(seed)
    return generator


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("Wilson interval requires a positive total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (z / denominator) * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    )
    return [center - half_width, center + half_width]


def trajectory_features(window: torch.Tensor) -> dict[str, float | int]:
    if window.ndim != 2 or window.shape[0] < 2:
        raise ValueError("trajectory window must be [time, state] with at least two steps")
    work = window.detach().to(dtype=torch.float64, device="cpu")
    norms = torch.linalg.vector_norm(work, dim=1)
    displacements = torch.linalg.vector_norm(work[1:] - work[:-1], dim=1)
    centered = work - work.mean(dim=0, keepdim=True)
    coordinate_variance = centered.square().mean(dim=0)
    gram_eigenvalues = torch.linalg.eigvalsh(centered @ centered.T).clamp_min(0.0)
    energy = gram_eigenvalues.flip(0)
    total_energy = float(energy.sum().item())
    if total_energy == 0.0:
        effective_rank = 0
    else:
        effective_rank = int(
            torch.searchsorted(torch.cumsum(energy, dim=0), 0.9 * energy.sum()).item() + 1
        )
    return {
        "mean_state_norm": float(norms.mean().item()),
        "median_step_displacement": float(displacements.median().item()),
        "mean_coordinate_variance": float(coordinate_variance.mean().item()),
        "effective_rank_90": effective_rank,
        "clamp_saturation_fraction": float((work.abs() >= 7.99).double().mean().item()),
    }


def _recovery_roll(
    model: ZeusCore,
    *,
    recovery_steps: int,
    final_window: int,
    finite_so_far: bool,
) -> dict[str, Any]:
    finite = finite_so_far
    states: deque[torch.Tensor] = deque(maxlen=final_window)
    with torch.no_grad():
        for _ in range(recovery_steps):
            model.step(None, emit_readout=False)
            finite = finite and bool(torch.isfinite(model.S).all().item())
            states.append(model.S.detach().to(dtype=torch.float32, device="cpu").clone())
    if len(states) != final_window:
        raise ValueError("final window exceeds recovery trajectory")
    stacked = torch.stack(list(states), dim=0)
    features = trajectory_features(stacked)
    feature_finite = all(math.isfinite(float(value)) for value in features.values())
    return {
        "finite": finite and feature_finite,
        "features": features,
    }


def evaluate_pair(
    model: ZeusCore,
    seed: int,
    *,
    warm_steps: int = WARM_STEPS,
    recovery_steps: int = RECOVERY_STEPS,
    final_window: int = FINAL_WINDOW,
) -> dict[str, Any]:
    if recovery_steps < final_window:
        raise ValueError("recovery_steps must be at least final_window")
    model.reset_state(RESET_NOISE, seeded_generator(seed))
    model.hcm = None
    model.hcm_pending = None
    model.heartbeat_pending = None
    warm_finite = True
    with torch.no_grad():
        for _ in range(warm_steps):
            model.step(None, emit_readout=False)
            warm_finite = warm_finite and bool(torch.isfinite(model.S).all().item())

    pre_kick_state = model.S.detach().clone()
    runtime = model.snapshot_runtime()
    spectral_u = model.rec.u.detach().clone()
    spectral_v = model.rec.v.detach().clone()
    control_norm = float(torch.linalg.vector_norm(pre_kick_state).item())
    kick_norm = KICK_RATIO * control_norm

    control = _recovery_roll(
        model, recovery_steps=recovery_steps, final_window=final_window,
        finite_so_far=warm_finite,
    )
    model.restore_runtime(runtime)
    model.rec.u.copy_(spectral_u)
    model.rec.v.copy_(spectral_v)
    restored_pre_kick_state = model.S.detach().clone()
    same_pre_kick_state = bool(torch.equal(pre_kick_state, restored_pre_kick_state))
    direction = torch.randn(model.cfg.dim, generator=seeded_generator(seed + 1))
    direction = direction.to(device=model.S.device, dtype=model.S.dtype)
    direction = direction / direction.norm().clamp_min(1e-12)
    model.S = model.S + kick_norm * direction
    perturbed = _recovery_roll(
        model, recovery_steps=recovery_steps, final_window=final_window,
        finite_so_far=warm_finite and bool(torch.isfinite(model.S).all().item()),
    )
    return {
        "seed": seed,
        "control": {"finite": control["finite"], "features": control["features"]},
        "perturbed": {"finite": perturbed["finite"], "features": perturbed["features"]},
        "pre_kick_state_equal": same_pre_kick_state,
        "control_pre_kick_norm": control_norm,
        "kick_norm": kick_norm,
    }


def _median(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float64).median().item())


def leave_one_out_envelope(records: list[dict[str, Any]], index: int, feature: str) -> dict[str, float]:
    values = [float(row["control"]["features"][feature])
              for position, row in enumerate(records) if position != index]
    if not values:
        raise ValueError("leave-one-out envelope requires at least two records")
    median = _median(values)
    mad = _median([abs(value - median) for value in values])
    scaled_mad = MAD_SCALE * mad
    half_width = ENVELOPE_MADS * scaled_mad
    fallback_used = half_width == 0.0
    if fallback_used:
        half_width = max(0.05 * abs(median), 1e-6)
    return {
        "median": median,
        "mad": mad,
        "scaled_mad": scaled_mad,
        "lower": median - half_width,
        "upper": median + half_width,
        "fallback_used": fallback_used,
    }


def adjudicate_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    judged = []
    for index, row in enumerate(records):
        envelopes = {feature: leave_one_out_envelope(records, index, feature)
                     for feature in FEATURES}
        feature_checks = {
            feature: (envelopes[feature]["lower"]
                      <= float(row["perturbed"]["features"][feature])
                      <= envelopes[feature]["upper"])
            for feature in FEATURES
        }
        saturation = float(row["perturbed"]["features"]["clamp_saturation_fraction"])
        saturation_pass = saturation < SATURATION_LIMIT
        finite = bool(row["control"]["finite"] and row["perturbed"]["finite"]
                      and row["pre_kick_state_equal"])
        recovered = finite and all(feature_checks.values()) and saturation_pass
        result = dict(row)
        result.update({
            "envelopes": envelopes,
            "feature_checks": feature_checks,
            "saturation_pass": saturation_pass,
            "recovered": recovered,
        })
        judged.append(result)
    count = sum(bool(row["recovered"]) for row in judged)
    return {
        "records": judged,
        "recovery_count": count,
        "total": len(judged),
        "recovery_rate": count / len(judged),
        "wilson_95": wilson_interval(count, len(judged)),
        "all_trajectories_finite": all(
            row["control"]["finite"] and row["perturbed"]["finite"]
            for row in judged
        ),
        "all_pre_kick_states_equal": all(row["pre_kick_state_equal"] for row in judged),
    }


def evaluate(
    checkpoint: pathlib.Path,
    *,
    device: str,
    seeds: list[int] = SEEDS,
    expected_sha256: str = EXPECTED_ARTIFACT_SHA256,
) -> dict[str, Any]:
    artifact_sha256 = file_sha256(checkpoint)
    if artifact_sha256 != expected_sha256:
        raise ValueError(
            f"frozen artifact hash mismatch: expected {expected_sha256}, got {artifact_sha256}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_dict = payload["config"]
    tokenizer_path = payload.get("tokenizer")

    trained_model = ZeusCore.load(checkpoint, device=device)
    trained_records = [evaluate_pair(trained_model, seed) for seed in seeds]
    del trained_model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    random_records = []
    for seed in seeds:
        torch.manual_seed(seed)
        random_model = ZeusCore(
            ZeusConfig(**config_dict), tokenizer_path=tokenizer_path
        ).to(device).eval()
        random_records.append(evaluate_pair(random_model, seed))
        del random_model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    trained = adjudicate_arm(trained_records)
    random_init = adjudicate_arm(random_records)
    all_finite = (trained["all_trajectories_finite"]
                  and random_init["all_trajectories_finite"])
    trained_count_bar = trained["recovery_count"] >= 15
    superiority_bar = trained["wilson_95"][0] > random_init["wilson_95"][1]
    no_saturated_trained_pass = all(
        (not row["recovered"])
        or row["perturbed"]["features"]["clamp_saturation_fraction"] < SATURATION_LIMIT
        for row in trained["records"]
    )
    exact_design = len(seeds) == len(SEEDS) and seeds == SEEDS
    bars = {
        "exact_registered_design": exact_design,
        "all_trajectories_finite": all_finite,
        "trained_recovery_at_least_15_of_16": trained_count_bar,
        "trained_wilson_lower_strictly_above_random_upper": superiority_bar,
        "no_trained_recovery_pass_saturated": no_saturated_trained_pass,
    }
    return {
        "kind": "dyn1_learned_resilience",
        "contract_version": CONTRACT_VERSION,
        "artifact": str(checkpoint),
        "artifact_sha256": artifact_sha256,
        "config": config_dict,
        "conditions": {
            "seeds": seeds,
            "reset_noise": RESET_NOISE,
            "warm_steps": WARM_STEPS,
            "recovery_steps": RECOVERY_STEPS,
            "final_window": FINAL_WINDOW,
            "kick_ratio": KICK_RATIO,
            "hcm": False,
            "heartbeat": False,
            "token_or_embedding_input": False,
            "body_or_action_input": False,
            "parameter_updates": False,
        },
        "trained": trained,
        "random_init": random_init,
        "bars": bars,
        "pass": all(bars.values()),
        "interpretation_if_pass": (
            "candidate intrinsic-resilience mechanism only; causal component ablations remain required"
        ),
        "interpretation_if_fail": (
            "bounded motion is not distinguished from architectural containment"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=pathlib.Path,
        default=ROOT / "zeus_sandbox" / "universe" / "shadow" / "milestone.pt",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out", type=pathlib.Path,
        default=ROOT / "zeus_sandbox" / "universe" / "reports"
        / "dyn1_resilience_20260905.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = evaluate(args.checkpoint.resolve(), device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "trained": f'{report["trained"]["recovery_count"]}/{report["trained"]["total"]}',
        "random_init": f'{report["random_init"]["recovery_count"]}/{report["random_init"]["total"]}',
        "bars": report["bars"],
        "pass": report["pass"],
    }, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
