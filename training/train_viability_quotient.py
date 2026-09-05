"""Train QV0 to predict held-out embodied transitions; no policy is trained."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from core.embodiment import EmbodiedWorldV2
from core.viability_quotient import ViabilityQuotient, homeostatic_error_tensor


TRAINING_VERSION = "qv0-predictive-quotient-2026-09-05"
DEFAULT_SEED = 20260931
TRAIN_WORLD_SEED_BASE = 202640000
TRAIN_ACTION_SEED_BASE = 202641000
DEFAULT_TRAJECTORIES = 384
DEFAULT_HORIZON = 96
DEFAULT_EPOCHS = 120
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_ERROR_WEIGHT = 0.10
DEFAULT_QUOTIENT_DIM = 12
DEFAULT_HIDDEN_DIM = 48


def configure_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def tensor_state_sha256(state_dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_trajectories(*, count: int, horizon: int, world_seed_base: int,
                         action_seed_base: int):
    """Collect fixed uniform-exploration sequences from the physical world."""
    if count < 1 or horizon < 1:
        raise ValueError("count and horizon must be positive")
    trajectories = []
    for index in range(count):
        world = EmbodiedWorldV2(seed=world_seed_base + index)
        action_rng = random.Random(action_seed_base + index)
        observations = [world.observation()]
        actions = []
        for _ in range(horizon):
            action = action_rng.randrange(6)
            effect = world.step(action)
            actions.append(action)
            observations.append(effect["after"])
            if not effect["viable"]:
                break
        trajectories.append({
            "observations": torch.tensor(observations, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
        })
    return trajectories


def trajectory_sha256(trajectories) -> str:
    digest = hashlib.sha256()
    for trajectory in trajectories:
        for key in ("observations", "actions"):
            tensor = trajectory[key].detach().cpu().contiguous()
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def target_observation_mean(trajectories):
    return torch.cat([
        trajectory["observations"][1:] for trajectory in trajectories
    ]).mean(dim=0)


def _batch_loss(model, trajectories, indices, *, device, error_weight):
    selected = [trajectories[int(index)] for index in indices]
    batch_size = len(selected)
    max_steps = max(len(item["actions"]) for item in selected)
    observations = torch.zeros(batch_size, max_steps + 1, 5, device=device)
    actions = torch.zeros(batch_size, max_steps, dtype=torch.long, device=device)
    lengths = torch.tensor(
        [len(item["actions"]) for item in selected], device=device
    )
    for row, item in enumerate(selected):
        steps = len(item["actions"])
        observations[row, :steps + 1] = item["observations"].to(device)
        actions[row, :steps] = item["actions"].to(device)

    state = model.initial_state(batch_size, device=device)
    squared_error_sum = torch.zeros((), device=device)
    homeostatic_error_sum = torch.zeros((), device=device)
    counted = torch.zeros((), device=device)
    previous_observation = observations[:, 0]
    for tick in range(max_steps):
        active = tick < lengths
        previous_action = None if tick == 0 else actions[:, tick - 1]
        proposed = model.update(
            observations[:, tick], previous_observation, previous_action, state
        )
        state = torch.where(active.unsqueeze(-1), proposed, state)
        prediction = model.predict(state, actions[:, tick])
        target = observations[:, tick + 1]
        mask = active.to(prediction.dtype)
        squared_error_sum = squared_error_sum + (
            (prediction - target).pow(2).mean(dim=-1) * mask
        ).sum()
        homeostatic_error_sum = homeostatic_error_sum + (
            homeostatic_error_tensor(prediction)
            - homeostatic_error_tensor(target)
        ).pow(2).mul(mask).sum()
        counted = counted + mask.sum()
        previous_observation = observations[:, tick]
    prediction_loss = squared_error_sum / counted.clamp_min(1.0)
    error_loss = homeostatic_error_sum / counted.clamp_min(1.0)
    return prediction_loss + error_weight * error_loss, prediction_loss, error_loss


def train_quotient(model, trajectories, *, epochs=DEFAULT_EPOCHS,
                   batch_size=DEFAULT_BATCH_SIZE, lr=DEFAULT_LR,
                   error_weight=DEFAULT_ERROR_WEIGHT, seed=DEFAULT_SEED,
                   device="cpu"):
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    rows = []
    for epoch in range(epochs):
        permutation = torch.randperm(len(trajectories), generator=generator)
        totals = torch.zeros(3, dtype=torch.float64)
        batches = 0
        for start in range(0, len(trajectories), batch_size):
            indices = permutation[start:start + batch_size]
            loss, prediction_loss, error_loss = _batch_loss(
                model, trajectories, indices, device=device,
                error_weight=error_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals += torch.tensor([
                float(loss.detach().cpu()),
                float(prediction_loss.detach().cpu()),
                float(error_loss.detach().cpu()),
            ], dtype=torch.float64)
            batches += 1
        rows.append({
            "epoch": epoch + 1,
            "loss": round(float(totals[0] / batches), 9),
            "prediction_mse": round(float(totals[1] / batches), 9),
            "homeostatic_error_mse": round(float(totals[2] / batches), 9),
        })
    model.eval()
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_determinism(DEFAULT_SEED)
    trajectories = collect_trajectories(
        count=DEFAULT_TRAJECTORIES, horizon=DEFAULT_HORIZON,
        world_seed_base=TRAIN_WORLD_SEED_BASE,
        action_seed_base=TRAIN_ACTION_SEED_BASE,
    )
    model = ViabilityQuotient(
        quotient_dim=DEFAULT_QUOTIENT_DIM, hidden_dim=DEFAULT_HIDDEN_DIM
    )
    initial_sha256 = tensor_state_sha256(model.state_dict())
    rows = train_quotient(model, trajectories, device=args.device)
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    payload = {
        "kind": "qv0_viability_quotient",
        "training_version": TRAINING_VERSION,
        "world_version": EmbodiedWorldV2.VERSION,
        "seed": DEFAULT_SEED,
        "config": {
            "quotient_dim": DEFAULT_QUOTIENT_DIM,
            "hidden_dim": DEFAULT_HIDDEN_DIM,
            "trajectories": DEFAULT_TRAJECTORIES,
            "horizon": DEFAULT_HORIZON,
            "epochs": DEFAULT_EPOCHS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "lr": DEFAULT_LR,
            "error_weight": DEFAULT_ERROR_WEIGHT,
            "world_seed_base": TRAIN_WORLD_SEED_BASE,
            "action_seed_base": TRAIN_ACTION_SEED_BASE,
            "exploration_policy": "uniform_random",
            "policy_training": False,
        },
        "training_data_sha256": trajectory_sha256(trajectories),
        "target_observation_mean": target_observation_mean(trajectories),
        "initial_state_sha256": initial_sha256,
        "state_sha256": tensor_state_sha256(state),
        "state_dict": state,
        "training": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "training_data_sha256": payload["training_data_sha256"],
        "initial_state_sha256": initial_sha256,
        "state_sha256": payload["state_sha256"],
        "final": rows[-1],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
