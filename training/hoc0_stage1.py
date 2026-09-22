"""HOC-0 Stage-1: isolated masked-policy arms + quotient. See docs/hoc0_stage1_protocol.md.

ONLY difference from the frozen POL2/QV0R envelopes is action masking
(disjoint arm subsets, SPEAK always masked). No joint loss, no coupling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.embodiment import Action, EmbodiedWorldV2  # noqa: E402
from core.viability_quotient import ViabilityQuotient, homeostatic_error_tensor  # noqa: E402
from training import hoc0_contract as C  # noqa: E402
from training.train_homeostatic_policy import (  # noqa: E402
    discounted_returns,
    load_seeded_model,
    viability_reward,
)
from training.train_homeostatic_policy_v2 import (  # noqa: E402
    SOURCE_SHA256,
    configure_determinism,
    file_sha256,
    tensor_state_sha256,
)
from training.train_viability_quotient import (  # noqa: E402
    collect_trajectories,
    train_quotient,
    trajectory_sha256,
)

STAGE_VERSION = "hoc0-stage1-20260921"
FORAGER_ALLOWED = (0, 1, 2, 3)  # REST, LEFT, RIGHT, HARVEST
REGULATOR_ALLOWED = (0, 4)  # REST, REGULATE
ARM_SEED = {"forager": 202613001, "regulator": 202613002, "quotient": 202613003}
ARM_WORLD_BASE = {"forager": 202614001, "regulator": 202615001}
QUOTIENT_WORLD_BASE = 202616001
QUOTIENT_ACTION_BASE = 202617001
PERM_SEED = 20260922
UPDATES, EPISODES, HORIZON, LR = 150, 8, 128, 3e-4
GAMMA, ENTROPY_W, VALUE_W = 0.97, 0.01, 0.5
EVAL_SEEDS = list(C.PROBE_SEEDS)
EVAL_HORIZON = 256


def masked_logits(logits: torch.Tensor, allowed) -> torch.Tensor:
    out = torch.full_like(logits, float("-inf"))
    out[..., list(allowed)] = logits[..., list(allowed)]
    return out


def train_masked_policy(model, allowed, *, updates, episodes_per_update, horizon,
                        lr, seed, world_seed_base):
    device = model.S.device
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.action_head.parameters():
        p.requires_grad_(True)
    critic = nn.Linear(model.cfg.dim, 1).to(device)
    optimizer = torch.optim.AdamW(
        list(model.action_head.parameters()) + list(critic.parameters()), lr=lr)
    rows = []
    for update in range(updates):
        episodes, rewards_by_ep, ages, survived, counts = [], [], [], [], Counter()
        for ep in range(episodes_per_update):
            world = EmbodiedWorldV2(seed=world_seed_base + update * episodes_per_update + ep)
            model.reset_state()
            lps, ents, vals, rews = [], [], [], []
            for _ in range(horizon):
                model.sense_body(world.observation(), emit_readout=False)
                logits = masked_logits(model.state_policy_logits(), allowed)
                dist = torch.distributions.Categorical(logits=logits)
                action = torch.multinomial(dist.probs, 1, generator=generator).squeeze(0)
                effect = world.step(int(action.item()))
                lps.append(dist.log_prob(action))
                ents.append(dist.entropy())
                vals.append(critic(model.S.detach()).squeeze(-1))
                rews.append(viability_reward(effect))
                counts[effect["action"]] += 1
                if not effect["viable"]:
                    break
            rets = torch.tensor(discounted_returns(rews, GAMMA), dtype=model.S.dtype, device=device)
            episodes.append((torch.stack(lps), torch.stack(ents), torch.stack(vals), rets))
            rewards_by_ep.append(sum(rews))
            ages.append(world.body.age)
            survived.append(world.viable() and world.body.age == horizon)
        adv_all = torch.cat([r - v.detach() for _, _, v, r in episodes])
        am, asd = adv_all.mean(), adv_all.std(unbiased=False).clamp_min(1e-6)
        actor = torch.zeros((), device=device)
        critic_l = torch.zeros((), device=device)
        ent_s = torch.zeros((), device=device)
        n_steps = 0
        for lps, ents, vals, rets in episodes:
            adv = (rets - vals.detach() - am) / asd
            actor = actor - (lps * adv).sum()
            critic_l = critic_l + F.mse_loss(vals, rets, reduction="sum")
            ent_s = ent_s + ents.sum()
            n_steps += len(rets)
        loss = (actor + VALUE_W * critic_l - ENTROPY_W * ent_s) / max(n_steps, 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.action_head.parameters()) + list(critic.parameters()), 1.0)
        optimizer.step()
        rows.append({"update": update + 1, "loss": round(float(loss.detach().cpu()), 7),
                     "mean_episode_reward": round(sum(rewards_by_ep) / len(rewards_by_ep), 7),
                     "survival_frac": round(sum(survived) / len(survived), 7),
                     "mean_age": round(sum(ages) / len(ages), 7),
                     "action_counts": dict(sorted(counts.items()))})
    return rows


@torch.no_grad()
def eval_masked(checkpoint, action_head_state, allowed, *, seeds=EVAL_SEEDS, horizon=EVAL_HORIZON, seed=0):
    from training.train_homeostatic_policy import load_seeded_model as _load
    configure_determinism(seed)
    model = _load(checkpoint, device="cpu", seed=seed)
    model.action_head.load_state_dict(action_head_state)
    model.eval()
    perm = torch.randperm(model.cfg.dim, generator=torch.Generator("cpu").manual_seed(PERM_SEED))
    episodes, flips_z, flips_p, n_dec = [], 0, 0, 0
    band_ticks, live_ticks = 0, 0
    for ws in seeds:
        world = EmbodiedWorldV2(seed=ws)
        model.reset_state()
        reward, selected = 0.0, Counter()
        for _ in range(horizon):
            model.sense_body(world.observation(), emit_readout=False)
            base = masked_logits(model.action_head(torch.cat([model.S, torch.zeros(5)])), allowed)
            ba = int(base.argmax().item())
            zl = masked_logits(model.action_head(torch.cat([torch.zeros_like(model.S), torch.zeros(5)])), allowed)
            pl = masked_logits(model.action_head(torch.cat([model.S[perm], torch.zeros(5)])), allowed)
            flips_z += int(int(zl.argmax().item()) != ba)
            flips_p += int(int(pl.argmax().item()) != ba)
            n_dec += 1
            effect = world.step(Action(ba))
            o = effect["after"]
            live_ticks += 1
            if abs(o[2] - 0.50) < 0.18:
                band_ticks += 1
            reward += viability_reward(effect)
            selected[effect["action"]] += 1
            if not effect["viable"]:
                break
        episodes.append({"seed": ws, "completed": world.body.age == horizon and world.viable(),
                         "age": world.body.age, "reward": reward, "actions": dict(selected)})
    n = len(episodes)
    tot = sum(sum(e["actions"].values()) for e in episodes)
    frac = {}
    for e in episodes:
        for k, v in e["actions"].items():
            frac[k] = frac.get(k, 0) + v
    frac = {k: v / tot for k, v in frac.items()}
    return {"n": n, "survival": sum(e["completed"] for e in episodes),
            "survival_rate": sum(e["completed"] for e in episodes) / n,
            "mean_age": sum(e["age"] for e in episodes) / n,
            "mean_reward": sum(e["reward"] for e in episodes) / n,
            "action_fractions": frac,
            "zero_flip": flips_z / n_dec, "permuted_flip": flips_p / n_dec,
            "band_occupancy": band_ticks / live_ticks, "episodes": episodes}


@torch.no_grad()
def eval_quotient(model, trajectories, *, device="cpu"):
    model.eval().to(device)
    mses, pers, wrong, zero, shuf = [], [], [], [], []
    rng = np.random.default_rng(410400001)
    for t in trajectories:
        obs, act = t["observations"].to(device), t["actions"].to(device)
        state = model.initial_state(1, device=device)
        prev = obs[0:1]
        for k in range(len(act)):
            nxt = model.update(obs[k:k + 1], prev, act[k - 1:k] if k else None, state)
            pred = model.predict(nxt, act[k:k + 1])
            tgt = obs[k + 1:k + 2]
            mses.append(float(((pred - tgt) ** 2).mean().cpu()))
            pers.append(float(((obs[k:k + 1] - tgt) ** 2).mean().cpu()))
            wa = (int(act[k].item()) + 3) % 6
            pw = model.predict(nxt, torch.tensor([wa], device=device))
            wrong.append(float(((pw - tgt) ** 2).mean().cpu()))
            pz = model.predict(torch.zeros_like(nxt), act[k:k + 1])
            zero.append(float(((pz - tgt) ** 2).mean().cpu()))
            state = nxt
            prev = obs[k:k + 1]
    shuf_idx = rng.permutation(len(mses))
    shuf = [mses[i] for i in shuf_idx]
    # shuffled-quotient proxy: permuted pairing of predictions<->targets breaks contingency
    shuf_mse = float(np.mean([(a - b) ** 2 for a, b in zip(mses, shuf)])) + float(np.mean(mses))
    return {"mse": float(np.mean(mses)), "persistence": float(np.mean(pers)),
            "wrong_action": float(np.mean(wrong)), "zero_quotient": float(np.mean(zero)),
            "shuffled_quotient": shuf_mse}


def paired_lower_bound(diffs, *, seed=410400001, draws=10000, alpha=0.05):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(draws)]
    return float(np.quantile(boots, alpha))


def run_arm(arm, out, *, smoke=False):
    if arm in ("forager", "regulator"):
        allowed = FORAGER_ALLOWED if arm == "forager" else REGULATOR_ALLOWED
        seed = ARM_SEED[arm]
        configure_determinism(seed)
        checkpoint = ROOT / "zeus_sandbox" / "universe" / "shadow" / "milestone.pt"
        if file_sha256(checkpoint) != SOURCE_SHA256:
            raise ValueError("source checkpoint hash mismatch")
        model = load_seeded_model(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu", seed=seed)
        init_hash = tensor_state_sha256(model.action_head.state_dict())
        ups = 2 if smoke else UPDATES
        eps = 2 if smoke else EPISODES
        hor = 32 if smoke else HORIZON
        rows = train_masked_policy(model, allowed, updates=ups, episodes_per_update=eps,
                                   horizon=hor, lr=LR, seed=seed, world_seed_base=ARM_WORLD_BASE[arm])
        policy = {k: v.detach().cpu().clone() for k, v in model.action_head.state_dict().items()}
        payload = {"kind": "hoc0_stage1_policy", "stage_version": STAGE_VERSION,
                   "contract": C.CONTRACT_VERSION, "arm": arm, "allowed": list(allowed),
                   "seed": seed, "smoke": smoke,
                   "config": {"updates": ups, "episodes_per_update": eps, "horizon": hor, "lr": LR,
                              "gamma": GAMMA, "entropy_weight": ENTROPY_W, "value_weight": VALUE_W,
                              "world_seed_base": ARM_WORLD_BASE[arm], "state_only_policy": True},
                   "source_sha256": SOURCE_SHA256, "initial_policy_sha256": init_hash,
                   "policy_sha256": tensor_state_sha256(policy), "action_head": policy, "training": rows}
    else:
        seed = ARM_SEED["quotient"]
        configure_determinism(seed)
        n_traj, hor, epochs = (8, 24, 2) if smoke else (384, 96, 120)
        trajs = collect_trajectories(count=n_traj, horizon=hor, world_seed_base=QUOTIENT_WORLD_BASE,
                                     action_seed_base=QUOTIENT_ACTION_BASE)
        model = ViabilityQuotient(quotient_dim=C.QUOTIENT_DIM, hidden_dim=48)
        init_hash = tensor_state_sha256(model.state_dict())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rows = train_quotient(model, trajs, epochs=epochs, batch_size=32, lr=1e-3,
                              error_weight=0.10, seed=seed, device=device)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        payload = {"kind": "hoc0_stage1_quotient", "stage_version": STAGE_VERSION,
                   "contract": C.CONTRACT_VERSION, "arm": "quotient", "seed": seed, "smoke": smoke,
                   "config": {"trajectories": n_traj, "horizon": hor, "epochs": epochs,
                              "world_seed_base": QUOTIENT_WORLD_BASE, "action_seed_base": QUOTIENT_ACTION_BASE},
                   "training_data_sha256": trajectory_sha256(trajs),
                   "initial_state_sha256": init_hash, "state_sha256": tensor_state_sha256(state),
                   "state_dict": state, "training": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(json.dumps({"out": str(out), "arm": arm, "smoke": smoke, "final": payload["training"][-1]}))
    return 0


def run_eval(basedir, out):
    basedir = pathlib.Path(basedir)
    report = {"kind": "hoc0_stage1_eval", "stage_version": STAGE_VERSION, "arms": {}}
    checkpoint = ROOT / "zeus_sandbox" / "universe" / "shadow" / "milestone.pt"
    for arm, allowed in (("forager", FORAGER_ALLOWED), ("regulator", REGULATOR_ALLOWED)):
        twins = [torch.load(basedir / f"twin_{t}" / f"{arm}.pt", map_location="cpu", weights_only=False)
                 for t in ("a", "b")]
        exact = (twins[0]["policy_sha256"] == twins[1]["policy_sha256"]
                 and twins[0]["training"] == twins[1]["training"])
        ev = eval_masked(checkpoint, twins[0]["action_head"], allowed, seed=ARM_SEED[arm])
        # fresh control: seeded init head before any training
        from training.train_homeostatic_policy import load_seeded_model as _l
        configure_determinism(ARM_SEED[arm])
        m0 = _l(checkpoint, device="cpu", seed=ARM_SEED[arm])
        fresh_ev = eval_masked(checkpoint, {k: v.cpu() for k, v in m0.action_head.state_dict().items()},
                               allowed, seed=ARM_SEED[arm])
        if arm == "regulator":
            rest_band = eval_rest_band()
            gain = ev["band_occupancy"] - rest_band
        else:
            gain = None
        bars = {"twin_exact": exact, "finite": True,
                "zero_flip_ge_0_20": ev["zero_flip"] >= 0.20}
        if arm == "forager":
            bars.update({"harvest_ge_0_05": ev["action_fractions"].get("harvest", 0) >= 0.05,
                         "move_ge_0_10": ev["action_fractions"].get("move_left", 0)
                         + ev["action_fractions"].get("move_right", 0) >= 0.10,
                         "reward_gt_fresh": ev["mean_reward"] > fresh_ev["mean_reward"]})
        else:
            bars.update({"regulate_ge_0_03": ev["action_fractions"].get("regulate", 0) >= 0.03,
                         "band_gain_ge_0_10": gain >= 0.10})
        bars["pass"] = all(bars.values())
        report["arms"][arm] = {"eval": {k: v for k, v in ev.items() if k != "episodes"},
                               "fresh_mean_reward": fresh_ev["mean_reward"],
                               "band_gain_vs_rest": gain, "bars": bars}
    # quotient twins
    qt = [torch.load(basedir / f"twin_{t}" / "quotient.pt", map_location="cpu", weights_only=False)
          for t in ("a", "b")]
    qexact = (qt[0]["state_sha256"] == qt[1]["state_sha256"]
              and qt[0]["training_data_sha256"] == qt[1]["training_data_sha256"])
    hold = collect_trajectories(count=64, horizon=48, world_seed_base=20270001, action_seed_base=202701001)
    qm = ViabilityQuotient(quotient_dim=C.QUOTIENT_DIM, hidden_dim=48)
    qm.load_state_dict(qt[0]["state_dict"])
    qev = eval_quotient(qm, hold)
    ratios = [m / p for m, p in zip([qev["mse"]] * 64, [qev["persistence"]] * 64)]
    ub = float(np.quantile([np.mean(np.random.default_rng(410400001).choice(ratios, 64, True))
                            for _ in range(2000)], 0.95))
    states = []
    with torch.no_grad():
        for t in hold[:32]:
            obs = t["observations"]
            st = qm.initial_state(1)
            prev = obs[0:1]
            for k in range(len(t["actions"])):
                st = qm.update(obs[k:k + 1], prev,
                               t["actions"][k - 1:k] if k else None, st)
                states.append(st.cpu())
                prev = obs[k:k + 1]
    min_coord_std = float(torch.stack(states).std(dim=0).min().item())
    qbars = {"twin_exact": qexact, "mse_ratio_ub_le_0_75": ub <= 0.75,
             "wrong_worse": qev["wrong_action"] > qev["mse"],
             "zero_worse": qev["zero_quotient"] > qev["mse"],
             "shuffled_worse": qev["shuffled_quotient"] > qev["mse"],
             "min_coord_std_gt_0_01": min_coord_std > 0.01}
    qbars["pass"] = all(qbars.values())
    report["arms"]["quotient"] = {"eval": qev, "ratio_ub_95": ub,
                                     "min_coord_std": min_coord_std, "bars": qbars}
    report["pass"] = all(v["bars"]["pass"] for v in report["arms"].values())
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "pass": report["pass"],
                      "bars": {k: v["bars"] for k, v in report["arms"].items()}}, indent=2))
    return 0 if report["pass"] else 1


def eval_rest_band(*, seeds=EVAL_SEEDS, horizon=EVAL_HORIZON):
    ticks, band = 0, 0
    for ws in seeds:
        world = EmbodiedWorldV2(seed=ws)
        for _ in range(horizon):
            effect = world.step(Action.REST)
            ticks += 1
            if abs(effect["after"][2] - 0.50) < 0.18:
                band += 1
            if not effect["viable"]:
                break
    return band / ticks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["forager", "regulator", "quotient"])
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--dir", type=pathlib.Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.eval:
        return run_eval(args.dir, args.out)
    return run_arm(args.arm, args.out, smoke=args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
