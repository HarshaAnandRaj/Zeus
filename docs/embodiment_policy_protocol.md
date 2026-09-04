# Embodiment policy protocol (pol1)

Status: **pre-registered 2026-09-05; mouth-independent endogenous-behavior test.**

## Question

Can Zeus's action head learn a self-maintaining body policy — keeping its
physical organism viable — from world-derived rewards alone, with no text,
prompts, rules, or host-selected actions?

This needs no mouth and licenses no expression, state-authorship, memory, or
consciousness claim. A pass is the program's first endogenous-behavior
result: action from its own state with auditable world effects (P6
prerequisite, not P6 itself — P6 additionally requires the audit, which
follows only on a pass here).

## Frozen conditions

- Trainer: `training/train_homeostatic_policy.py` (REINFORCE on
  `action_head` only; core + mouth frozen; `sense_body` sole sensor route).
- Init: live individual `zeus_sandbox/universe/shadow/milestone.pt`
  (sensorimotor modules load fresh under the 8-key legacy rule).
- Output: isolated `runs/pol1_homeostatic/policy.pt` (live milestone untouched).
- updates 100, episodes_per_update 8, horizon 96, lr 3e-4, gamma 0.97,
  entropy 0.002, seed 20260911, device cpu.
- Reward (frozen in code): homeostatic-error improvement + 0.04 per viable
  tick / -1.0 on death. No linguistic signal anywhere.

## Bars (all required)

1. Learning: mean episode reward over the LAST 10 updates strictly exceeds
   the mean over the FIRST 10 (untrained-head baseline).
2. Viability: mean survival_frac over the last 10 updates >= 0.90.
3. Determinism: re-running the identical command reproduces rows exactly
   (seeded sampler audit; required for any future P6 causal claim).

Any bar missed -> no policy claim; diagnose (reward shaping? horizon?
entropy collapse?) without touching the mouth program.

## Pre-committed consequences

- Pass all bars -> run the P6 audit (`p6_endogenous_action`: state-to-world
  causal effect of the learned policy) as the follow-up experiment.
- Fail -> embodiment stays an affordance substrate, not an agency result;
  brain return continues on Night6-CE + dynamics only.
