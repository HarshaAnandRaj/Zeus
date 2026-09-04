"""Fail-closed evidence criteria for unsolicited conversational initiation."""

from collections import Counter

from training.free_run_gate import gate_decision


def assess_initiative(events, counterfactuals, *, min_speech=3):
    """Assess model-selected, state-caused initiation without topic templates.

    ``events`` are model decision opportunities, not arbitrary timer ticks. A
    speaking event must be explicitly unprompted and contain only model output.
    ``counterfactuals`` replay the same state/seed and then perturb state while
    holding the policy/noise condition fixed; they establish that selection is
    reproducible yet state-sensitive rather than a periodic scheduler.
    """
    speaks = [e for e in events if e.get("action") == "speak"]
    waits = [e for e in events if e.get("action") == "wait"]
    unprompted = bool(speaks) and all(not e.get("external_prompt", True) and
                                      e.get("source") == "model" for e in speaks)
    gates = [gate_decision(e.get("text", "").split())[0] for e in speaks]
    readable = bool(gates) and all(g.passes for g in gates)
    prefixes = [" ".join(e.get("text", "").lower().split()[:6]) for e in speaks]
    non_template = bool(prefixes) and max(Counter(prefixes).values()) == 1
    replay = [c for c in counterfactuals if c.get("replay_action") == c.get("base_action")]
    changed = [c for c in counterfactuals if c.get("perturbed_action") != c.get("base_action")]
    state_causal = (len(counterfactuals) >= 8 and
                    len(replay) / len(counterfactuals) >= 0.8 and
                    len(changed) >= 3)
    passed = bool(len(speaks) >= min_speech and len(waits) >= min_speech and
                  unprompted and readable and non_template and state_causal)
    return {
        "n_events": len(events), "speak_n": len(speaks), "wait_n": len(waits),
        "unprompted": unprompted, "readable": readable, "non_template": non_template,
        "state_causal": state_causal, "counterfactual_n": len(counterfactuals),
        "replay_match_frac": round(len(replay) / len(counterfactuals), 3)
                              if counterfactuals else 0.0,
        "state_flip_n": len(changed), "pass": passed,
    }
