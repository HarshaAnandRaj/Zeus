"""Evidence gate for Zeus's functional self-organization program.

This is deliberately a *necessary-conditions* gate, not a consciousness
detector.  It keeps four distinct claims separate:

1. the mouth can express a sustained legible behaviour;
2. autonomous state has an observable causal effect on that behaviour;
3. live memory is both mechanically retrievable and self-selected/useful;
4. the unassisted dynamics remain viable after an internal perturbation.
5. the system initiates consequential actions from its own state, not only
   externally requested or scaffolded operations.
6. it can choose to begin a legible conversational topic without a prompt,
   and that choice is reproducibly caused by its internal state.

An all-pass result is evidence for a functionally self-organizing system under
this battery.  It is not evidence for human-like consciousness or phenomenology.
"""


def assess(language, causal, memory, resilience, action, initiative):
    """Combine independently measured necessary conditions without hiding failures."""
    expression = bool(language.get("pass", False))
    causal_state = bool(causal.get("pass", False)) and expression
    selective_memory = bool(memory.get("pass", False))
    intrinsic_resilience = bool(resilience.get("pass", False))
    endogenous_action = bool(action.get("pass", False))
    autonomous_initiation = bool(initiative.get("pass", False))

    gates = {
        "legible_expression": expression,
        "causal_state_expression": causal_state,
        "selective_live_memory": selective_memory,
        "intrinsic_resilience": intrinsic_resilience,
        "endogenous_consequential_action": endogenous_action,
        "state_caused_unsolicited_initiation": autonomous_initiation,
    }
    missing = [name for name, passed in gates.items() if not passed]
    if not expression:
        stage = "unreadable: behaviour cannot yet serve as evidence"
    elif not causal_state:
        stage = "expressive but not causally state-authored"
    elif not selective_memory:
        stage = "state-expressive but memory is not self-selected/useful"
    elif not intrinsic_resilience:
        stage = "stateful but not intrinsically resilient"
    elif not endogenous_action:
        stage = "self-maintaining but without endogenous consequential action"
    elif not autonomous_initiation:
        stage = "acting but without state-caused unsolicited conversation"
    else:
        stage = "functional self-organization pillars met"

    return {
        "gates": gates,
        "missing": missing,
        "stage": stage,
        "functionally_self_organizing": not missing,
    }
