# Necessary energy budget in the repair-dependent world

This is an explanatory bound, not an added gate or a rescue of LMT1.

For the configured body, initial energy is .85, death threshold .02, metabolism
.012 per step, extraction limit .13, efficiency floor .05, efficiency gain .55,
and harvest cost .002. Quality and tool efficiency are at most one; other action
costs are nonnegative. There is no external energy gain or within-body reset.

For any viable prefix of T steps, with N_H total HARVEST actions, energy satisfies

    E_T <= .85 - .012 T + [.13(.05 + .55) - .002] N_H
        = .85 - .012 T + .076 N_H.

Each harvest contribution is bounded using maximum extraction, quality and tool
efficiency. Upper clipping can only remove energy. Lower clipping cannot activate
in a viable prefix, because it would terminate the body. The inequality is
conditional on viability; it is not a bound on arbitrarily lower-clipped terminal
energy when the right side becomes negative.

Survival through T = 4096 requires E_T > .02, hence

    N_H > (.012 * 4096 - .83) / .076 = 635.828947...

At least 636 total HARVEST actions are therefore necessary. This is not sufficient:
real resource limits, tool wear, inspection, travel and repair costs tighten the
budget. Count total harvests, not only actions with positive net energy change;
a harvest can contribute energy without exceeding that step's metabolism.

Short-horizon feeding counts and occasional repairs do not establish this budget
over long operation. The bound supplies a physical explanation to inspect, not
an inference that any recorded short policy continues unchanged indefinitely.
