# Fresh repair-dependent family: proposed physics and necessity bound

Design draft only. No factory, controller fit, calibration or endpoint launched.
No existing world configuration or frozen campaign changes. Motivated by the
audited original-world no-repair counterexample, this supports the still-open
long self-maintenance obligation rather than substituting for memory transfer.

## One proposed physical change

Create a separately versioned static lineage family with original map, action
costs, resources, inspection interface, damage, repair and birth conditions.
Change only `efficiency_floor` from .35 to .05. Retain `efficiency_gain=.55`,
`extraction_limit=.13`, metabolism .012, harvest cost .002, initial tool .90,
wear .008 per HARVEST, initial energy .85 and death threshold .02.
This is engineered physiology, not an emergent need. The workshop must genuinely
restore a useful capability; successful repair counts alone are insufficient.

## Model-level necessity result under explicit assumptions

Consider a single continuous body with the above constants, no MAINTAIN actions,
no external energy or tool restoration and no body reset. Only HARVEST supplies
energy, each amount is at most .13, quality is at most1, every HARVEST incurs
.008 tool wear, and all other action costs are nonnegative. Tool condition before
the j-th harvest, indexing j from0, is max(.90-.008j,0).

Total energy obtainable from the initial tool bonus is bounded by

`B = .13*.55*sum(max(.90-.008j,0), j=0..112) = 3.651934`.

For T survived steps and N harvests, with N<=T, clipping at the upper energy
limit can only discard energy. Viability excludes lower clipping before death.
Therefore

`E_T <= .85-.012T+(.13*.05-.002)N+B`

`    <= .85-.0075T+B`.

At T=598 this upper bound is .016934, below the .02 viable threshold. Thus
**no action sequence without MAINTAIN can survive through step598** in this
proposed single-body model, even with perfect safe-side knowledge and maximum
extraction every harvest. Contamination, low stock, travel and inspection can
only worsen the bound. This does not rely on a failed reference policy or a
neural optimizer. Arithmetic was checked directly with the113 positive terms;
the result is a configured model bound, not a tested runtime or universal theorem.

An attempted MAINTAIN away from the workshop also cannot restore the tool.
Extending the bound to policies with such ineffective attempts is immediate
from their nonnegative costs and unchanged wear/bonus accounting. To establish
that *effective physical repair* is necessary, a prospective audit must bind
actual workshop tool increases rather than count action names alone.

## Calibration still owed before model evaluation

Freeze the separate factory, exact public/physical auditor and balanced fresh
ecologies before compute. A declared public inspection/repair reference must
survive a long horizon, for example4096 continuous steps, with all costs and
deaths retained. Public no-repair and passive references must remain visible.
Independently replay every transition and scalar balance; verify the factory
only applies the declared configuration change and preserves world randomness.
Check the analytical energy envelope on no-effective-repair histories.

Repair-enabled feasibility is **unproved**. If calibration fails, preserve the
failure; do not quietly tune constants, reference thresholds or remove worlds.
A fresh mechanism/protocol would be needed. Changing worlds is a new engineering
choice, not a rescue of LMB4 or a claim that current body controllers transfer.

Later neural maintenance needs its own preregistered raw controller qualification,
effective-repair interventions and independent audit. Memory inheritance, own
acquisition, revision, authorship, priorities, initiation, expression and joint
six-pillar operation remain separate obligations.
