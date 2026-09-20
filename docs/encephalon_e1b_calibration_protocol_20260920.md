# E1-B resource-world calibration

Prospective instrument protocol, 2026-09-20. This precedes E1-B neural fitting.
No result from the scripted references qualifies Zeus. Original E0/E1-A sources
and verdicts remain unchanged. The user authorized this preparation and E1-B.

Use `core/encephalon_resources.py`: the original bodily costs, five positions,
six actions and nine public sensors; both end stations provide food, only the
seed-selected station provides repair. Finite stock capacity is420 per patch,
renewal4 per patch per physical tick. The abundant control stays full and records
the external replacement of every harvested unit. Food slots report stock/420.
All observations are public. Consumption transfers only actual bodily headroom.
Pay bodily costs first, execute only if still alive, then renew both patches,
including the terminal tick. Death is final; crossing a boundary never refills
a patch. Snapshot/restore preserves complete stocks, body, clock and ledger.

Seed modulo16 balances both repair sides and four starting stock pairs:
(420,420), (105,420), (420,105), (210,210). Abundant patches always start full.
Seeds320100000 through320100031, separately reserved from neural training and
evaluation. Three starting reserve profiles: (850,900), (180,900), (850,120).
Each of two exact repeats runs both resource modes and adaptive, resident, and
repair-disabled adaptive references, 4096 ticks:576 bodies per repeat.

All18 mode/control/profile cells have an exact expected result: every adaptive
body survives unless repair is disabled; every abundant resident survives;
every finite resident dies. All finite adaptive survivors must consume food at
both patches. Disabled repair dies by ceil(initial integrity/3). Finite residents
never harvest the other patch and die by474 ticks, the conservative bound from
at most1420 initial local/body energy, metabolism7 and local renewal4. Action
costs, transit and discarded renewal can only shorten that bound.

Two mirrored public witnesses independently establish a reachable indefinitely
repeatable100-tick maintenance cycle. From the balanced birth, move twice to
repair; then FEED, REPAIR,44 WAIT, four moves away, FEED,45 WAIT, four moves back.
Run two cycles. The entire state determining future physics and scheduled-cycle
phase repeats at ticks102 and202; the absolute clock and cumulative accounting
continue increasing and do not affect the time-homogeneous rules. Each repeated
cycle transfers746 food energy and spends746. This is a scripted feasibility
witness, never an action label or a learned/emergent cycle claim.

The independent implementation reconstructs every initial body, every public
reference action, all integer transitions, actual consumption, supply/overflow,
body accounting, terminal states and both witnesses. Exact compressed twin
bytes and all per-cell gates are required. Commit source identities and this
protocol before the formal calibration. Publish action packets, manifests,
hashes and the independently verified result in compact gzip/JSON artifacts.
Any gate failure rejects world qualification; an uninterpretable integrity defect
is VOID. Do not launch neural fitting from either outcome.

Development note: an initial reference harvested whenever energy fell below650.
That rule overharvested small renewals and failed low-energy starts. Before
formal calibration, its adaptive threshold was changed to400, retaining an
affordable transit reserve. No resource parameters or candidate neural behavior
were tuned from an endpoint. Six development tests cover independent arithmetic,
terminal cost ordering, stock conservation, corruption rejection, continuation,
the repeatable witness and reference agreement across starting-stock strata.
