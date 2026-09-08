"""Write the observation ledger, with all groups and explicit interpretation limits."""
import numpy as np
from tools import obs1_state_discovery_20260908 as O


def main():
    r=O.R.N.read(O.OUT/'catalogue.json');audit=O.R.N.read(O.OUT/'completion_audit.json');assert audit['passed']
    derived={};geometry=[];tails=[];prefixes=[]
    for ti,t in enumerate(O.R.TRIALS):
        raw=O.R.N.read(O.OUT/f'{t}.json');derived[t]={}
        for c,s in r['trials'][t]['summaries'].items():
            linear=r['trials'][t]['linear'][c]
            geometry.append(f"| {ti+1} | {c} | {s['participation_dimension']['median']:.3f} | {s['saturation']['median']:.3f} | {s['null_variation_fraction']['median']:.3f} | {linear['current_input']['test_r2']:.3f} | {linear['input_body_age']['test_r2']:.3f} |")
            eligible=s['tail_nondegenerate'];fmt=lambda key: f"{s[key]['median']:.4f}" if eligible else 'NA'
            modal=max(s['tail_lag_counts'],key=s['tail_lag_counts'].get) if eligible else None
            mode=f"{modal} ({s['tail_lag_counts'][modal]}/{eligible})" if eligible else 'NA'
            tails.append(f"| {ti+1} | {c} | {eligible}/256 | {mode} | {fmt('tail_lag_one_ratio')} | {fmt('tail_best_ratio')} | {fmt('tail_input_ratio_at_best')} | {fmt('tail_action_agreement')} |")
        for c in ('erased','swapped'):
            ps=[p for p in raw['history_pairs'] if p['control']==c];long=[p for p in ps if p['equal_input_prefix']>=2]
            full=[p for p in ps if p['equal_input_prefix']==496]
            ratio=[p['final_prefix_hidden_distance']/p['initial_hidden_distance'] for p in long if p['initial_hidden_distance']>0]
            d=dict(pairs=len(ps),prefix_at_least_two=len(long),endpoint_distance_growth=sum(v>1 for v in ratio),
                median_endpoint_distance_ratio=float(np.median(ratio)) if ratio else None,longest_prefix=max(p['equal_input_prefix'] for p in ps),
                full_496_prefixes=len(full),full_prefix_final_hidden_median=float(np.median([p['final_prefix_hidden_distance'] for p in full])) if full else None)
            derived[t][c]=d
            ratio_text=f"{d['median_endpoint_distance_ratio']:.6g}" if ratio else 'NA'
            prefixes.append(f"| {ti+1} | {c} | {len(long)}/256 | {d['longest_prefix']} | {ratio_text} | {d['endpoint_distance_growth']} | {len(full)} |")
    O.R.C.save(O.OUT/'derived_prefix_descriptions.json',dict(kind='OBS1_DERIVED_DESCRIPTIONS',exploratory=True,
        derivation='Post-extraction summaries of audited initial/final equal-input prefix distances; not new trajectories or a contraction test.',
        catalogue_sha=O.sha(O.OUT/'catalogue.json'),source_sha=O.sha(__file__),trials=derived))
    text='''# OBS1: observed recurrent organization, 2026-09-08

Status: **discovery pass complete; nine audit checks pass.** No usefulness gate
was applied. The scope is eight CYC6 GRU memory models, four conditions and8192
saved episodes. This is not a new ZeusCore/mouth experiment or a claim that an
endogenous self has emerged. "Emergent" is the search umbrella requested by the
user; the catalogue first records organization and leaves its function open.

Protocol9249d4a and five-test instrument8c79d5a precede extraction. Prior survival
outcomes were already known. Every initialization/control is included. Twin A is
used once; the exact twin is not extra independent evidence. No new neural
forward pass, training, world simulation or deployment was used by extraction.

## Observation ledger

**O1. Concentrated state variation.** Although the hidden state has32 coordinates,
median covariance participation dimension on intact trajectories ranges1.546 to
3.341 across models. In six models it lies2.980..3.341; two lie1.546 and1.896.
Those two also happened to fail CYC6, but failure was not an admission criterion
and this association is not a causal result. Low participation dimension measures
concentration of linear variance, not an exact-dimensional manifold, a symbolic
representation, or a mechanism discovered by itself. Input repertoire, lifetime
length and supplied control all affect the sampled geometry.

**O2. Near-boundary occupation varies substantially.** Trained intact trajectories
spend median10.1%..43.5% of coordinate-time samples beyond|h|>.95; untrained group
medians are zero. The bounded activation is designed. Its degree of occupation
is measured, not a target in the resource loss. Different trajectory inputs and
durations prevent attributing the whole difference to learned weights here.
The number does not establish persistent individual neurons or an attractor.

**O3. Smooth evolution and partial return rhythms coexist.** Six intact models
choose lag2 in every eligible tail, but lag1 displacement is smaller still and
the median return profile rises with lag. This is consistent with smooth local
evolution/drift, not a detected two-step cycle. Their lag2 hidden-displacement
ratios are.051..078, while corresponding input ratios are approximately.77..81;
the hidden stream is temporally smoother in this normalized comparison.

Model7(initialization20261107) instead selects lag6 in221/256 intact tails.
Across its eligible tails, median best return ratio is.423 versus lag1 .617
and a shuffled-order minimum baseline .835. Median matching-lag action agreement
is.517 and input ratio .621. This is a partial coupled return rhythm, not exact
recurrence or an autonomous oscillator: the world, input and supplied controller
also have structure. The first fixed illustration actually selects lag7, which
is retained rather than replaced by a more attractive lag6 example.

Untrained models also show return structure: model5 favors lag7 in70/128 eligible
tails and model6 favors lag6 in97/128. This keeps initialization and controller-
driven rhythms in the catalogue. Recurrence is not restricted to trained or
successful systems. Only3104/8192 episodes have64 decision rows; omitted tails
are NA with denominators, not absence of structure or exclusion from other metrics.

**O4. State motion occupies directions invisible to the immediate readout.** The
linear32->9 readout has rank9 in every group, so a23-dimensional null space is
structurally guaranteed. Intact trained median fractions of centered movement in
that null space range20.0%..34.1%; untrained medians range49.2%..87.6%. The occupied
fraction is observed. Null-space existence is architectural. Motion invisible
to the present readout can still influence later recurrent states; no hidden
function, protected memory or irrelevance is inferred.

**O5. Historical separation can persist or wash out under identical inputs.**
All4096 intact/erased and intact/swapped comparisons start with the same current
neural input.1152 have at least two consecutive identical inputs. Every one of
those has a smaller final-prefix hidden separation than its initial separation.
This endpoint comparison is not monotonic contraction, a Lyapunov estimate or
proof that every input sequence forgets history.

Model6(initialization20261106) has70 intact/erased pairs with all496 subsequent
neural inputs identical; their median final hidden separation is8.10e-6. Model8
has one such full-length pair but retains separation.14049. These are recorded
differences in response to history, without requiring either preservation or
forgetting to be beneficial. Bodies/actions are not assumed matched merely
because the neural inputs match. Prefix summaries below are derived after
extraction from the audited distance records and are labelled exploratory.

**O6. Current input describes different amounts of hidden variation.** A linear
description from current location/resource has test R2 .294..370 in six intact
models, .895 in model5 and .615 in model7. Adding current body variables and age
raises these scores to.466..570,.973 and.733 respectively. The partition uses
already-observed world seeds, not a new confirmatory dataset. Unexplained linear
variance does not equal stored information, agency or nonlinear independence
from inputs. Near-perfect fits to short trajectories are retained and labelled;
they can reflect limited visited configurations and duplicate sampled rows.

## Alternative explanations and next questions

The small short-lag distances are not cycle evidence; the rank-deficient readout
does not by itself demonstrate emergent storage. GRU gating already permits
smoothing and the activation is bounded. All plots sample an engineered world
under a supplied controller. Those explanations do not erase the observations;
they identify what must be separated to understand how a pattern arises.

Candidate causal questions are: whether the lag6 pattern persists with a matched
or fixed input stream; whether history-sensitive directions affect subsequent
state evolution despite immediate readout invisibility; and which input sequences
preserve or erase initial distinctions. None is admitted or rejected on utility.
No such new intervention was performed or automatically authorized by OBS1.

This pass is bounded, not exhaustive. Full per-episode measurements remain in
the local OBS1 artifacts, including small effects not highlighted here. No
functional verdict is revised; the CYC6 reliability FAIL remains intact.

## Complete group tables

Initialization index1..8 means20261101..20261108. Each group has256 episodes.
Dimension/saturation/null columns are medians across whole episodes, with equal
episode weights. They therefore cover different lifetime lengths. Linear fits
sample16 equally spaced rows per episode, repeating indices for short episodes;
first64 world seeds fit, last64 evaluate. R2 is against the training hidden mean.

| Model | Condition | Dimension | Saturated fraction | Null fraction | Input R2 | Input+body+age R2 |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(geometry)+'''

Tail ratios normalize mean squared displacement by twice centered hidden energy.
The minimum excludes lag1; inspected lags are1..16,24,32. Action/input entries
are measured at each episode's selected lag. Selection is exploratory, and
the shuffled-order baseline undergoes the same minimum search.

| Model | Condition | Eligible | Modal lag (count) | Lag1 ratio | Best ratio | Input ratio | Action agreement |
|---|---|---:|---|---:|---:|---:|---:|
'''+ '\n'.join(tails)+'''

Paired history table: the distance ratio uses only prefixes of at least two
identical neural inputs. It compares last-prefix with first-prefix hidden
distance; it does not measure intermediate excursions.

| Model | Comparator | Prefix>=2 | Longest prefix | Median distance ratio | Endpoint growth count | Full496 prefixes |
|---|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(prefixes)+'''

## Figures and evidence

![All-group observation map](obs1_observation_map_20260908.png)

![Return-distance profiles with eligibility](obs1_return_profiles_20260908.png)

![Prespecified fixed cases](obs1_fixed_state_paths_20260908.png)

All figures were visually inspected. The fixed examples use world202678000,
orientation2 for every model. PCA axes differ between models. Untrained paths
share raw unit indices with their trained counterpart but are not aligned learned
representations; those distances should not be interpreted semantically.

Nine independent audit checks pass: source/artifact hashes; all8192 episode IDs
and lengths;128 raw geometry/recurrence cases using independent SVD and direct
distance calculations; all4096 input prefixes/null projections; every sampled
array matched to raw trajectories; all64 linear descriptions via independent
SVD; all group aggregates/eligibility; fixed examples; final source integrity.
Extraction and audit exit0. This checks measurement integrity, not emergence.

Canonical catalogue/audit/derived-prefix records are in
zeus_sandbox/universe/reports/obs1_{catalogue,completion_audit,derived_prefix_descriptions}_20260908.json.
Detailed per-episode records and sample arrays remain in runs/obs1_20260908,
with hashes in the catalogue. Frozen source and prior-run hashes are retained.

## Charter interpretation

This is user-authorized Class O discovery, with usefulness evaluated after
identification when appropriate. The designed setup and selection context are
explicit; unexpected amounts or patterns are candidates, not certified
unprogrammed setpoints. Simple recurrent-system explanations remain available;
no special CDT or pillar claim follows. The audit passes. This observational
authorization is complete and purchases no automatic next experiment.
'''
    path=O.ROOT/'docs/obs1_state_discovery_review_20260908.md'
    with path.open('x',encoding='utf-8') as f:f.write(text)


if __name__=='__main__':main()
