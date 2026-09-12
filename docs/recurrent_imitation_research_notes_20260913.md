# Recurrent imitation: literature guidance after LMB2

Elicit search attempted under the user's authorization; API returns
api_access_denied because the connected plan lacks API access. No Elicit evidence
was obtained. Undermind semantic search and full-text reader returned [Ngu22].
Primary arXiv HTML2211.01991v2 was checked directly; the PMLR publication is2023,
while the preprint/CoRL meeting is2022. These notes do not amend a frozen campaign.

[Ngu22] studies a fully observable state expert supervising a partially observable
history policy. Sections3.3–4 require recurrent history handling and warn that
state experts can omit information-gathering behavior. COSIL combines task reward
with an expert-divergence penalty and adapts its weight; it uses privileged state
access during training. AppendixB defines an optimality gap for this projection.
Section5.3 reports degradation when RL starts after BC; this is a BC-to-RL result.
The paper also reports settings where imitation works well. It is not a general
claim that imitation under partial observability must fail.

Zeus has a public-history teacher, not a state oracle. LMB2 uses BC and cue losses,
with no RL update. Therefore the cross-observability gap and BC-to-RL degradation
are not established explanations of its learner-history failure. The full-text
reader's suggested RL analogy is specifically rejected here. Episode-based
training does not verify our exact32-step truncation or guarantee our endpoint.

Useful next diagnostic question: is the fixed public teacher actually able to
recover from the learner's unusual histories, and does it use all information
available in its public consequences? Test this directly before designing another
aggregation mechanism. Memory changes after repeated writes, recurrent state
coverage and objective interference remain separate hypotheses. None is diagnosed
by this paper or the raw LMB2 arm comparison alone. First complete fresh LMB3
controller qualification; keep all four candidates and its protocol unchanged.

Paper: [Ngu22]. Primary sources checked:
https://arxiv.org/html/2211.01991v2 and
https://proceedings.mlr.press/v205/nguyen23a.html.
