# Prospective dual replay instrument validation

Engineered instrument only. All fixtures are exposed LMB2 source collections or
small declared mechanics data; no old endpoint is requalified. LMB2 remains VOID.
Commit this protocol, instrument, observer and tests before collection replay.

The complete replayer starts from zero and public preparation records, reconstructs
eligibility and previous executed action/reward/done, continues both recurrent
states, independently samples raw actions and replays every public physical step.
Saved hidden states/probabilities are assertions only, never inputs or resets.
Full CPU float32 probabilities and h/z must match exactly. No action masks or
teacher substitutions. Public-only teacher labels are checked independently.

It uses the installed PyTorch GRU primitive and linear/softmax/sampling operators.
This is deliberately shared implementation, not an independent numerical engine.
Separate explicit NumPy algebra checks every neural transition from the replayer's
own computed pre-state: probability error<2e-5 and recurrent state error<1e-4.
These are local checks, not a relaxed whole-trajectory NumPy bound. This instrument
does not establish cross-engine portability or independently reconstruct Adam.
Training provenance/optimizer twins remain separate audit obligations.

Validation covers all four demonstration parents, all eight recorded collection
phases and eight bodies per phase:256 actual exposed bodies, without choosing
favourable cases or skipping deaths. All original source/data identities, twin
collection bytes and source checkpoint logical identities must verify. Complete
source physics and public teacher calibration are independently replayed. The
known accumulating-engine rejection is included explicitly, not hidden.

Tamper tests must reject changed actions, rewards, labels, probabilities and h/z;
public invalid sensor fields must not leak into neural inputs. Public preparation
storage must match the original consolidator exactly. NumPy GRU corruption must
be rejected even where shared-engine trajectory agrees. All decisions/writes and
maximum local errors are reported. Any exception or incomplete coverage means
instrument FAIL, not a neural qualification. PASS buys only preparation of a
fresh fixed-candidate body qualification protocol and endpoint, not transfer.
