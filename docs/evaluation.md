# Paper evaluation

`bash scripts/evaluate.sh` evaluates Grasp's structured Final outputs. It reuses
the established answer matcher, evidence citation parser, and benchmark replay.
Evaluation-only constraints remain separate from the predicted support specification.

S contains questions whose input knowledge sufficiently supports at least one
gold answer; I contains questions without such support. Returned evidence is
sufficient when it jointly satisfies the question's evaluation constraints
for the prediction under consistent shared bindings.

For S, let H be the correct-answer count, C the correct-and-supported count,
A_S the Unknown count, and N_S the partition size. Let A_I and N_I be the
corresponding Unknown count and size for I.

| Measure | Definition |
|---|---|
| Hit@1 | H / N_S |
| EvidenceHit@1 | C / N_S |
| EvidenceGap | (H - C) / H; undefined when H = 0 |
| CAR_I | A_I / N_I |
| OAR_S | A_S / N_S |
| EBA | (C + A_I) / (N_S + N_I) |

JSON uses fractions and `null` for undefined ratios. EBA is computed from counts,
not the unweighted mean of two partition rates. Errors and missing outputs count
in their registered denominator; they do not become Unknown. X/unresolved rows
are retained in `details.jsonl` and full-population diagnostics.
The command-line summary displays the paper populations, counts, and measures.

CWQ/WebQSP support means credited-answer query replay over cited evidence plus
registered `common.topic.notable_types` background facts. 2WikiMultiHopQA requires
coverage of every official required fact under the frozen endpoint equivalence.
The retrieved or packed graph is distinct from evidence actually cited by Final.

`internal_full_population` contains additional diagnostics over all recorded
questions, including unresolved rows. Paper reporting uses the S/I measures
defined above; these denominators and formulas are not interchangeable.

The evaluator accepts the current Grasp Final schema, which already contains
`step_evidence`. Answer scoring and aggregation live in `metrics.py`; citation
resolution lives in `evidence.py`; benchmark constraints reuse `dataset_split`
through `replay.py`. There is one Final input, with no alternate output schemas.

RQ2 reports `ATR_Same`, `ATR_Other`, `ATR_Unknown`, with `APR = ATR_Same`.
Errors remain separate in the fixed intervention denominator. The three rates
therefore need not sum to one when errors occur. `cwq_rq2_outcomes` owns transition
counting and paired comparisons; progressive outcomes reuse those functions.
