# Frozen support-specification audit

Protocol version: **v4.1 with verdict confidence**, 2026-10-06. All 8,756 frozen paper questions receive fresh three-way judgments and an integer 0-10 confidence score. Earlier runs retain their original prompts and decisions; no earlier verdict is reused or assigned a score afterward. Normalization preserves material reference conflicts and distinguishes intrinsic temporal conditions from independent acquisitions of date values.

## Scope and record boundary

This audit measures whether Grasp's frozen predicted support specification represents the benchmark's answer-support requirements. It evaluates the complete `Plan`: retrieval/intersection steps, input dependencies, role annotations, and final `decision`. It does not evaluate relation-scoring choices, graph coverage, retrieved evidence, final predictions or support-availability labels.

The evaluator is a model-assisted semantic judge. Its outputs are not human annotations or executable proofs of semantic equivalence. Report the evaluator model/configuration, audited population, failed requests, uncertain cases, and actual pilot-review findings with the results. The existing GPT-OSS planner and the audit evaluator are separate components even when both are LLMs.

`judge_prompt.txt` is the fixed evaluator instruction. `decision.schema.json` is its strict JSON output contract, with the wrapper `{"audits": [...]}`. A batch returns one record per input in input order. Record IDs and dataset names are copied exactly. Every object has all required fields and forbids extra fields.

Inputs contain `dataset`, `id`, `question`, `linked_entities`, `plan`, structured gold annotations, `plan_missing`, `plan_invalid`, and `plan_validation_error`; readable invalid output may be supplied as `raw_plan_output`. CWQ/WebQSP provide `gold_query` with eligible queries and per-query entity-name mappings. 2WikiData provides `gold_evidence` and `gold_annotation_identity`, official required evidence as an anonymized role graph plus role metadata: question entities remain anchors, and intermediate/answer entities become roles whose repeated occurrences retain identity. Neither predicted answers nor benchmark gold-answer lists are needed by the judge. Do not attach S/I labels, gold-query replay outcomes, downstream answers, success labels or evidence scores to judge records.

## Semantic unit and reference selection

Extract gold semantic atoms first, then predicted semantic atoms, then match them. The counted atoms use three kinds:

| Kind | Unit |
|---|---|
| `relation` | A directed question-level relation from an anchor/input role to an output role; acquisition of a property value is a relation. |
| `type` | A meaningful category restriction on an intermediate or answer role. |
| `filter` | A required literal/property condition with its operator, direction, threshold or temporal scope. |

An atom includes its participant roles and constants. Wrong direction, wrong anchor or a weaker/stronger relation is not a full match. Semantic equivalence allows paraphrase, role/variable renaming, equivalent decompositions, valid inverse formulations and CVT folding with preserved observable roles. This comparison is not one SPARQL triple versus one Retrieve, exact text matching, or equality of step counts.

Deduplicate synonymous repeated atoms before one-to-one matching. Normalize hidden CVT-mediated paths into question-level semantic relations; preserve a distinct question-relevant intermediate role. Types expressed by a precise `result_role` or `decision` do not require a separate type-retrieval step. Exclude prefixes, DISTINCT, label/name lookup, language filters and literal-format guards without an independent question-level condition. Retain meaningful exclusions, categories, temporal bounds and answer-changing comparisons. Do not discard all FILTER clauses. A declared inverse reverses endpoints consistently; approximate relation similarity alone does not establish equivalence.

WebQSP eligible complete parses are alternative references, not a conjunction across annotations. Choose one coherent eligible reference and identify it in `gold_reference`; never combine pieces of different parses. Within a single query, UNION retains the full question-level OR semantics. Do not demand all disjuncts jointly. Conversely, do not select a convenient branch to turn an incomplete prediction into a correct full specification: if the question genuinely asks for alternatives, extract their distinct atoms with branch locators and diagnose an omitted alternative in coverage and structure. Required conjunctions retain all conditions and shared bindings. If the Plan changes OR into AND, diagnose the extra restriction and structural error.

A complete valid branch can provide support for one answer without recovering the complete question-level disjunction. Such route validity is a limited diagnostic, recorded in `notes`, rather than full semantic equivalence or an automatic `correct` verdict. Equivalent storage-path alternatives need not be separately reproduced. Official temporal `NOT EXISTS` alternatives may be normalized to a complete positive-value branch under the active evaluation's explicit-evidence criterion; attribute absence alone does not supply temporal comparison evidence. Record that specific normalization. The audit does not require enumeration of all gold answers, but does require the relevant question logic.

For 2WikiData, preserve direction and question-required shared anonymous role identities in the required-triple graph. Equal annotated dates/numbers can be incidental equality, not a required binding: follow the question's dependencies/conjunctions and supplied `role_policy`, rather than forcing a join merely because two evidence facts reuse a value role. Determine the requested final operation from the question because evidence triples alone may not specify comparison, superlative or projection behavior. Anonymous gold roles are not missing predictions. Do not require the judge to recover their original entity names or produce the gold answer. Material gold-question conflicts are recorded as uncertainty rather than silently changing the reference.

Keep benchmark-derived requirements distinct from question-alignment judgements. An explicit gold-only qualification can yield a definite relative-to-gold omission even when the natural question does not state that qualification: preserve the omission and record the wording discrepancy in `notes`, rather than claiming a demonstrated natural-question misunderstanding. Do not erase the gold clause or add it to the Plan.

A materially conflicting operation/role scope needs localized uncertainty. For example, a question-specific office/event scope versus an unrestricted gold ordering domain does not justify declaring the Plan's operand or binding definitely wrong merely by privileging the unrestricted reference. Mark affected atom alignments and applicable dimensions uncertain, with their source discrepancy. Independent, clearly missing property acquisitions can remain definite errors, so an overall incorrect verdict may coexist with these local uncertainties. Do not infer answers or resolve the conflict using external knowledge.

Projection, filtering/comparison direction, aggregation and ordering are also judged in `answer_operation`, wherever they occur in the complete Plan. Do not add a second counted atom merely for projection/min/max/count. A threshold condition and acquisition of its property are distinct atoms; a superlative's property acquisition is counted, while min/max is an operation. This keeps constraint coverage interpretable while exposing answer-operation failures separately.

A `decision` can state a type/filter condition, but merely mentioning unavailable values does not create a relation-acquisition atom without a retrieval path. Missing acquisition is a coverage error; label `wrong_answer_operation` only for an incorrect/undefined operation or incorrect operands/answer role, rather than automatically duplicating every missing-property error in this dimension.

Normalize both gold and Plan into the same observable requirements. A time-qualified Retrieve such as "Governor of Ohio in 2011" or "presidents serving during a named war" expresses a tenure/time condition, not separate returned start/end date values. Fold date fields used solely for that intrinsic condition, including tenure/CVT fields and the named event's boundaries, into the qualified relation on both sides. Retain the meaningful temporal condition as a filter atom. Dates independently used for comparison, ordering or another dependency beyond that condition remain separate acquisitions and require an expressed retrieval/binding. Apply this equally to readable invalid raw output. Never infer a missing acquisition from gold, or interpret raw `aN` tokens using the order of `linked_entities` without an explicit supplied mapping.

Name/linguistic-hint lookup used only to express an answer-changing category or text condition is normalized into that type/filter requirement. It does not create a second question-level relation or make structure/binding applicable by itself. Preserve the meaningful condition; do not discard it as generic query machinery.

The required operation can be expressed in a Retrieve instruction, role/dependency context, or `decision`. The final instruction need not repeat an already unambiguous operation: a step asking for the earliest/latest/predominant result followed by "Return the results from s1" can express the correct min-time/max-time/argmax-share operation. If the necessary date/percentage/measure acquisition is absent, diagnose that as `missing_constraint`; the operation may still be correct. A wrong direction, wrong operand/association, wrong projection, nonexistent referenced result, or absence of the required operation itself warrants `wrong_answer_operation`. Do not infer an operation solely from the original question when the entire Plan omits it.

## Decisions and error labels

| Item | Correctness criterion | Applicability |
|---|---|---|
| Constraint coverage | All unique necessary gold atoms have semantically equivalent predicted atoms. | Every record with extractable gold atoms. |
| Extra requirements | No unmatched predicted atom imposes an unjustified requirement, unrelated retrieval obligation, or candidate restriction. Harmless redundancy is deduplicated. | Every record with predicted atoms. |
| `structure` | Composition inputs/order, branch dependencies and AND/OR behavior represent a coherent gold support route. | Multiple relational requirements compose, branch, or form alternatives. |
| `binding` | Required intermediate/answer/value identities and candidate-property associations are explicit and preserved. | Multiple requirements or a final operation must share a role/value. |
| `answer_operation` | The whole Plan expresses the correct operation and returned role; `decision` preserves an operation already expressed in a retrieval instruction or states it explicitly. Missing attribute acquisition alone is assessed in coverage. | Every record, including direct questions. |

Each dimension returns `correct`, `incorrect`, `uncertain`, or `not_applicable` with its reason and gold/Plan locators. Absence of an `Intersect` token alone is not an error if dependencies or an explicit decision express the same shared-identity requirement and preserve the needed associations. Independently retrieved local facts without the required common entity/value are insufficient. Do not mistake a semantically correct Plan whose grounding later fails for an incorrect specification.

Applicability uses normalized question-level acquisitions, not raw SPARQL joins. A hidden CVT identity or intrinsic qualifier of a single direct relation does not create a composition/binding task. For example, direct "married to" remains one question-level relation even when its gold query includes a marriage mediator and a union-type qualifier: its structure and binding are `not_applicable`. Distinct chained acquisitions and separately acquired candidate properties still require compatible identities.

Judge later steps with their resolved input role and inherited Plan context. Do not strip an earlier show/jurisdiction qualifier and then label the later generic wording wrong. Conversely, do not infer a shared episode/casting solely from the original question when the Plan loses that association. When a contextualized generic relation leaves a genuinely unresolved scoped/unscoped reading, keep the relevant alignment/binding uncertain. A predicate word such as `regular_cast` does not independently demand the English word "regular" from a Plan unless that distinction is a question-level requirement; preserve show-specific portrayal semantics instead of treating ontology vocabulary as a separate omission.

The complete verdict is:

- `incorrect`: missing/invalid Plan, a definite missing or spurious atom, or an incorrect applicable dimension;
- `uncertain`: no definite failure is established, but a material semantic alignment, reference or applicable dimension remains unresolved;
- `correct`: all gold atoms match, no spurious atoms exist, every applicable dimension is correct, and no material uncertainty remains.

Definite failures take precedence over unresolved other details. Keep uncertainty in the relevant records/dimensions and report it explicitly. A wrong relation may create both a missing gold atom and a spurious predicted atom. Fixed multilabel errors are `invalid_plan`, `missing_constraint`, `spurious_constraint`, `wrong_structure`, `wrong_binding`, `wrong_answer_operation`, and `wrong_anchor`. `wrong_anchor` requires an explicit wrong linked question entity; an anonymous intermediate/answer role is not an anchor error.

Supplied `plan_missing`/`plan_invalid` flags establish `plan_valid`; the judge does not invent a second parser. Missing Plans remain in the audited population, have empty predicted atoms/matches and missing extracted gold atoms. Readable invalid Plans retain any established semantic diagnostics, but overall remain incorrect. Failed evaluator requests are execution failures, not missing or invalid planner outputs, and must be counted separately.

## Output schema and source localization

Every audit contains:

- identity/validity: `dataset`, `id`, `plan_valid`, `gold_reference`;
- extracted requirements: `gold_constraints` and `predicted_constraints`, each containing `{id, kind, text, source}`;
- one-to-one `matches` containing `{gold_id, predicted_id, reason}`;
- `missing_gold_ids`, `spurious_predicted_ids`, and `uncertain_alignments` containing `{gold_ids, predicted_ids, reason}`;
- `structure`, `binding`, `answer_operation`, each containing `{status, reason, gold_sources, plan_sources}`;
- `overall`, fixed-enum `error_labels`, and concise `notes`.
- `confidence_score` (integer 0-10) and `confidence_reason` for the decisive verdict or main unresolved conflict.

Confidence measures the strength of support for a decisive correct/incorrect judgment. Both confidently correct and confidently incorrect Plans can score highly. Unresolved judgments have limited support for either binary label and receive lower scores; confidence in the existence of ambiguity is not a high-confidence binary decision. A recorded invalid Plan can yield high-confidence incorrect while local semantic details remain uncertain. Scores are self-reported, not P(Plan correct) or calibrated probabilities. The prompt provides descriptive anchors; no filtering cutoff is fixed during judging. Retain the original label and score for later threshold analysis, never automatically relabel from the score.

Use `g1...`/`p1...` IDs within a record. Each atom is matched once, definitely unmatched, or part of an unresolved alignment. Uncertain IDs are not simultaneously counted as definite omissions/extras. Locators identify fields and step/role IDs; they do not repeat full queries or long evidence lists. Judge-generated accuracy rates are not part of this schema.

## Aggregation and denominators

Report overall correct/incorrect/uncertain counts over the declared audited population, alongside Plan missing/invalid and evaluator failure counts. Keep uncertain cases visible; do not silently remove them to inflate specification accuracy. A full-population result is a model-assisted audit rather than a manually annotated benchmark.

Constraint recall is matched gold atoms divided by unique gold atoms; precision is matched predicted atoms divided by unique predicted atoms. Report micro counts and per-question macro averages with their respective populations. A zero denominator is undefined. Cases with unresolved atom alignments do not yield a point estimate of precision/recall: report their count separately rather than treating unresolved atoms as either matched or definitely wrong. Overall constraint coverage can additionally be reported as the fraction of assessable records with no missing gold atoms and no unresolved alignment. These atom measures do not substitute for structure, binding or final-operation accuracy.

For each structural dimension, list `N_correct`, `N_incorrect`, `N_uncertain`, and `N_not_applicable`. The applicable population excludes only `not_applicable`. Give correct/applicable as the observed correct fraction, plus the uncertain fraction; if reporting correct/(correct+incorrect), label that denominator as determinate applicable cases. Answer-operation applicability includes every record. Multilabel error frequencies may sum beyond 100%.

Join audit labels to downstream outcomes only after judgement. Within each overall group, report its own S and I question counts. S-group Hit@1/EvidenceHit@1/OAR_S use that group's S denominator; EvidenceGap uses that group's S correct-answer count and is undefined at zero; CAR_I uses that group's I denominator. EBA uses the group's count-weighted S+I population. Incorrect, correct and uncertain groups can have different question difficulty and S/I composition: these comparisons describe associations, not the causal effect of fixing a Plan.

## Actual 20-question pilot review

Before accepting all-question model decisions, review actual outputs on 20 fixed records covering the available datasets, direct/compositional/conjunctive questions, shared intermediate and answer bindings, filters, comparisons/superlatives, CVT/inverse representations, eligible alternatives, and any observed invalid/missing Plans. Selection is for rubric calibration, not estimation of population accuracy. Where a category has no available record, state that coverage gap.

For each pilot record inspect the gold extraction, predicted extraction, role mapping, one-to-one matches, exclusions of technical clauses, dimension applicability, source locators, and overall/error justification. Record concrete judgement defects and corrections; do not claim agreement merely because JSON validates. Fix a systematic rubric/payload problem before expanding the run. If the final prompt or input normalization changes, rerun affected pilot decisions and use the same final rubric for the full population. Preserve model output and the separate pilot-review finding; do not silently overwrite original verdicts.

This pilot is an agent/model review unless a person actually reviews it. It provides neither human inter-annotator agreement nor a measured human-validation accuracy. Further manual validation may be described only when performed.

## Local authority

- `ICLR_2027_overleaf/sections/05method.tex:6`: full specification contains retrieval/intersection dependencies and final answering instruction.
- `ICLR_2027_overleaf/sections/appendix_prompts.tex:22`: the Evidence DAG excludes the final answering instruction, while Plan includes it.
- `ICLR_2027_overleaf/sections/08Appendix.tex:239`: bindings and UNION alternatives in gold replay.
- `ICLR_2027_overleaf/sections/08Appendix.tex:243`: CVT folding and inverse relation conventions.
- `ICLR_2027_overleaf/sections/08Appendix.tex:254`: explicit comparison values and candidate ordering.
- `CONTEXT.md:54`: predicted reasoning-plan requirements remain distinct from evaluation-only constraints.
- `CONTEXT.md:141`: paper-facing metric populations and stored-field mapping.
