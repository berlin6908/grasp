# Data and evidence contracts

## Input knowledge

The input knowledge G_q is a question-local graph. Grasp accepts JSONL, one
question per row:

```json
{"id":"q1","question":"Where was Ada born?","q_entity":["Ada"],"graph":[["Ada","birthplace","London"]],"answer":["London"]}
```

`id`, `question`, nonempty `q_entity`, and directed `[head, relation, tail]`
triples are required for support specification induction and grounding.
CWQ/WebQSP evaluation uses the `answer` list. The 2WikiMultiHopQA graph adapter
uses `ground_truth`, or the source
collection's scalar `answer` when no answer list is supplied.

## Sources

- CWQ and WebQSP: RoG question-local graph collections; official benchmark
  queries; NSM entity alignment for CWQ; official topic/constraint names and
  a fixed supplemental MID-name table for WebQSP; GrailQA inverse properties.
- 2WikiMultiHopQA: the graph snapshot released with DoG; official development
  records and official entity aliases/demonyms. These are data sources, not model
  predictions. The graph snapshot and annotation versions must match.

Obtain the annotations from the [CWQ authors](https://github.com/alontalmor/WebAsKB),
[WebQSP authors](https://www.microsoft.com/en-us/download/details.aspx?id=52763),
and [2WikiMultiHopQA authors](https://github.com/Alab-NII/2wikimultihop).
The existing graph sources are [RoG](https://github.com/RManLuo/reasoning-on-graphs)
and [DoG](https://github.com/leekum2018/Decoding-on-Graph).
The frozen 2WikiMultiHopQA graphs are [DoG's 2wiki_4000.jsonl at commit
de0df080](https://github.com/leekum2018/Decoding-on-Graph/blob/de0df080ab82f4dbd7615471f2ad33fed5fbf78a/data/2wiki/2wiki_4000.jsonl).

Keep the exact source revisions and any author-reviewed exclusion IDs with the
dataset. Split construction and paper population curation are separate operations.
The original populations are CWQ 3,531, WebQSP 1,628, and 2WikiMultiHopQA 4,016 rows.
The current curated CWQ paper population contains 3,138 rows (S/I: 1,679/1,459).

## S/I split rules

S contains questions whose input knowledge provides sufficient support for at
least one gold answer; I contains questions without such support. Sufficient
support jointly satisfies the evaluation constraints with consistent shared
entity or value bindings. These annotation-derived constraints are used for
evaluation, separately from Grasp's predicted support specification.

CWQ/WebQSP replay official executable query constraints with shared bindings.
Folded relation chains, declared inverses, and supported filters/order operations
retain their current semantics. WebQSP uses only complete parses. S requires at
least one credited answer with support; incomplete replay stays unresolved.

2WikiMultiHopQA requires every official evidence fact. Endpoint identities/aliases
follow the official evidence normalizer; relation and direction match exactly.
An absent fact yields I. Conflicting official surface names and IDs yield X.
The implementation preserves the frozen v08 decision rule and label fields;
unused property-schema and typed-literal experiment modes are omitted.

Stored `K` / `complete_witness` maps to S, and `U` / `no_witness` maps to I.
`X` / `unresolved` stays in raw records and is excluded from S/I paper measures.

## Output schemas

Support specification induction writes one indexed JSON artifact per question,
with `id`, `question`, `plan`, and model-generation metadata. `Plan` denotes the
complete support specification, as in the paper's prompt appendix. Invalid
specifications have `.error.json` records. Specification-guided grounding
records assignments, evidence rows, and the shared proof bank.
Packing produces `step_evidence` under the current Plan step IDs.

Final records retain `example_id`, `status`, `answers`, `reason`, `decision`, and
`step_evidence`. `status=insufficient` is the paper's Unknown outcome. The answer
generator cites `[sN.tM]` evidence IDs; the flat-evidence component uses `[eN]`.
The public evaluator reads the evidence already carried by Final records.
