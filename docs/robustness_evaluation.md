# Distribution-aware robustness evaluation

## Why this audit focuses on targets, not paraphrases

The organizer's current [Final Evaluation FAQ](https://github.com/TechJam2026/techjam-conversational-search/blob/main/docs/final_evaluation_faq.md)
states that the 800 final sessions use the same deterministic customer-message templates and
`ask_attribute` response policy as the released evaluator, with no undisclosed paraphrases. The official
[README](https://github.com/TechJam2026/techjam-conversational-search) also says that sessions are sampled
from the Amazon Reviews 2023 Clothing 5-core leave-last-out split. The main unknown is therefore the target
product cohort, not a hidden language policy.

Static inspection found no direct label lookup in the Agent: it does not read `public_set.jsonl`, sample IDs,
or ground truth. It does, however, deliberately reproduce the released metadata-to-intent-card algorithm and
the fixed simulator protocol. That is valid under the published final-evaluation contract, but it means the
reported score measures performance on this synthetic protocol rather than general conversational search.

## Public-set selection effects

The 200 public targets are not representative of a uniform draw from the 50,000-row catalog:

- roughly 73% occur in the catalog's first 1,000 high-popularity rows;
- median `rating_number` is about 6,800 for public targets and 12 for the full catalog;
- all public targets have non-empty features, details, and store metadata;
- no public target has more than 50 candidates after intersecting its coarse category and complete intent
  card, while roughly 700 catalog products do.

The existing `results_subset_40*.json` files are useful implementation checks, but they are not blind
holdouts: their sessions were included in earlier public-set experiments and policy sweeps. The older
`results.json` also predates the current Agent. The 0.95-level subset scores should therefore be treated as
development scores.

## Construction

`scripts/evaluate_robustness.py` creates two deterministic suites, each with the exact official small-sample
mix of 8 Buying, 8 Browsing, 3 Intent Override, and 1 Boundary session. All 200 public target ASINs are
excluded from synthetic ground truth, but remain in the searchable catalog as they would in final scoring.
The Agent receives only the normal `reset` and `respond` inputs.

### Matched pseudo-private suite

Twenty public sessions are selected with scenario and catalog-cohort stratification. Each target is replaced,
without reuse, by an unseen catalog product matched on:

- broad category and first-constraint type;
- high-popularity (`row < 1,000`), middle, or long-tail catalog cohort;
- `log1p(rating_number)`, average rating, price presence/value, and metadata completeness;
- metadata richness and one-, two-, and three-constraint candidate-set sizes.

The matched suite preserved broad-category and catalog-cohort membership for all 20 pairs. Its largest
absolute standardized mean difference across the numeric matching features was 0.326; most were below 0.2.
This suite is the only one intended as a small private-performance proxy.

### Collision stress suite

The second suite contains 12 ordinary matched targets and 8 targets with a complete-card candidate set above
50. This deliberately overweights a rare public-set blind spot and permutes whole user profiles within each
scenario. It is a failure-mode diagnostic, not a private-score estimate.

## Results

The experiment reused one production Agent and the existing embedding cache. It evaluated 40 new sessions in
214 seconds on CPU, including 32 seconds for target-statistics construction and index setup.

| Suite | N | HitRate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched pseudo-private | 20 | 1.000 | 1.000 | 2.400 | **0.972** |
| Collision stress (12 matched + 8 collision) | 20 | 0.700 | 0.567 | 5.700 | **0.626** |

Every matched pseudo-private target was found at rank 1. Scenario MTTC was 1.875 for Buying, 2.5 for
Browsing, 3.333 for Intent Override, and 3.0 for the single Boundary session. The conditional stratified
session-bootstrap interval for TechnicalScore was 0.967–0.976, but this interval does **not** include
uncertainty in the synthetic target-generation model; with only 20 sessions, it must not be read as a final
leaderboard confidence interval.

The eight explicitly high-collision targets exposed a sharp failure:

- HitRate@10: 0.25;
- MRR: 0.0429;
- MTTC: 10.75;
- TechnicalScore: 0.1429;
- six complete misses; the two hits occurred only on turn 10, at ranks 7 and 5.

All 12 ordinary matched targets in the same stress run were found. This isolates candidate-set collision,
rather than session ordering or Agent initialization, as the primary cause of the drop.

## Interpretation

The current 0.95-level development result is not explained by memorizing public ASINs: performance remained
strong on unseen targets that closely matched the observable purchase-driven target distribution. It is,
however, partly a protocol-specialized result. The metadata inversion is expected to transfer because the
official FAQ freezes that protocol.

The concrete robustness defect is the result-count policy for large exact-card intersections. Candidate sets
above 50 are classified as medium evidence; the Agent usually emits two new products per early turn and ten
on the last turn. It can expose at most about 28 distinct candidates, so many targets in a 51–604 item
collision group are unreachable even after all available constraints are known.

## Recommended next experiment

Before changing ranking logic, freeze `robustness_results.json` as an audit artifact. A fix should make result
count deadline-aware for medium/high-collision sets and should then be evaluated on a new seed, not tuned on
these same eight targets. Preserve the matched suite as the private proxy and report collision results as a
separate slice. Expanding the matched proxy from 20 to 40 is warranted only after a policy change or if a new
seed produces at least two misses or a TechnicalScore drop above 0.05.

Reproduce the checked-in run with:

```bash
pixi run evaluate-robustness
```

Generate only the target manifest and calibration, without loading the retrieval model, with:

```bash
pixi run python -m scripts.evaluate_robustness --skip-evaluation --output robustness_manifest.json
```
