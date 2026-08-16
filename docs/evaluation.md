# Evaluation

Phase 5 implements the golden-dataset runner and configurable regression gate. The seed dataset has
six manually verified cases: four answerable policy questions and two unsupported questions. It is
an executable framework seed, not the final 100-question production corpus.

Run the deterministic profile:

```bash
python -m evaluation.evaluate
```

Run configured production providers after indexing the matching corpus:

```bash
python -m evaluation.evaluate --profile production
```

Both profiles generate `evaluation/results/golden_evaluation.json` and
`evaluation/results/golden_evaluation.md`. A failed threshold exits non-zero.

## Metrics

- Retrieval recall@k compares expected sources with first-stage retrieved sources.
- Deterministic faithfulness measures answer-token coverage in validated citation text.
- Citation accuracy compares returned citations with manually verified expected sources.
- Refusal accuracy checks the structured answerability decision.
- Latency reports mean, median, and P95 total query latency.

The faithfulness scorer is injectable. The deterministic lexical scorer is suitable for a small,
repeatable CI gate; release evaluation should use an explicit LLM-based scorer such as Ragas and
record that method in the report.

Install the optional Ragas adapter dependency with `pip install -e '.[evaluation]'`, construct a
Ragas collections `Faithfulness` metric with an explicit evaluator LLM, and inject it through
`RagasFaithfulnessScorer`. No evaluator provider or credentials are selected implicitly.

## Executed result

The latest deterministic run evaluated six questions and passed the configured gate. Retrieval
recall@5, lexical citation faithfulness, citation accuracy, and refusal accuracy were all 1.000.
P95 latency was 0 ms. These results apply only to the checked-in deterministic seed profile; they
are not presented as production-model quality.

Earlier executed retrieval and reranking fixtures remain in `evaluation/results/phase2_retrieval.*`
and `evaluation/results/phase3_reranking.*`.
