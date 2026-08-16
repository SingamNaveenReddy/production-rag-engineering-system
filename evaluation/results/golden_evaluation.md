# Golden RAG evaluation

Profile: `deterministic`  
Dataset: `evaluation/golden_dataset.jsonl`  
Questions evaluated: 6  
Faithfulness method: `lexical_citation_token_coverage`

## Metrics

| Metric | Result |
|---|---:|
| Retrieval recall@5 | 1.000 |
| Faithfulness | 1.000 |
| Citation accuracy | 1.000 |
| Refusal accuracy | 1.000 |
| Mean latency | 0.0 ms |
| Median latency | 0.0 ms |
| P95 latency | 0 ms |

## Quality gate: PASS

- None

## Interpretation

The deterministic profile exercises the real ingestion, hybrid retrieval, answerability, and
citation-validation service with local deterministic providers. Its lexical faithfulness score is
a CI-safe proxy based on answer-token coverage in validated citation text, not an LLM-judged Ragas
score. Inject an explicit LLM-based scorer for release-quality production assessment.
