# Evaluation

`question_specs.yaml` contains 44 manually curated questions with short reference answers,
source IDs, evidence terms and optional page hints. `build_evaluation_set.py` resolves the
specifications against the current processed corpus and refuses to bind a question unless all
evidence terms occur in the selected chunk. The generated `queries.jsonl` records the source,
document, chunk, page or web section, and `verified_against_local_source` status.

## Final local result

Environment: Apple Silicon CPU, Python 3.9.13; 949 chunks from 16 enabled sources; 44 queries
(30 Chinese and 14 English). One-time model loading and corpus encoding are excluded from
per-query latency.

| Mode | Recall@5 | Recall@20 | MRR@10 | nDCG@10 | P95 latency |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.966 | 0.977 | 0.835 | 0.868 | 1.46 ms |
| Dense E5 | 0.920 | 1.000 | 0.755 | 0.798 | 18.09 ms |
| BM25 + E5 + RRF | 0.966 | 1.000 | 0.875 | **0.901** | **18.44 ms** |
| RRF + Cross-Encoder | 0.960 | 1.000 | **0.876** | 0.891 | 1473.14 ms |

The reranker slightly improves first-relevant-result rank but reduces nDCG and adds substantial
CPU latency. The project therefore defaults to RRF hybrid retrieval and exposes reranking as an
optional experiment. These figures describe this small curated set and should not be presented
as a general benchmark.

Rebuild and rerun:

```bash
./scripts/build_evaluation_set.py
./scripts/rail_agent.sh evaluate --modes bm25,dense,hybrid,rerank
```

Before using the set for a public claim, a domain practitioner should independently review the
question wording, reference answers and locators. Programmatic evidence matching detects source
drift but is not a substitute for domain-expert adjudication.
