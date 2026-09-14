# Architecture

The project implements a reproducible retrieve-rerank-answer pipeline for rail-transit
regulations, maintenance guidance and accident reports.

```text
source_manifest.yaml
        |
        v
license-aware downloader ----> data/raw/ (local only)
        |
        v
PDF / HTML / ZIP parser -----> documents.jsonl -----> chunks.jsonl
                                                       |
                              +------------------------+------------------+
                              |                                           |
                         BM25 retrieval                         multilingual E5
                              |                                           |
                              +---------------- RRF ----------------------+
                                                   |
                                      multilingual Cross-Encoder
                                                   |
                                  cited evidence + optional LLM answer
```

## Retrieval stages

1. **BM25** uses language-aware tokenization: Jieba search-mode tokens for Chinese and
   normalized alphanumeric tokens for English and equipment identifiers.
2. **Dense retrieval** uses `intfloat/multilingual-e5-small`, with the model's `query:` and
   `passage:` prefixes and normalized cosine-equivalent dot products.
3. **RRF** fuses BM25 and dense ranks without requiring score calibration.
4. **Reranking** applies `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` to the top hybrid
   candidates. A query-aware window prevents the answer span of long Chinese chunks from
   being silently removed by the model's 512-token limit.
5. **Answering** supplies numbered evidence to the OpenAI Responses API when an API key is
   configured. A deterministic extractive fallback keeps the demo runnable without a key.

Model weights and embeddings are stored below `data/cache/` and are never included in Git or
the release archive.

The default online path is RRF hybrid retrieval. On the current CPU evaluation it has the
best nDCG/latency trade-off; Cross-Encoder reranking remains available as a measured optional
stage rather than being assumed to improve every corpus.

## Evaluation protocol

The reviewed set contains Chinese and English questions mapped to evidence terms, source IDs,
document IDs, chunk IDs and PDF pages or web sections. The resolver rebuilds those mappings
after corpus changes. All four retrieval stages are evaluated under one corpus and one query
set using Recall@K, Hit@K, MRR@10, nDCG@10 and end-to-end P95 latency. One-time model loading
and corpus embedding construction are reported separately from query latency.

## Compliance boundary

Restricted source files and full derived text stay in ignored local directories. The release
archive contains source code, the manifest, questions, aggregate metrics and a small demo made
only from the EPL-2.0 OpenRail specification.
