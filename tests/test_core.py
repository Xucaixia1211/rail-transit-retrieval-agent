from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from rail_agent.answering import answer_question
from rail_agent.metrics import hit_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from rail_agent.retrieval import BM25Retriever, query_aware_excerpt, reciprocal_rank_fusion
from rail_agent.tokenize import tokenize


class CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "source_id": "s1",
                "title": "轨道检修规程",
                "text": "钢轨出现裂纹时应立即检查并采取限速措施。",
                "page_start": 3,
                "page_end": 3,
                "section_path": ["钢轨检查"],
                "source_page": "https://example.invalid/source",
                "repository_policy": "manifest_only",
                "license": {"status": "test"},
            },
            {
                "chunk_id": "c2",
                "document_id": "d2",
                "source_id": "s2",
                "title": "应急演练办法",
                "text": "运营单位每半年至少组织一次综合应急预案实战演练。",
                "page_start": None,
                "page_end": None,
                "section_path": ["演练频率"],
                "source_page": "https://example.invalid/drill",
                "repository_policy": "manifest_only",
                "license": {"status": "test"},
            },
            {
                "chunk_id": "c3",
                "document_id": "d3",
                "source_id": "s3",
                "title": "车站客运组织办法",
                "text": "车站应根据客流变化及时调整进出站组织措施。",
                "page_start": 8,
                "page_end": 8,
                "section_path": ["客流组织"],
                "source_page": "https://example.invalid/station",
                "repository_policy": "manifest_only",
                "license": {"status": "test"},
            },
        ]

    def test_mixed_language_tokenizer(self) -> None:
        tokens = tokenize("轨道 Track-Circuit B2-304")
        self.assertIn("轨道", tokens)
        self.assertIn("track-circuit", tokens)
        self.assertIn("b2-304", tokens)

    def test_bm25_returns_relevant_chunk_first(self) -> None:
        results = BM25Retriever(self.chunks).search("综合应急预案多久演练一次", 3)
        self.assertEqual(results[0]["chunk_id"], "c2")

    def test_rrf_combines_rankings(self) -> None:
        lookup = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        first = [
            {"chunk_id": "c1", "stage": "bm25"},
            {"chunk_id": "c2", "stage": "bm25"},
        ]
        second = [
            {"chunk_id": "c2", "stage": "dense"},
            {"chunk_id": "c1", "stage": "dense"},
        ]
        results = reciprocal_rank_fusion([first, second], lookup, rrf_k=60, top_k=2)
        self.assertEqual({item["chunk_id"] for item in results}, {"c1", "c2"})
        self.assertTrue(all("component_ranks" in item for item in results))

    def test_query_aware_excerpt_keeps_late_evidence(self) -> None:
        text = "无关说明。" * 100 + "检修记录应保存至设备使用寿命终止。" + "附则。" * 100
        excerpt = query_aware_excerpt("检修记录保存多久", text, max_characters=120)
        self.assertIn("检修记录应保存", excerpt)

    def test_metrics(self) -> None:
        ranked = ["x", "c1", "c2"]
        relevant = {"c1", "c2"}
        self.assertEqual(hit_at_k(ranked, relevant, 1), 0.0)
        self.assertEqual(hit_at_k(ranked, relevant, 2), 1.0)
        self.assertEqual(recall_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(reciprocal_rank(ranked, relevant, 3), 0.5)
        self.assertGreater(ndcg_at_k(ranked, relevant, 3), 0.0)

    def test_answer_has_traceable_sources_without_api_key(self) -> None:
        result = {**self.chunks[1], "rank": 1, "score": 1.0, "stage": "bm25"}
        answer = answer_question(
            "综合应急预案多久演练一次？",
            [result],
            {"max_evidence_chunks": 3, "max_evidence_characters": 2000, "default_model": "unused"},
            llm_mode="never",
        )
        self.assertEqual(answer["generator"], "extractive-fallback")
        self.assertIn("[S1]", answer["answer"])
        self.assertEqual(answer["sources"][0]["locator"], "演练频率")
        self.assertIn("每半年", answer["sources"][0]["evidence"])

    def test_openai_responses_path_uses_selected_model(self) -> None:
        result = {**self.chunks[1], "rank": 1, "score": 1.0, "stage": "bm25"}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch("openai.OpenAI") as client:
            client.return_value.responses.create.return_value.output_text = "每半年一次。[S1]"
            answer = answer_question(
                "综合应急预案多久演练一次？",
                [result],
                {"max_evidence_chunks": 3, "max_evidence_characters": 2000, "default_model": "fallback"},
                llm_mode="required",
                model="test-model",
            )
        self.assertEqual(answer["generator"], "openai:test-model")
        self.assertEqual(answer["answer"], "每半年一次。[S1]")
        self.assertEqual(client.return_value.responses.create.call_args.kwargs["model"], "test-model")


class EvaluationDataTests(unittest.TestCase):
    def test_reviewed_query_set_integrity(self) -> None:
        project = Path(__file__).resolve().parents[1]
        query_path = project / "evaluation" / "queries.jsonl"
        if not query_path.exists():
            self.skipTest("evaluation/queries.jsonl has not been built")
        records = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines()]
        self.assertGreaterEqual(len(records), 30)
        self.assertLessEqual(len(records), 50)
        self.assertEqual(len(records), len({record["query_id"] for record in records}))
        self.assertTrue(all(record["review_status"] == "verified_against_local_source" for record in records))
        self.assertTrue(all(record["relevant_chunk_ids"] for record in records))


if __name__ == "__main__":
    unittest.main()
