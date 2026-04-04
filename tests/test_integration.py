"""Tests for MM-RAGBench ↔ mmeval-vrag integration.

Run with: pytest tests/ -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from mmeval_vrag import MultimodalRAGEvaluator, EvalConfig
from mmeval_vrag.types import EvalSample, RetrievedItem, ImageInput
from mmeval_vrag.evaluators.pipeline import EvalPipeline, QueryItem
from mmeval_vrag.datasets import load_dataset as mmeval_load

from mm_ragbench.loader import MMRAGBenchLoader, load_eval_samples, load_query_items
from mm_ragbench.builder import MMRAGBenchBuilder, QueryRecord, DocumentRecord


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

SAMPLE_RECORD = {
    "query_id": "test-001",
    "query": "What type of bridge is shown in the image?",
    "query_image": None,
    "domain": "geography_travel",
    "difficulty": "medium",
    "gold_doc_ids": ["wit-abc123"],
    "gold_doc_texts": ["The Golden Gate Bridge is a suspension bridge spanning the Golden Gate strait."],
    "gold_doc_images": [],
    "gold_doc_image_urls": ["https://upload.wikimedia.org/example/golden_gate.jpg"],
    "gold_doc_licenses": ["CC-BY-SA-3.0"],
    "hard_negatives": ["wit-xyz789"],
    "gold_answer": "The Golden Gate Bridge is a suspension bridge.",
    "answer_modality": "cross_modal",
    "requires_image": True,
    "requires_text": True,
    "hallucination_traps": json.dumps([
        {"type": "attribute", "description": "May confuse completion year"}
    ]),
    "faithfulness_criteria": json.dumps([
        "Must identify bridge type from visual features"
    ]),
}


@pytest.fixture
def sample_jsonl(tmp_path):
    """Create a temp JSONL file with sample data."""
    path = tmp_path / "test_data.jsonl"
    with open(path, "w") as f:
        for i in range(5):
            record = {**SAMPLE_RECORD, "query_id": f"test-{i:03d}"}
            f.write(json.dumps(record) + "\n")
    return str(path)


# ──────────────────────────────────────────────────────────
# Loader tests
# ──────────────────────────────────────────────────────────

class TestMMRAGBenchLoader:
    def test_load_jsonl_returns_eval_samples(self, sample_jsonl):
        loader = MMRAGBenchLoader()
        samples = loader.load(sample_jsonl)

        assert len(samples) == 5
        assert all(isinstance(s, EvalSample) for s in samples)

    def test_eval_sample_fields(self, sample_jsonl):
        loader = MMRAGBenchLoader()
        sample = loader.load(sample_jsonl, max_samples=1)[0]

        assert sample.query_text == "What type of bridge is shown in the image?"
        assert sample.reference_answer == "The Golden Gate Bridge is a suspension bridge."
        assert sample.sample_id == "test-000"
        assert sample.metadata["domain"] == "geography_travel"
        assert sample.metadata["difficulty"] == "medium"
        assert sample.metadata["requires_image"] is True

    def test_retrieved_items_from_gold_docs(self, sample_jsonl):
        loader = MMRAGBenchLoader()
        sample = loader.load(sample_jsonl, max_samples=1)[0]

        assert len(sample.retrieved) == 1
        item = sample.retrieved[0]
        assert isinstance(item, RetrievedItem)
        assert item.is_relevant is True
        assert item.metadata["id"] == "wit-abc123"
        assert "suspension bridge" in item.text

    def test_image_url_fallback(self, sample_jsonl):
        """When gold_doc_images is empty, loader uses gold_doc_image_urls."""
        loader = MMRAGBenchLoader()
        sample = loader.load(sample_jsonl, max_samples=1)[0]

        item = sample.retrieved[0]
        assert item.image is not None
        assert "wikimedia" in item.image.path
        assert item.metadata["image_url"] == "https://upload.wikimedia.org/example/golden_gate.jpg"

    def test_hallucination_traps_parsed(self, sample_jsonl):
        loader = MMRAGBenchLoader()
        sample = loader.load(sample_jsonl, max_samples=1)[0]

        traps = sample.metadata["hallucination_traps"]
        assert isinstance(traps, list)
        assert len(traps) == 1
        assert traps[0]["type"] == "attribute"

    def test_faithfulness_criteria_parsed(self, sample_jsonl):
        loader = MMRAGBenchLoader()
        sample = loader.load(sample_jsonl, max_samples=1)[0]

        criteria = sample.metadata["faithfulness_criteria"]
        assert isinstance(criteria, list)
        assert "bridge type" in criteria[0].lower()

    def test_load_query_items(self, sample_jsonl):
        loader = MMRAGBenchLoader()
        queries = loader.load_query_items(sample_jsonl)

        assert len(queries) == 5
        assert all(isinstance(q, QueryItem) for q in queries)

    def test_query_item_fields(self, sample_jsonl):
        loader = MMRAGBenchLoader()
        query = loader.load_query_items(sample_jsonl, max_samples=1)[0]

        assert query.query_text == "What type of bridge is shown in the image?"
        assert query.reference_answer == "The Golden Gate Bridge is a suspension bridge."
        assert "wit-abc123" in query.relevant_ids
        assert query.metadata["domain"] == "geography_travel"

    def test_max_samples_limit(self, sample_jsonl):
        loader = MMRAGBenchLoader()
        samples = loader.load(sample_jsonl, max_samples=2)
        assert len(samples) == 2

    def test_convenience_functions(self, sample_jsonl):
        samples = load_eval_samples(sample_jsonl, max_samples=3)
        assert len(samples) == 3

        queries = load_query_items(sample_jsonl, max_samples=3)
        assert len(queries) == 3


# ──────────────────────────────────────────────────────────
# Registry integration
# ──────────────────────────────────────────────────────────

class TestRegistryIntegration:
    def test_registered_in_mmeval_vrag(self):
        """mm_ragbench import should register the loader."""
        from mmeval_vrag.datasets.loaders import _LOADER_REGISTRY
        assert "mm_ragbench" in _LOADER_REGISTRY

    def test_load_via_mmeval_registry(self, sample_jsonl):
        """Load using mmeval-vrag's generic load_dataset function."""
        samples = mmeval_load("mm_ragbench", sample_jsonl, max_samples=2)
        assert len(samples) == 2
        assert isinstance(samples[0], EvalSample)


# ──────────────────────────────────────────────────────────
# Evaluator integration
# ──────────────────────────────────────────────────────────

class TestEvaluatorIntegration:
    def test_evaluate_with_multimodal_rag_evaluator(self, sample_jsonl):
        """End-to-end: load → fill answers → evaluate."""
        samples = load_eval_samples(sample_jsonl)

        # Simulate system output
        for s in samples:
            s.generated_answer = s.reference_answer or "No answer"

        evaluator = MultimodalRAGEvaluator(
            config=EvalConfig(
                metrics=["faithfulness", "hallucination_rate"],
                device="cpu",
            )
        )
        results = evaluator.evaluate(samples, show_progress=False)

        assert len(results.results) == 5
        summary = results.summary()
        assert "faithfulness" in summary
        assert "hallucination_rate" in summary

    def test_evaluate_with_pipeline(self, sample_jsonl):
        """End-to-end: load queries → run pipeline → get results."""
        queries = load_query_items(sample_jsonl, max_samples=3)

        def dummy_retriever(query_text=None, query_image=None, top_k=5):
            return [RetrievedItem(text="A bridge.", is_relevant=True, metadata={"id": "wit-abc123"})]

        def dummy_generator(query_text, contexts):
            return "The bridge is a suspension bridge."

        pipeline = EvalPipeline(
            retriever=dummy_retriever,
            generator=dummy_generator,
            config=EvalConfig(metrics=["faithfulness", "retrieval_precision"]),
        )
        results = pipeline.run(queries, show_progress=False)

        assert len(results.results) == 3
        assert all("faithfulness" in r.scores for r in results.results)


# ──────────────────────────────────────────────────────────
# Builder tests
# ──────────────────────────────────────────────────────────

class TestBuilder:
    def test_domain_classification(self):
        builder = MMRAGBenchBuilder(output_dir="/tmp/test-build")

        assert builder._classify_domain("The Golden Gate Bridge in San Francisco") == "geography_travel"
        assert builder._classify_domain("DNA molecule structure in biology") == "science_nature"
        assert builder._classify_domain("Italian pasta recipe with cooking instructions") == "food_cooking"

    def test_storage_mode_default(self):
        builder = MMRAGBenchBuilder(output_dir="/tmp/test-build")
        assert builder.storage_mode == "thumbnails"

    def test_storage_mode_urls(self):
        builder = MMRAGBenchBuilder(output_dir="/tmp/test-build", storage_mode="urls")
        assert builder.storage_mode == "urls"

    def test_verify_and_balance(self):
        builder = MMRAGBenchBuilder(output_dir="/tmp/test-build")

        # Create fake queries across domains and difficulties
        for i in range(60):
            domain = builder.DOMAINS[i % len(builder.DOMAINS)]
            difficulty = ["easy", "medium", "hard"][i % 3]
            builder.queries.append(QueryRecord(
                query_id=f"q-{i:04d}",
                query=f"Test query {i}",
                domain=domain,
                difficulty=difficulty,
                gold_doc_ids=[f"doc-{i}"],
                gold_answer=f"Answer {i}",
            ))

        builder.verify_and_balance(target=30)
        assert len(builder.queries) <= 30

        # Check balance
        domains = set(q.domain for q in builder.queries)
        assert len(domains) > 1  # multiple domains represented

    def test_export_jsonl(self, tmp_path):
        builder = MMRAGBenchBuilder(output_dir=str(tmp_path))

        builder.documents = [DocumentRecord(
            doc_id="doc-001", source="test", domain="technology",
            image_path="/tmp/test.jpg", image_url=None,
            text="A test document about circuits.",
            caption="Circuit board image",
        )]
        builder.queries = [QueryRecord(
            query_id="q-001", query="What is shown?", domain="technology",
            difficulty="easy", gold_doc_ids=["doc-001"],
            gold_answer="A circuit board.",
            hallucination_traps=[{"type": "object", "description": "May misidentify component"}],
            faithfulness_criteria=["Must identify circuit board"],
        )]

        path = builder.export_jsonl()
        assert Path(path).exists()

        # Verify the JSONL is loadable by mmeval-vrag
        samples = load_eval_samples(path)
        assert len(samples) == 1
        assert samples[0].query_text == "What is shown?"
        assert samples[0].metadata["domain"] == "technology"
