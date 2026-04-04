"""MM-RAGBench dataset loader for mmeval-vrag.

Registers ``"mm_ragbench"`` in mmeval-vrag's DatasetLoader registry so
users can evaluate with:

    from mmeval_vrag.datasets import load_dataset
    samples = load_dataset("mm_ragbench", "EmmanuelleB985/mm-ragbench", split="test")

Or use the pipeline evaluator with QueryItem:

    from mm_ragbench import load_query_items
    queries = load_query_items("EmmanuelleB985/mm-ragbench", split="test")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mmeval_vrag.types import EvalSample, ImageInput, RetrievedItem
from mmeval_vrag.evaluators.pipeline import QueryItem
from mmeval_vrag.datasets.loaders import DatasetLoader, _LOADER_REGISTRY


# ──────────────────────────────────────────────────────────
# HuggingFace Hub loader
# ──────────────────────────────────────────────────────────

class MMRAGBenchLoader(DatasetLoader):
    """Load MM-RAGBench from HuggingFace Hub or local files.

    Converts each benchmark sample into an :class:`EvalSample` with:
    - query_text / query_image from the benchmark query
    - retrieved items from gold_documents (with is_relevant=True)
    - reference_answer from gold_answer
    - metadata with domain, difficulty, hallucination_traps, faithfulness_criteria
    """

    name = "mm_ragbench"

    def load(
        self,
        path: Union[str, Path] = "EmmanuelleB985/mm-ragbench",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> List[EvalSample]:
        """Load benchmark samples as EvalSample objects.

        Parameters
        ----------
        path : str or Path
            HuggingFace dataset ID (e.g. ``"EmmanuelleB985/mm-ragbench"``)
            or path to a local JSONL file.
        split : str
            Dataset split (``"test"`` or ``"dev"``).
        max_samples : int or None
            Cap on samples to load.
        """
        path_str = str(path)

        if path_str.endswith(".jsonl"):
            raw_samples = self._load_jsonl(path_str, max_samples)
        else:
            raw_samples = self._load_hub(path_str, split, max_samples)

        return [self._to_eval_sample(s) for s in raw_samples]

    def load_query_items(
        self,
        path: Union[str, Path] = "EmmanuelleB985/mm-ragbench",
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> List[QueryItem]:
        """Load benchmark samples as QueryItem objects for EvalPipeline.

        Use this when evaluating a live retriever + generator pipeline:

            pipeline = EvalPipeline(retriever=my_ret, generator=my_gen)
            results = pipeline.run(loader.load_query_items())
        """
        path_str = str(path)

        if path_str.endswith(".jsonl"):
            raw_samples = self._load_jsonl(path_str, max_samples)
        else:
            raw_samples = self._load_hub(path_str, split, max_samples)

        return [self._to_query_item(s) for s in raw_samples]

    # ── internal helpers ──────────────────────────────────

    @staticmethod
    def _load_hub(repo_id: str, split: str, max_samples: Optional[int]) -> list[dict]:
        from datasets import load_dataset as hf_load
        ds = hf_load(repo_id, split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        return list(ds)

    @staticmethod
    def _load_jsonl(path: str, max_samples: Optional[int]) -> list[dict]:
        records = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _to_eval_sample(raw: dict) -> EvalSample:
        """Convert a raw benchmark record into an mmeval_vrag EvalSample."""
        # Build retrieved items from gold documents
        retrieved = []
        gold_doc_ids = raw.get("gold_doc_ids", [])
        gold_doc_texts = raw.get("gold_doc_texts", [])
        gold_doc_images = raw.get("gold_doc_images", [])
        gold_doc_image_urls = raw.get("gold_doc_image_urls", [])
        gold_doc_licenses = raw.get("gold_doc_licenses", [])

        for idx in range(len(gold_doc_ids)):
            text = gold_doc_texts[idx] if idx < len(gold_doc_texts) else None

            # Prefer local path, fall back to URL
            img_path = gold_doc_images[idx] if idx < len(gold_doc_images) else ""
            img_url = gold_doc_image_urls[idx] if idx < len(gold_doc_image_urls) else ""
            license_str = gold_doc_licenses[idx] if idx < len(gold_doc_licenses) else ""

            image_input = None
            if img_path:
                image_input = ImageInput(path=img_path)
            elif img_url:
                # ImageInput.path accepts URLs — mmeval-vrag loads lazily
                image_input = ImageInput(path=img_url)

            retrieved.append(RetrievedItem(
                text=text,
                image=image_input,
                score=1.0,
                metadata={
                    "id": gold_doc_ids[idx],
                    "license": license_str,
                    "image_url": img_url,
                    "is_gold": True,
                },
                is_relevant=True,
            ))

        # Parse JSON-encoded annotation fields
        hallucination_traps = raw.get("hallucination_traps", "[]")
        if isinstance(hallucination_traps, str):
            hallucination_traps = json.loads(hallucination_traps)

        faithfulness_criteria = raw.get("faithfulness_criteria", "[]")
        if isinstance(faithfulness_criteria, str):
            faithfulness_criteria = json.loads(faithfulness_criteria)

        return EvalSample(
            query_text=raw.get("query", raw.get("query_text")),
            query_image=(
                ImageInput(path=raw["query_image"])
                if raw.get("query_image") else None
            ),
            retrieved=retrieved,
            generated_answer="",  # to be filled by the system under test
            reference_answer=raw.get("gold_answer", raw.get("reference_answer")),
            sample_id=raw.get("query_id", raw.get("id", "")),
            metadata={
                "domain": raw.get("domain", ""),
                "difficulty": raw.get("difficulty", ""),
                "answer_modality": raw.get("answer_modality", ""),
                "requires_image": raw.get("requires_image", True),
                "requires_text": raw.get("requires_text", True),
                "hallucination_traps": hallucination_traps,
                "faithfulness_criteria": faithfulness_criteria,
                "hard_negatives": raw.get("hard_negatives", []),
            },
        )

    @staticmethod
    def _to_query_item(raw: dict) -> QueryItem:
        """Convert a raw benchmark record into an mmeval_vrag QueryItem."""
        hallucination_traps = raw.get("hallucination_traps", "[]")
        if isinstance(hallucination_traps, str):
            hallucination_traps = json.loads(hallucination_traps)

        faithfulness_criteria = raw.get("faithfulness_criteria", "[]")
        if isinstance(faithfulness_criteria, str):
            faithfulness_criteria = json.loads(faithfulness_criteria)

        return QueryItem(
            query_text=raw.get("query", raw.get("query_text")),
            query_image=(
                ImageInput(path=raw["query_image"])
                if raw.get("query_image") else None
            ),
            reference_answer=raw.get("gold_answer", raw.get("reference_answer")),
            relevant_ids=raw.get("gold_doc_ids", []),
            metadata={
                "domain": raw.get("domain", ""),
                "difficulty": raw.get("difficulty", ""),
                "answer_modality": raw.get("answer_modality", ""),
                "requires_image": raw.get("requires_image", True),
                "requires_text": raw.get("requires_text", True),
                "hallucination_traps": hallucination_traps,
                "faithfulness_criteria": faithfulness_criteria,
            },
        )


# ──────────────────────────────────────────────────────────
# Register in mmeval-vrag's loader registry
# ──────────────────────────────────────────────────────────

_LOADER_REGISTRY["mm_ragbench"] = MMRAGBenchLoader


# ──────────────────────────────────────────────────────────
# Public convenience functions
# ──────────────────────────────────────────────────────────

_default_loader = MMRAGBenchLoader()


def load_eval_samples(
    path: str = "EmmanuelleB985/mm-ragbench",
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[EvalSample]:
    """Load MM-RAGBench as a list of EvalSample (for MultimodalRAGEvaluator).

    >>> from mm_ragbench import load_eval_samples
    >>> samples = load_eval_samples(split="test")
    """
    return _default_loader.load(path, split, max_samples)


def load_query_items(
    path: str = "EmmanuelleB985/mm-ragbench",
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[QueryItem]:
    """Load MM-RAGBench as a list of QueryItem (for EvalPipeline).

    >>> from mm_ragbench import load_query_items
    >>> queries = load_query_items(split="test")
    >>> pipeline = EvalPipeline(retriever=my_ret, generator=my_gen)
    >>> results = pipeline.run(queries)
    """
    return _default_loader.load_query_items(path, split, max_samples)
