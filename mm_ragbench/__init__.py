"""MM-RAGBench — Companion benchmark dataset for mmeval-vrag.

    pip install mm-ragbench

Quick start
-----------
>>> from mm_ragbench import load_eval_samples, load_query_items
>>> from mmeval_vrag import MultimodalRAGEvaluator, EvalConfig

>>> # Option 1: evaluate pre-computed answers
>>> samples = load_eval_samples(split="test", max_samples=100)
>>> evaluator = MultimodalRAGEvaluator(config=EvalConfig(metrics=["all"]))
>>> results = evaluator.evaluate(samples)

>>> # Option 2: evaluate a live pipeline
>>> from mmeval_vrag.evaluators.pipeline import EvalPipeline
>>> queries = load_query_items(split="test")
>>> pipeline = EvalPipeline(retriever=my_retriever, generator=my_generator)
>>> results = pipeline.run(queries)
"""

from mm_ragbench.loader import (
    MMRAGBenchLoader,
    load_eval_samples,
    load_query_items,
)
from mm_ragbench.builder import MMRAGBenchBuilder

__version__ = "0.1.0"
__all__ = [
    "MMRAGBenchLoader",
    "load_eval_samples",
    "load_query_items",
    "MMRAGBenchBuilder",
]
