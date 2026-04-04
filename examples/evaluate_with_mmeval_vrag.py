"""
MM-RAGBench + mmeval-vrag: Complete evaluation examples.

Shows both evaluation modes:
  1. Pre-computed answers (MultimodalRAGEvaluator)
  2. Live pipeline (EvalPipeline)
"""

from mmeval_vrag import MultimodalRAGEvaluator, EvalConfig
from mmeval_vrag.evaluators.pipeline import EvalPipeline
from mmeval_vrag.types import EvalSample, RetrievedItem, ImageInput

# ──────────────────────────────────────────────────────────
# MODE 1: Evaluate pre-computed answers
# ──────────────────────────────────────────────────────────

def example_precomputed():
    """Load benchmark, fill in your system's answers, evaluate."""
    from mm_ragbench import load_eval_samples

    # Load 50 samples for a quick test
    samples = load_eval_samples(
        path="EmmanuelleB985/mm-ragbench",
        split="test",
        max_samples=50,
    )

    # Simulate your system generating answers
    for sample in samples:
        # Your system would normally do:
        #   retrieved = your_retriever(sample.query_text, sample.query_image)
        #   answer = your_generator(sample.query_text, retrieved)
        sample.generated_answer = f"Based on the documents, {sample.reference_answer}"

    # Evaluate with all 11 mmeval-vrag metrics
    evaluator = MultimodalRAGEvaluator(
        config=EvalConfig(
            metrics=["faithfulness", "hallucination_rate", "retrieval_precision",
                     "answer_relevance", "cross_modal_alignment"],
            device="cpu",
        )
    )
    results = evaluator.evaluate(samples)

    # Print aggregate results
    print(results)  # mean ± std for each metric
    print()

    # Export
    results.to_json("results_precomputed.json")

    # Per-domain breakdown (using metadata from MM-RAGBench)
    by_domain = {}
    for res, sample in zip(results.results, samples):
        domain = sample.metadata.get("domain", "unknown")
        by_domain.setdefault(domain, []).append(res.scores)

    print("Per-domain faithfulness:")
    for domain, scores_list in sorted(by_domain.items()):
        faith_scores = [s.get("faithfulness", 0) for s in scores_list]
        print(f"  {domain}: {sum(faith_scores)/len(faith_scores):.3f}")


# ──────────────────────────────────────────────────────────
# MODE 2: Evaluate a live RAG pipeline end-to-end
# ──────────────────────────────────────────────────────────

def example_pipeline():
    """Plug in your retriever + generator, let mmeval-vrag run everything."""
    from mm_ragbench import load_query_items

    queries = load_query_items(
        path="EmmanuelleB985/mm-ragbench",
        split="dev",
        max_samples=20,
    )

    # Define your retriever (replace with your real implementation)
    def my_retriever(query_text=None, query_image=None, top_k=5):
        """Example: return dummy retrieved items."""
        return [
            RetrievedItem(
                text=f"Relevant passage for: {query_text[:50]}",
                is_relevant=True,
                metadata={"id": "doc-001"},
            )
        ]

    # Define your generator (replace with your real implementation)
    def my_generator(query_text, contexts):
        """Example: simple concatenation."""
        ctx = " ".join(c.text or "" for c in contexts[:3])
        return f"Based on the context: {ctx[:200]}"

    # Create the pipeline evaluator
    pipeline = EvalPipeline(
        retriever=my_retriever,
        generator=my_generator,
        config=EvalConfig(
            metrics=["faithfulness", "hallucination_rate", "retrieval_precision",
                     "retrieval_recall", "answer_relevance"],
            top_k=5,
        ),
    )

    # Run end-to-end
    results = pipeline.run(queries)

    # Display results
    print(results)
    print()

    # Detailed per-sample analysis
    df = results.to_dataframe()
    print(df.describe())


# ──────────────────────────────────────────────────────────
# MODE 3: Build your own benchmark from scratch
# ──────────────────────────────────────────────────────────

def example_build():
    """Build the benchmark dataset from WIT + COCO."""
    from mm_ragbench import MMRAGBenchBuilder

    builder = MMRAGBenchBuilder(
        output_dir="data/mm-ragbench",
        llm_provider="anthropic",         # or "openai"
        llm_model="claude-sonnet-4-20250514",  # or "gpt-4o"
    )

    # Step 1: Collect ~12K image-text documents
    builder.collect_sources(wit_samples=9000, coco_samples=3000)

    # Step 2: Generate 3 queries per document
    builder.generate_queries(max_docs=1000)  # start small, scale up

    # Step 3: Assign hard negatives (same-domain distractors)
    builder.generate_hard_negatives()

    # Step 4: Balance across domains and difficulties
    builder.verify_and_balance(target=3000)

    # Step 5: Export as JSONL (loadable by mmeval-vrag)
    jsonl_path = builder.export_jsonl()

    # Step 6: Push to HuggingFace Hub
    # builder.push_to_hub("EmmanuelleB985/mm-ragbench", jsonl_path)

    # Now evaluate with it
    from mm_ragbench import load_eval_samples
    samples = load_eval_samples(jsonl_path, max_samples=10)
    print(f"Loaded {len(samples)} samples from {jsonl_path}")
    print(f"  First query: {samples[0].query_text}")
    print(f"  Domain: {samples[0].metadata['domain']}")
    print(f"  Difficulty: {samples[0].metadata['difficulty']}")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "precomputed"

    if mode == "precomputed":
        example_precomputed()
    elif mode == "pipeline":
        example_pipeline()
    elif mode == "build":
        example_build()
    else:
        print(f"Usage: python {sys.argv[0]} [precomputed|pipeline|build]")
