# MM-RAGBench

[![PyPI](https://img.shields.io/badge/PyPI-mm--ragbench-blue.svg)](https://pypi.org/project/mm-ragbench/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![mmeval-vrag](https://img.shields.io/badge/powered%20by-mmeval--vrag-orange.svg)](https://github.com/EmmanuelleB985/mmeval-vrag)

**MM-RAGBench** is a benchmark dataset for evaluating **multimodal** Retrieval-Augmented Generation systems. It is built as a native companion to [mmeval-vrag](https://github.com/EmmanuelleB985/mmeval-vrag). Install, load, and evaluate in 5 lines.

## The Problem

Existing RAG benchmarks (RAGBench, ViDoRe, NoMIRACL) evaluate text-only retrieval and generation. Real-world RAG systems retrieve **images alongside text**: medical scans with clinical notes, product photos with descriptions, diagrams with documentation. None of the existing benchmarks test whether a system can reason across both modalities and avoid hallucinating visual details.

## What MM-RAGBench Provides

- **3,000 multimodal queries** across 6 domains requiring cross-modal reasoning
- **12,000 candidate documents** (image-text pairs) from open-license sources
- **Fine-grained annotations**: hallucination traps (object/attribute/relation/fabrication), faithfulness criteria
- **Native mmeval-vrag integration**: loads directly as `EvalSample` or `QueryItem`
- **All 11 mmeval-vrag metrics** work out of the box

## Install

```bash
pip install mm-ragbench
```

This pulls in `mmeval-vrag` automatically.

## Quick Start

### Option 1: Evaluate pre-computed answers

```python
from mm_ragbench import load_eval_samples
from mmeval_vrag import MultimodalRAGEvaluator, EvalConfig

# Load benchmark as mmeval-vrag EvalSample objects
samples = load_eval_samples(split="test", max_samples=100)

# Your system fills in generated_answer for each sample
for sample in samples:
    docs = your_retriever(sample.query_text, sample.query_image)
    sample.retrieved = docs
    sample.generated_answer = your_generator(sample.query_text, docs)

# Evaluate with mmeval-vrag (all 11 metrics)
evaluator = MultimodalRAGEvaluator(config=EvalConfig(metrics=["all"]))
results = evaluator.evaluate(samples)
print(results.summary())
```

### Option 2: Evaluate a live pipeline

```python
from mm_ragbench import load_query_items
from mmeval_vrag.evaluators.pipeline import EvalPipeline
from mmeval_vrag import EvalConfig

# Load benchmark as mmeval-vrag QueryItem objects
queries = load_query_items(split="test")

# Plug in your retriever + generator
pipeline = EvalPipeline(
    retriever=my_retriever,   # (query_text, query_image, top_k) → List[RetrievedItem]
    generator=my_generator,   # (query_text, contexts) → str
    config=EvalConfig(metrics=["all"]),
)
results = pipeline.run(queries)
results.to_json("my_system_results.json")
```

### Option 3: Use via mmeval-vrag's loader registry

```python
from mmeval_vrag.datasets import load_dataset
import mm_ragbench  # registers the "mm_ragbench" loader

samples = load_dataset("mm_ragbench", "EmmanuelleB985/mm-ragbench", split="test")
```

## Metrics (via mmeval-vrag)

All 11 metrics from mmeval-vrag are supported:

| Category | Metric | What it measures |
|----------|--------|------------------|
| Retrieval | `retrieval_precision` | Fraction of top-K items that are relevant |
| | `retrieval_recall` | Fraction of relevant items found in top-K |
| | `retrieval_mrr` | Reciprocal rank of first relevant item |
| | `retrieval_ndcg` | Normalised DCG accounting for rank |
| Generation | `faithfulness` | Are generated claims supported by context? |
| | `hallucination_rate` | Fraction of unsupported claims (lower = better) |
| | `answer_relevance` | Similarity between answer and query |
| | `context_relevance` | Relevance of retrieved passages to query |
| Cross-Modal | `cross_modal_alignment` | CLIP similarity: images ↔ query |
| | `visual_grounding` | CLIP similarity: images ↔ answer |
| | `multimodal_consistency` | CLIP similarity within (image, text) pairs |

## Dataset Schema

Each sample maps to mmeval-vrag types:

```
MM-RAGBench field          -> mmeval-vrag type
─────────────────────────────────────────────
query / query_image        -> EvalSample.query_text / query_image
gold_doc_texts/images      -> EvalSample.retrieved (List[RetrievedItem])
gold_answer                -> EvalSample.reference_answer
domain, difficulty, ...    -> EvalSample.metadata
hallucination_traps        -> EvalSample.metadata["hallucination_traps"]
faithfulness_criteria      -> EvalSample.metadata["faithfulness_criteria"]
gold_doc_ids               -> QueryItem.relevant_ids (for EvalPipeline)
```

### Annotation Fields

Each query includes:

- **hallucination_traps**: Known failure modes, e.g. `{"type": "attribute", "description": "May confuse bridge completion year (1937) with construction start (1933)"}`
- **faithfulness_criteria**: Verifiable checks, e.g. `"Must identify bridge type from visual features"`
- **answer_modality**: Whether the answer needs `text_only`, `image_only`, or `cross_modal` reasoning
- **difficulty**: `easy` (40%), `medium` (35%), `hard` (25%)

### Domains

| Domain | Queries | Sources |
|--------|:-------:|---------|
| Science & Nature | 500 | Wikipedia diagrams, species photos, experiments |
| Geography & Travel | 500 | Landmarks, maps, cultural sites |
| History & Art | 500 | Historical photos, artworks, architecture |
| Technology | 500 | Product images, diagrams, interfaces |
| Food & Cooking | 500 | Recipe images, ingredients, techniques |
| Daily Life | 500 | Everyday objects, how-to guides, sports |

## Build the Dataset from Scratch

```python
from mm_ragbench import MMRAGBenchBuilder

builder = MMRAGBenchBuilder(
    output_dir="data/mm-ragbench",
    storage_mode="urls",                   # "urls" (~10 MB) | "thumbnails" (~300 MB) | "full" (~50 GB)
    llm_provider="anthropic",              # or "openai"
    llm_model="claude-sonnet-4-20250514",  # or "gpt-4o"
)

builder.collect_sources()          # Pull from WIT + COCO (CC-BY/CC-BY-SA)
builder.generate_queries()         # LLM generates queries + annotations
builder.generate_hard_negatives()  # Same-domain distractors
builder.verify_and_balance()       # Balance to 3,000 queries
builder.export_jsonl()             # JSONL compatible with mmeval-vrag
builder.push_to_hub("EmmanuelleB985/mm-ragbench")
```

### Storage Modes

| Mode | Disk | What's saved | Image access |
|------|:----:|--------------|--------------|
| `"urls"` | ~10 MB | Text + Wikimedia/COCO image URLs | Fetched on demand during eval |
| `"thumbnails"` | ~300 MB | 224px JPEG thumbnails | Local files |
| `"full"` | ~50 GB | Original resolution images | Local files |

The default is `"thumbnails"` — good balance of quality and size. Use `"urls"` if disk is tight; images are fetched lazily when mmeval-vrag's CLIP metrics need them.

## Per-Domain Analysis

MM-RAGBench metadata enables granular analysis:

```python
from mm_ragbench import load_eval_samples

samples = load_eval_samples(split="test")

# Group results by domain
by_domain = {}
for sample, result in zip(samples, results.results):
    domain = sample.metadata["domain"]
    by_domain.setdefault(domain, []).append(result.scores)

for domain, scores in sorted(by_domain.items()):
    faith = [s["faithfulness"] for s in scores if "faithfulness" in s]
    halluc = [s["hallucination_rate"] for s in scores if "hallucination_rate" in s]
    print(f"{domain}: faithfulness={sum(faith)/len(faith):.3f}  hallucination={sum(halluc)/len(halluc):.3f}")
```

## Leaderboard

| System | Retrieval R@5 | Faithfulness | Hallucination ↓ | Cross-Modal | Overall |
|--------|:---:|:---:|:---:|:---:|:---:|
| *Submit yours* | — | — | — | — | — |

Submit by opening a PR with your `results.json` (exported via `results.to_json()`).

## Data Sources & Licensing

| Source | License | Used for |
|--------|---------|----------|
| Wikipedia (via WIT) | CC-BY-SA 3.0 | Article text + images |
| Wikimedia Commons | CC-BY / CC-BY-SA | Images |
| COCO | CC-BY 4.0 | Everyday scene images |
| Generated annotations | CC-BY 4.0 | Queries, answers, traps |

## Citation

```bibtex
@software{bourigault2026mmragbench,
  author = {Bourigault, Emmanuelle},
  title = {MM-RAGBench: Multimodal Benchmark for Evaluating RAG Systems},
  year = {2026},
  url = {https://github.com/EmmanuelleB985/mm-ragbench},
}

@software{bourigault2025mmeval,
  author = {Bourigault, Emmanuelle},
  title = {mmeval-vrag: Evaluation Framework for Multimodal Vision-Language RAG Systems},
  year = {2025},
  url = {https://github.com/EmmanuelleB985/mmeval-vrag},
}
```

## License

Code: Apache 2.0 . Dataset: CC-BY 4.0 . Source images: per-sample (all CC-BY or CC-BY-SA)
