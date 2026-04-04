---
license: cc-by-4.0
task_categories:
  - question-answering
  - visual-question-answering
  - text-retrieval
  - image-text-retrieval
language:
  - en
tags:
  - rag
  - retrieval-augmented-generation
  - multimodal
  - evaluation
  - benchmark
  - hallucination
  - faithfulness
  - vision-language
  - cross-modal
  - mmeval-vrag
size_categories:
  - 1K<n<10K
pretty_name: "MM-RAGBench"
---

# MM-RAGBench: Multimodal RAG Evaluation Benchmark

Companion benchmark dataset for [mmeval-vrag](https://github.com/EmmanuelleB985/mmeval-vrag). 3,000 queries with cross-modal reasoning requirements, hallucination traps, and faithfulness criteria.

## Use with mmeval-vrag

```python
pip install mm-ragbench  # pulls mmeval-vrag automatically
```

```python
from mm_ragbench import load_eval_samples, load_query_items
from mmeval_vrag import MultimodalRAGEvaluator, EvalConfig

# Load as EvalSample objects
samples = load_eval_samples(split="test")

# Or as QueryItem objects for pipeline evaluation
queries = load_query_items(split="test")
```

## What's Inside

- **3,000 queries** across 6 domains (science, geography, history, tech, food, daily life)
- **Difficulty levels**: easy (40%), medium (35%), hard (25%)
- **Hallucination traps**: annotated failure modes per query (object/attribute/relation/fabrication)
- **Faithfulness criteria**: verifiable checks per query
- **Cross-modal**: queries that require both image and text to answer

## Schema

| Field | Description |
|-------|-------------|
| `query_id` | Unique identifier |
| `query` | Natural language question |
| `domain` | science_nature / geography_travel / history_art / technology / food_cooking / daily_life |
| `difficulty` | easy / medium / hard |
| `gold_doc_ids` | IDs of correct documents |
| `gold_doc_texts` | Text content of correct documents |
| `gold_answer` | Verified answer |
| `answer_modality` | text_only / image_only / cross_modal |
| `requires_image` | Whether the image is needed |
| `requires_text` | Whether the text is needed |
| `hallucination_traps` | JSON list of known failure patterns |
| `faithfulness_criteria` | JSON list of verifiable criteria |
| `hard_negatives` | IDs of plausible but wrong documents |

## Source Data

All from openly licensed sources: Wikipedia/WIT (CC-BY-SA 3.0), COCO (CC-BY 4.0). Generated annotations released under CC-BY 4.0.

## Citation

```bibtex
@software{bourigault2026mmragbench,
  author = {Bourigault, Emmanuelle},
  title = {MM-RAGBench: Multimodal Benchmark for Evaluating RAG Systems},
  year = {2026},
  url = {https://github.com/EmmanuelleB985/mm-ragbench},
}
```
