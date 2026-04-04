"""MM-RAGBench dataset builder.

Constructs the benchmark from open-license sources (WIT, COCO, Visual Genome)
and outputs samples compatible with mmeval-vrag's EvalSample / QueryItem types.

Usage
-----
    from mm_ragbench import MMRAGBenchBuilder

    builder = MMRAGBenchBuilder(output_dir="data/mm-ragbench")
    builder.collect_sources()          # Step 1: pull images + text
    builder.generate_queries()         # Step 2: LLM-based query generation
    builder.generate_hard_negatives()  # Step 3: mine confusing distractors
    builder.verify_and_balance()       # Step 4: quality checks + balancing
    builder.export_jsonl()             # Step 5: JSONL for mmeval-vrag
    builder.push_to_hub("EmmanuelleB985/mm-ragbench")  # Step 6: upload

Storage modes (pick based on your disk budget)
----------------------------------------------
    "urls"       ->  ~10 MB   — text + image URLs only, no local images
    "thumbnails"  ->  ~300 MB  — 224px JPEG thumbnails stored locally
    "full"        ->  ~50 GB   — original-resolution images stored locally
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────
# Query generation prompt (fed to Claude / GPT-4o)
# ──────────────────────────────────────────────────────────

QUERY_GEN_SYSTEM = (
    "You are an expert benchmark designer for multimodal AI systems. "
    "Output only valid JSON with no markdown fences."
)

QUERY_GEN_TEMPLATE = """\
You are creating evaluation queries for a multimodal RAG benchmark called MM-RAGBench.

Here is a document from the knowledge base:

PAGE TITLE: {page_title}
TEXT: {text}
IMAGE CAPTION: {caption}

Generate exactly 3 queries (easy, medium, hard). Each query should test whether a
RAG system can retrieve this document AND reason over both the image and the text.

Return JSON:
{{
  "queries": [
    {{
      "query": "<natural language question>",
      "difficulty": "easy|medium|hard",
      "gold_answer": "<correct answer from the document>",
      "answer_modality": "text_only|image_only|cross_modal",
      "requires_image": true/false,
      "requires_text": true/false,
      "hallucination_traps": [
        {{"type": "object|attribute|relation|fabrication", "description": "..."}}
      ],
      "faithfulness_criteria": ["<verifiable criterion>"]
    }}
  ]
}}

Difficulty definitions:
- easy: answerable from text OR image alone
- medium: requires combining text AND image
- hard: requires careful reasoning or resolving ambiguity across modalities
"""


@dataclass
class DocumentRecord:
    """Internal representation of a source document."""
    doc_id: str
    source: str
    domain: str
    text: str
    caption: str
    page_title: str = ""
    page_url: str = ""
    image_license: str = "CC-BY-SA-3.0"
    text_license: str = "CC-BY-SA-3.0"
    # Storage-mode dependent — exactly one is set
    image_path: Optional[str] = None   # local path  (thumbnails / full)
    image_url: Optional[str] = None    # remote URL   (urls mode)


@dataclass
class QueryRecord:
    """Internal representation of a benchmark query."""
    query_id: str
    query: str
    domain: str
    difficulty: str
    gold_doc_ids: List[str]
    gold_answer: str
    answer_modality: str = "cross_modal"
    requires_image: bool = True
    requires_text: bool = True
    hallucination_traps: List[Dict[str, str]] = field(default_factory=list)
    faithfulness_criteria: List[str] = field(default_factory=list)
    hard_negatives: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────
# Disk-size estimates printed at startup
# ──────────────────────────────────────────────────────────

_SIZE_ESTIMATES = {
    "urls":       "~10 MB   (text + image URLs, no local images)",
    "thumbnails": "~300 MB  (224 px JPEG thumbnails)",
    "full":       "~50 GB   (original-resolution images)",
}


class MMRAGBenchBuilder:
    """Build the MM-RAGBench dataset step by step.

    Parameters
    ----------
    output_dir : str
        Where to write intermediate and final data.
    storage_mode : "urls" | "thumbnails" | "full"
        Controls disk usage:
        - ``"urls"`` (~10 MB): stores only text + Wikimedia image URLs.
          Images are fetched lazily during evaluation via ``ImageInput(url=...)``.
        - ``"thumbnails"`` (~300 MB): saves 224 px JPEG thumbnails locally.
        - ``"full"`` (~50 GB): saves original-resolution images locally.
    llm_provider : str
        "anthropic" or "openai" — used for query generation.
    llm_model : str
        Model name for query generation.
    seed : int
        Random seed for reproducibility.
    """

    DOMAINS = [
        "science_nature",
        "geography_travel",
        "history_art",
        "technology",
        "food_cooking",
        "daily_life",
    ]

    DOMAIN_KEYWORDS = {
        "science_nature": ["biology", "chemistry", "physics", "species", "cell", "molecule", "diagram"],
        "geography_travel": ["bridge", "mountain", "river", "landmark", "city", "monument", "map"],
        "history_art": ["painting", "sculpture", "cathedral", "ruins", "historical", "century", "museum"],
        "technology": ["circuit", "device", "robot", "satellite", "engine", "software", "interface"],
        "food_cooking": ["recipe", "dish", "ingredient", "cooking", "baking", "cuisine", "ferment"],
        "daily_life": ["tool", "vehicle", "furniture", "garden", "exercise", "clothing", "sport"],
    }

    THUMB_SIZE = 224  # pixels — longest edge for thumbnail mode

    def __init__(
        self,
        output_dir: str = "data/mm-ragbench",
        storage_mode: Literal["urls", "thumbnails", "full"] = "thumbnails",
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-20250514",
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.storage_mode = storage_mode
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.documents: List[DocumentRecord] = []
        self.queries: List[QueryRecord] = []

        print(f"Storage mode: {storage_mode}  →  {_SIZE_ESTIMATES[storage_mode]}")

    # ── Step 1: Collect sources ───────────────────────────

    def collect_sources(
        self,
        wit_samples: int = 9000,
        coco_samples: int = 3000,
        min_text_tokens: int = 30,
    ):
        """Pull image-text pairs from WIT and COCO datasets.

        Disk usage depends on ``storage_mode`` set in the constructor:
        - ``"urls"``       → no images saved, only Wikimedia URLs (~10 MB)
        - ``"thumbnails"`` → 224 px JPEGs (~300 MB for 12 K images)
        - ``"full"``       → original resolution (~50 GB)
        """
        from datasets import load_dataset as hf_load

        save_images = self.storage_mode in ("thumbnails", "full")
        img_dir = None
        if save_images:
            img_dir = self.output_dir / "images"
            img_dir.mkdir(exist_ok=True)

        # ─── WIT (Wikipedia Image-Text) ───
        print("📚 Loading WIT dataset (streaming — no bulk download)...")
        ds = hf_load("wikimedia/wit_base", split="train", streaming=True)
        domain_counts = {d: 0 for d in self.DOMAINS}
        target_per_domain = wit_samples // len(self.DOMAINS)

        for sample in tqdm(ds, desc="WIT", total=wit_samples):
            if sum(domain_counts.values()) >= wit_samples:
                break

            wit_features = sample.get("wit_features")
            if not wit_features:
                continue

            en_entry = next((f for f in wit_features if f.get("language") == "en"), None)
            if not en_entry:
                continue

            caption = en_entry.get("caption_reference_description", "")
            if not caption or len(caption.split()) < min_text_tokens // 3:
                continue

            image = sample.get("image")
            if image is None:
                continue
            if min(image.size) < 100:
                continue

            page_title = en_entry.get("page_title", "")
            page_url = en_entry.get("page_url", "")
            section_text = en_entry.get("context_section_description", "")
            full_text = f"{page_title}. {caption}. {section_text}"

            domain = self._classify_domain(full_text)
            if domain_counts[domain] >= target_per_domain:
                continue

            doc_id = f"wit-{hashlib.md5(page_url.encode() or str(len(self.documents)).encode()).hexdigest()[:12]}"

            # ── image handling per storage mode ──
            image_path = None
            image_url = None

            if self.storage_mode == "urls":
                # Construct the Wikimedia thumbnail URL (no local save)
                image_url = en_entry.get("image_url") or en_entry.get("url", "")
            elif self.storage_mode == "thumbnails":
                thumb = self._make_thumbnail(image)
                image_path = str(img_dir / f"{doc_id}.jpg")
                thumb.save(image_path, "JPEG", quality=80)
            else:  # full
                image_path = str(img_dir / f"{doc_id}.jpg")
                image.save(image_path, "JPEG", quality=85)

            self.documents.append(DocumentRecord(
                doc_id=doc_id,
                source="wit",
                domain=domain,
                text=full_text[:1500],
                caption=caption,
                page_title=page_title,
                page_url=page_url,
                image_path=image_path,
                image_url=image_url,
            ))
            domain_counts[domain] += 1

        print(f"  WIT: {sum(domain_counts.values())} documents. Per-domain: {domain_counts}")

        # ─── COCO ───
        print("📸 Loading COCO dataset...")
        try:
            coco_ds = hf_load("detection-datasets/coco", split="val", streaming=True)
            coco_count = 0
            for sample in tqdm(coco_ds, desc="COCO", total=coco_samples):
                if coco_count >= coco_samples:
                    break
                image = sample.get("image")
                if image is None:
                    continue

                objects = sample.get("objects", {})
                categories = objects.get("category", []) if isinstance(objects, dict) else []
                caption = f"Image containing: {', '.join(set(str(c) for c in categories[:10]))}" if categories else "Everyday scene"

                img_id = sample.get("image_id", coco_count)
                doc_id = f"coco-{img_id:08d}" if isinstance(img_id, int) else f"coco-{coco_count:08d}"

                image_path = None
                image_url = None
                if self.storage_mode == "urls":
                    image_url = f"http://images.cocodataset.org/val2017/{img_id:012d}.jpg"
                elif self.storage_mode == "thumbnails":
                    thumb = self._make_thumbnail(image)
                    image_path = str(img_dir / f"{doc_id}.jpg")
                    thumb.save(image_path, "JPEG", quality=80)
                else:
                    image_path = str(img_dir / f"{doc_id}.jpg")
                    image.save(image_path, "JPEG", quality=85)

                domain = self._classify_domain(caption)
                self.documents.append(DocumentRecord(
                    doc_id=doc_id,
                    source="coco",
                    domain=domain,
                    text=caption,
                    caption=caption,
                    image_path=image_path,
                    image_url=image_url,
                    image_license="CC-BY-4.0",
                    text_license="CC-BY-4.0",
                ))
                coco_count += 1
        except Exception as e:
            print(f"  COCO loading failed ({e}), continuing with WIT only")

        # Save document metadata
        meta_path = self.output_dir / "documents.jsonl"
        with open(meta_path, "w") as f:
            for doc in self.documents:
                f.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")

        # Report actual disk usage
        if save_images and img_dir:
            total_bytes = sum(f.stat().st_size for f in img_dir.iterdir() if f.is_file())
            print(f"  Image folder: {total_bytes / 1e6:.0f} MB ({len(list(img_dir.iterdir()))} files)")

        print(f"Total documents: {len(self.documents)} → {meta_path}")

    def _make_thumbnail(self, image) -> "Image":
        """Resize keeping aspect ratio so longest edge = THUMB_SIZE."""
        from PIL import Image as PILImage
        image = image.copy()
        image.thumbnail((self.THUMB_SIZE, self.THUMB_SIZE), PILImage.LANCZOS)
        return image

    def _classify_domain(self, text: str) -> str:
        text_lower = text.lower()
        scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            scores[domain] = sum(1 for kw in keywords if kw in text_lower)
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            return random.choice(self.DOMAINS)
        return best

    # ── Step 2: Generate queries ──────────────────────────

    def generate_queries(self, max_docs: Optional[int] = None):
        """Generate evaluation queries using an LLM.

        Each document produces up to 3 queries (easy, medium, hard).
        """
        docs = self.documents[:max_docs] if max_docs else self.documents
        print(f"🔍 Generating queries for {len(docs)} documents...")

        for doc in tqdm(docs, desc="Query generation"):
            prompt = QUERY_GEN_TEMPLATE.format(
                page_title=doc.page_title or "Untitled",
                text=doc.text[:800],
                caption=doc.caption[:300],
            )

            result = self._call_llm(prompt)
            if not result or "queries" not in result:
                continue

            for i, q in enumerate(result["queries"]):
                if not q.get("query") or not q.get("gold_answer"):
                    continue
                self.queries.append(QueryRecord(
                    query_id=f"{doc.doc_id}-q{i:02d}",
                    query=q["query"],
                    domain=doc.domain,
                    difficulty=q.get("difficulty", "medium"),
                    gold_doc_ids=[doc.doc_id],
                    gold_answer=q["gold_answer"],
                    answer_modality=q.get("answer_modality", "cross_modal"),
                    requires_image=q.get("requires_image", True),
                    requires_text=q.get("requires_text", True),
                    hallucination_traps=q.get("hallucination_traps", []),
                    faithfulness_criteria=q.get("faithfulness_criteria", []),
                ))

        print(f"  Generated {len(self.queries)} raw queries")

    # ── Step 3: Hard negatives ────────────────────────────

    def generate_hard_negatives(self, negatives_per_query: int = 3):
        """Assign hard negative document IDs to each query.

        Uses domain overlap + random selection within the same domain.
        """
        print("Generating hard negatives...")
        docs_by_domain: Dict[str, List[str]] = {}
        for doc in self.documents:
            docs_by_domain.setdefault(doc.domain, []).append(doc.doc_id)

        for query in self.queries:
            gold_set = set(query.gold_doc_ids)
            candidates = [
                d for d in docs_by_domain.get(query.domain, [])
                if d not in gold_set
            ]
            if len(candidates) < negatives_per_query:
                # Pull from other domains
                all_ids = [d.doc_id for d in self.documents if d.doc_id not in gold_set]
                candidates = all_ids
            query.hard_negatives = random.sample(
                candidates, min(negatives_per_query, len(candidates))
            )

    # ── Step 4: Verify and balance ────────────────────────

    def verify_and_balance(self, target: int = 3000):
        """Quality check and balance across domains / difficulties."""
        print("Balancing dataset...")

        # Remove queries with empty answers or queries
        self.queries = [q for q in self.queries if q.query.strip() and q.gold_answer.strip()]

        target_per_domain = target // len(self.DOMAINS)
        balanced = []

        by_domain: Dict[str, List[QueryRecord]] = {}
        for q in self.queries:
            by_domain.setdefault(q.domain, []).append(q)

        for domain, domain_qs in by_domain.items():
            random.shuffle(domain_qs)
            easy = [q for q in domain_qs if q.difficulty == "easy"]
            medium = [q for q in domain_qs if q.difficulty == "medium"]
            hard = [q for q in domain_qs if q.difficulty == "hard"]

            n_easy = min(len(easy), int(target_per_domain * 0.40))
            n_med = min(len(medium), int(target_per_domain * 0.35))
            n_hard = min(len(hard), int(target_per_domain * 0.25))

            balanced.extend(easy[:n_easy])
            balanced.extend(medium[:n_med])
            balanced.extend(hard[:n_hard])

        random.shuffle(balanced)
        self.queries = balanced[:target]

        # Stats
        diff_counts = {}
        domain_counts = {}
        for q in self.queries:
            diff_counts[q.difficulty] = diff_counts.get(q.difficulty, 0) + 1
            domain_counts[q.domain] = domain_counts.get(q.domain, 0) + 1

        print(f"  Final: {len(self.queries)} queries")
        print(f"  Difficulty: {diff_counts}")
        print(f"  Domains: {domain_counts}")

    # ── Step 5: Export JSONL ──────────────────────────────

    def export_jsonl(self, filename: str = "mm_ragbench.jsonl"):
        """Export the benchmark as JSONL compatible with mmeval-vrag.

        The output can be loaded with:
            from mmeval_vrag.datasets import load_dataset
            samples = load_dataset("mm_ragbench", "path/to/mm_ragbench.jsonl")

        Or with the MM-RAGBench loader directly:
            from mm_ragbench import load_eval_samples
            samples = load_eval_samples("path/to/mm_ragbench.jsonl")
        """
        doc_lookup = {d.doc_id: d for d in self.documents}
        out_path = self.output_dir / filename

        with open(out_path, "w") as f:
            for q in self.queries:
                gold_docs = [doc_lookup[did] for did in q.gold_doc_ids if did in doc_lookup]

                record = {
                    "query_id": q.query_id,
                    "query": q.query,
                    "query_image": None,
                    "domain": q.domain,
                    "difficulty": q.difficulty,
                    "gold_doc_ids": q.gold_doc_ids,
                    "gold_doc_texts": [d.text for d in gold_docs],
                    "gold_doc_images": [d.image_path or "" for d in gold_docs],
                    "gold_doc_image_urls": [d.image_url or "" for d in gold_docs],
                    "gold_doc_licenses": [d.image_license for d in gold_docs],
                    "hard_negatives": q.hard_negatives,
                    "gold_answer": q.gold_answer,
                    "answer_modality": q.answer_modality,
                    "requires_image": q.requires_image,
                    "requires_text": q.requires_text,
                    "hallucination_traps": json.dumps(q.hallucination_traps),
                    "faithfulness_criteria": json.dumps(q.faithfulness_criteria),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Exported {len(self.queries)} queries → {out_path}")
        return str(out_path)

    # ── Step 6: Push to HuggingFace Hub ───────────────────

    def push_to_hub(
        self,
        repo_id: str = "EmmanuelleB985/mm-ragbench",
        jsonl_path: Optional[str] = None,
    ):
        """Upload the dataset to HuggingFace Hub."""
        from datasets import Dataset, DatasetDict
        from huggingface_hub import HfApi

        path = jsonl_path or str(self.output_dir / "mm_ragbench.jsonl")
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))

        n_dev = min(500, len(records) // 6)
        dev_records = records[:n_dev]
        test_records = records[n_dev:]

        ds_dict = DatasetDict({
            "test": Dataset.from_list(test_records),
            "dev": Dataset.from_list(dev_records),
        })

        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        ds_dict.push_to_hub(repo_id, private=False)

        print(f" Pushed to https://huggingface.co/datasets/{repo_id}")
        print(f"  test: {len(test_records)} | dev: {len(dev_records)}")

    # ── LLM helper ────────────────────────────────────────

    def _call_llm(self, prompt: str) -> Optional[dict]:
        """Call the configured LLM and parse JSON response."""
        try:
            if self.llm_provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic()
                msg = client.messages.create(
                    model=self.llm_model,
                    max_tokens=2000,
                    temperature=0.7,
                    system=QUERY_GEN_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = msg.content[0].text
            elif self.llm_provider == "openai":
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=self.llm_model,
                    temperature=0.7,
                    messages=[
                        {"role": "system", "content": QUERY_GEN_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = resp.choices[0].message.content
            else:
                return None

            clean = text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except Exception as e:
            print(f" LLM call failed: {e}")
            return None
