#!/usr/bin/env python3
"""Scalable GitHub repository mining pipeline.

Outputs JSON files consumed by the static dashboard:
- repository preprocessing
- technology graph with PageRank, HITS, and communities
- Apriori association rules
- BERT/sentence-transformer classification when available, with a deterministic
  TF-IDF/SVD fallback for offline classroom runs
- nearest-neighbor recommendations
- report and presentation outline
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import struct
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "github_repos.json"
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BERT_TINY_REPO = "phob0s/bert-tiny"
BERT_TINY_URLS = {
    "config.json": f"https://huggingface.co/{BERT_TINY_REPO}/resolve/main/config.json",
    "vocab.txt": f"https://huggingface.co/{BERT_TINY_REPO}/resolve/main/vocab.txt",
    "model.safetensors": f"https://huggingface.co/{BERT_TINY_REPO}/resolve/main/model.safetensors",
}
BERT_TINY_DIR = ROOT / "models" / "bert-tiny"
SAFETENSOR_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

STOPWORDS = {
    "awesome",
    "list",
    "lists",
    "tool",
    "tools",
    "resource",
    "resources",
    "tutorial",
    "book",
    "course",
    "open-source",
    "hacktoberfest",
    "cli",
    "api",
    "sdk",
    "library",
    "framework",
    "platform",
    "cloud-native",
    "serverless",
    "real-time",
    "high-performance",
    "scalable",
    "production-ready",
    "enterprise",
    "example",
    "examples",
    "template",
    "boilerplate",
}

NORMALIZE_TOPIC = {
    "ai": "artificial-intelligence",
    "dl": "deep-learning",
    "ml": "machine-learning",
    "nlp": "natural-language-processing",
    "cv": "computer-vision",
    "genai": "generative-ai",
    "llm": "large-language-model",
    "llms": "large-language-model",
    "js": "javascript",
    "ts": "typescript",
    "node": "nodejs",
    "node-js": "nodejs",
    "reactjs": "react",
    "vuejs": "vue",
    "next": "nextjs",
    "next-js": "nextjs",
    "rn": "react-native",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "pt": "pytorch",
    "cv2": "opencv",
    "web-3": "web3",
}

CATEGORIES = {
    "AI/ML Infrastructure": [
        "machine-learning",
        "deep-learning",
        "pytorch",
        "tensorflow",
        "keras",
        "neural-network",
        "training",
        "inference",
        "mlops",
        "automl",
        "scikit-learn",
        "model",
        "feature-store",
        "experiment-tracking",
    ],
    "Natural Language Processing": [
        "nlp",
        "natural-language-processing",
        "transformers",
        "bert",
        "text",
        "sentiment",
        "ner",
        "chatbot",
        "language-model",
        "tokenizer",
        "translation",
        "summarization",
    ],
    "Computer Vision": [
        "computer-vision",
        "opencv",
        "image",
        "object-detection",
        "yolo",
        "face",
        "segmentation",
        "ocr",
        "video",
        "pose",
        "cnn",
        "recognition",
    ],
    "Generative AI": [
        "large-language-model",
        "generative-ai",
        "stable-diffusion",
        "rag",
        "retrieval-augmented-generation",
        "ai-agents",
        "langchain",
        "fine-tuning",
        "lora",
        "prompt",
        "gpt",
        "diffusion",
        "vector-database",
        "embeddings",
    ],
    "Web Development": [
        "react",
        "vue",
        "angular",
        "nextjs",
        "svelte",
        "frontend",
        "ui",
        "components",
        "design-system",
        "css",
        "tailwindcss",
        "web",
        "spa",
        "ssr",
        "typescript",
        "javascript",
        "html",
    ],
    "Backend & APIs": [
        "api",
        "rest",
        "graphql",
        "backend",
        "fastapi",
        "django",
        "flask",
        "express",
        "microservices",
        "grpc",
        "server",
        "gateway",
        "middleware",
        "authentication",
        "nodejs",
    ],
    "Mobile Development": [
        "flutter",
        "react-native",
        "ios",
        "android",
        "swift",
        "kotlin",
        "mobile",
        "cross-platform",
        "dart",
        "swiftui",
        "jetpack-compose",
    ],
    "DevOps & Cloud": [
        "devops",
        "docker",
        "kubernetes",
        "ci-cd",
        "terraform",
        "ansible",
        "monitoring",
        "prometheus",
        "grafana",
        "gitops",
        "helm",
        "infrastructure",
        "cloud",
        "aws",
        "azure",
        "deployment",
    ],
    "Data Science": [
        "data-science",
        "pandas",
        "numpy",
        "visualization",
        "statistics",
        "analytics",
        "jupyter",
        "matplotlib",
        "seaborn",
        "etl",
        "data-pipeline",
        "forecasting",
        "time-series",
        "data-analysis",
    ],
    "Security": [
        "security",
        "vulnerability",
        "penetration-testing",
        "cryptography",
        "firewall",
        "encryption",
        "cybersecurity",
        "scanning",
        "compliance",
        "intrusion",
        "audit",
        "devsecops",
    ],
    "Blockchain & Web3": [
        "blockchain",
        "ethereum",
        "solidity",
        "smart-contracts",
        "web3",
        "defi",
        "nft",
        "crypto",
        "solana",
        "decentralized",
        "dao",
        "token",
        "dapp",
        "dex",
    ],
    "Game Development": [
        "game-development",
        "game-engine",
        "opengl",
        "vulkan",
        "unity",
        "godot",
        "unreal",
        "graphics",
        "rendering",
        "physics",
        "3d",
        "2d",
        "webgl",
        "gamedev",
    ],
    "Database & Storage": [
        "database",
        "sql",
        "nosql",
        "redis",
        "mongodb",
        "postgresql",
        "orm",
        "cache",
        "storage",
        "query",
        "replication",
        "graph-db",
        "key-value",
        "search",
        "index",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full GitHub mining pipeline.")
    parser.add_argument("--raw", default=str(RAW), help="Path to raw repository JSON.")
    parser.add_argument("--out", default=str(OUT), help="Processed JSON output directory.")
    parser.add_argument("--min-support", type=float, default=0.012, help="Minimum support ratio for Apriori.")
    parser.add_argument("--min-confidence", type=float, default=0.25, help="Minimum association confidence.")
    parser.add_argument("--min-lift", type=float, default=1.1, help="Minimum association lift.")
    parser.add_argument("--max-itemset", type=int, default=3, choices=[2, 3, 4], help="Max Apriori itemset size.")
    parser.add_argument("--max-features", type=int, default=12000, help="TF-IDF vocabulary size.")
    parser.add_argument("--svd-dims", type=int, default=96, help="Embedding dimensions for TF-IDF/SVD fallback.")
    parser.add_argument("--clusters", type=int, default=13, help="Number of repository clusters.")
    parser.add_argument("--neighbors", type=int, default=8, help="Similar repositories per repository.")
    parser.add_argument("--semantic-model", default="bert-tiny-numpy",
                        help="Use 'bert-tiny-numpy' for local real BERT, a SentenceTransformer model name, or 'tfidf'.")
    parser.add_argument("--bert-batch-size", type=int, default=64, help="BERT embedding batch size.")
    parser.add_argument("--bert-model-dir", default=str(BERT_TINY_DIR),
                        help="Directory containing BERT-Tiny config.json, vocab.txt, and model.safetensors.")
    parser.add_argument("--bert-max-length", type=int, default=96, help="Maximum BERT token sequence length.")
    parser.add_argument("--no-bert-download", action="store_true",
                        help="Do not download missing BERT-Tiny model files.")
    return parser.parse_args()


def clean_text(value: object, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\+\-\#\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    if limit:
        return text[:limit]
    return text


def norm_topic(topic: object) -> str:
    value = str(topic or "").lower().strip().replace("_", "-")
    value = re.sub(r"\s+", "-", value)
    value = NORMALIZE_TOPIC.get(value, value)
    return value


def normalize_topics(topics: object) -> list[str]:
    if not isinstance(topics, list):
        return []
    cleaned = []
    for topic in topics:
        value = norm_topic(topic)
        if value and value not in STOPWORDS and len(value) > 2:
            cleaned.append(value)
    return sorted(set(cleaned))


def parse_year(value: object, default: int = 2020) -> int:
    try:
        return int(str(value or "")[:4])
    except ValueError:
        return default


def popularity_bucket(stars: int) -> str:
    if stars >= 50000:
        return "mega"
    if stars >= 10000:
        return "high"
    if stars >= 1000:
        return "mid"
    return "rising"


def safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (ValueError, TypeError):
        return 0


def load_repos(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def preprocess(repos: list[dict]) -> list[dict]:
    cleaned = []
    seen = set()
    for repo in repos:
        name = str(repo.get("name") or repo.get("full_name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)

        language = str(repo.get("language") or "Unknown").strip() or "Unknown"
        topics = normalize_topics(repo.get("topics", []))
        tags = list(topics)
        if language != "Unknown":
            tags.append(language.lower())

        description = str(repo.get("description") or "")
        readme = str(repo.get("readme") or repo.get("readme_snippet") or "")
        clean_desc = clean_text(description, 600)
        clean_readme = clean_text(readme, 1800)
        combined_text = " ".join(
            part for part in [
                clean_desc,
                clean_readme,
                " ".join(tags),
                language.lower() if language else "",
            ] if part
        )

        year = parse_year(repo.get("created_at"))
        stars = safe_int(repo.get("stars") or repo.get("stargazers_count"))
        forks = safe_int(repo.get("forks") or repo.get("forks_count"))
        watchers = safe_int(repo.get("watchers") or repo.get("watchers_count"))
        issues = safe_int(repo.get("open_issues") or repo.get("open_issues_count"))

        cleaned.append({
            **repo,
            "name": name,
            "stars": stars,
            "forks": forks,
            "watchers": watchers,
            "open_issues": issues,
            "language": language,
            "topics": topics,
            "clean_desc": clean_desc,
            "clean_readme": clean_readme,
            "combined_text": combined_text,
            "tech_stack": sorted(set(tags)),
            "created_year": year,
            "era": "Pre-2018" if year < 2018 else ("2018-2021" if year <= 2021 else "2022+"),
            "popularity": popularity_bucket(stars),
            "license": str(repo.get("license") or "Unknown"),
            "html_url": repo.get("html_url") or f"https://github.com/{name}",
            "data_origin": repo.get("data_origin") or repo.get("source") or "unknown",
        })
    return cleaned


def pagerank(all_nodes: list[str], adj: dict[str, dict[str, float]], degree: dict[str, float],
             damping: float = 0.85, iterations: int = 80) -> dict[str, float]:
    n = max(len(all_nodes), 1)
    ranks = {node: 1.0 / n for node in all_nodes}
    for _ in range(iterations):
        new_ranks = {}
        dangling = sum(ranks[node] for node in all_nodes if degree.get(node, 0) == 0)
        for node in all_nodes:
            value = (1.0 - damping) / n
            value += damping * dangling / n
            for nb, weight in adj.get(node, {}).items():
                nb_degree = degree.get(nb, 0)
                if nb_degree:
                    value += damping * ranks[nb] * weight / nb_degree
            new_ranks[node] = value
        total = sum(new_ranks.values()) or 1.0
        ranks = {node: value / total for node, value in new_ranks.items()}
    return ranks


def hits(all_nodes: list[str], adj: dict[str, dict[str, float]], iterations: int = 60) -> tuple[dict[str, float], dict[str, float]]:
    hubs = {node: 1.0 for node in all_nodes}
    auths = {node: 1.0 for node in all_nodes}
    for _ in range(iterations):
        new_auths = {
            node: sum(hubs.get(nb, 0.0) * weight for nb, weight in adj.get(node, {}).items())
            for node in all_nodes
        }
        norm = math.sqrt(sum(value * value for value in new_auths.values())) or 1.0
        auths = {node: value / norm for node, value in new_auths.items()}

        new_hubs = {
            node: sum(auths.get(nb, 0.0) * weight for nb, weight in adj.get(node, {}).items())
            for node in all_nodes
        }
        norm = math.sqrt(sum(value * value for value in new_hubs.values())) or 1.0
        hubs = {node: value / norm for node, value in new_hubs.items()}
    return hubs, auths


def label_propagation(all_nodes: list[str], adj: dict[str, dict[str, float]], iterations: int = 50) -> dict[str, int]:
    labels = {node: i for i, node in enumerate(all_nodes)}
    for _ in range(iterations):
        changed = False
        for node in sorted(all_nodes, key=lambda n: (-sum(adj.get(n, {}).values()), n)):
            if not adj.get(node):
                continue
            weights = defaultdict(float)
            for nb, weight in adj[node].items():
                weights[labels[nb]] += weight
            best_label = max(weights.items(), key=lambda item: (item[1], -item[0]))[0]
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True
        if not changed:
            break

    ordered = sorted(set(labels.values()))
    remap = {old: i for i, old in enumerate(ordered)}
    return {node: remap[label] for node, label in labels.items()}


def semantic_communities(all_nodes: list[str]) -> dict[str, int]:
    """Fallback communities for dense real GitHub graphs.

    Topic co-occurrence graphs often become one large connected component because
    popular languages connect many domains. If propagation collapses into one
    community, category-keyword membership gives a clearer graph explanation.
    """
    category_ids = {label: i for i, label in enumerate(CATEGORIES)}
    fallback_id = len(category_ids)
    assignments = {}
    for tech in all_nodes:
        normalized = norm_topic(tech)
        assigned = fallback_id
        for label, keywords in CATEGORIES.items():
            if normalized in keywords or any(keyword in normalized or normalized in keyword for keyword in keywords):
                assigned = category_ids[label]
                break
        assignments[tech] = assigned
    return assignments


def build_graph(repos: list[dict]) -> tuple[list[dict], list[dict]]:
    tech_repos: dict[str, set[int]] = defaultdict(set)
    edge_counts: Counter[tuple[str, str]] = Counter()

    for i, repo in enumerate(repos):
        stack = sorted(set(repo.get("tech_stack", [])))
        for tech in stack:
            tech_repos[tech].add(i)
        for a, b in combinations(stack, 2):
            edge_counts[(a, b)] += 1

    all_techs = sorted(tech_repos)
    adj: dict[str, dict[str, float]] = defaultdict(dict)
    degree: dict[str, float] = defaultdict(float)
    for (a, b), weight in edge_counts.items():
        adj[a][b] = float(weight)
        adj[b][a] = float(weight)
        degree[a] += weight
        degree[b] += weight

    ranks = pagerank(all_techs, adj, degree)
    hubs, auths = hits(all_techs, adj)
    communities = label_propagation(all_techs, adj)
    if len(set(communities.values())) < 3 and len(all_techs) >= 20:
        communities = semantic_communities(all_techs)

    members: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for tech in all_techs:
        members[communities[tech]].append((tech, ranks[tech]))
    community_names = {
        community: f"{max(items, key=lambda item: item[1])[0].title()} Stack"
        for community, items in members.items()
    }

    nodes = [{
        "id": tech,
        "pagerank": round(ranks.get(tech, 0.0), 8),
        "hub": round(hubs.get(tech, 0.0), 8),
        "authority": round(auths.get(tech, 0.0), 8),
        "community": communities.get(tech, 0),
        "tribe": community_names.get(communities.get(tech, 0), "Technology Stack"),
        "count": len(tech_repos[tech]),
        "degree_weight": int(degree.get(tech, 0)),
    } for tech in all_techs]
    nodes.sort(key=lambda node: (-node["pagerank"], -node["count"], node["id"]))

    edges = [
        {"source": a, "target": b, "weight": int(weight)}
        for (a, b), weight in edge_counts.most_common(1500)
    ]
    return nodes, edges


def subset_iter(items: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    for size in range(1, len(items)):
        yield from combinations(items, size)


def mine_rules(repos: list[dict], min_support: float, min_confidence: float, min_lift: float,
               max_itemset: int) -> tuple[list[dict], dict[str, int]]:
    transactions = [tuple(sorted(set(repo.get("tech_stack", [])))) for repo in repos if repo.get("tech_stack")]
    n = len(transactions)
    if not n:
        return [], {}

    min_count = max(2, math.ceil(min_support * n))
    counts_by_size: dict[int, Counter[tuple[str, ...]]] = {}
    for size in range(1, max_itemset + 1):
        counts = Counter()
        for tx in transactions:
            if len(tx) >= size:
                counts.update(combinations(tx, size))
        counts_by_size[size] = Counter({items: count for items, count in counts.items() if count >= min_count})

    all_counts = {}
    for counts in counts_by_size.values():
        all_counts.update(counts)

    rules = []
    seen = set()
    for size in range(2, max_itemset + 1):
        for itemset, itemset_count in counts_by_size[size].items():
            itemset_set = set(itemset)
            support = itemset_count / n
            for antecedent in subset_iter(itemset):
                consequent = tuple(sorted(itemset_set - set(antecedent)))
                antecedent = tuple(sorted(antecedent))
                if not consequent:
                    continue
                antecedent_count = all_counts.get(antecedent, 0)
                consequent_count = all_counts.get(consequent, 0)
                if antecedent_count == 0 or consequent_count == 0:
                    continue
                confidence = itemset_count / antecedent_count
                consequent_support = consequent_count / n
                lift = confidence / consequent_support if consequent_support else 0
                if confidence >= min_confidence and lift >= min_lift:
                    key = (antecedent, consequent)
                    if key in seen:
                        continue
                    seen.add(key)
                    conviction = (1 - consequent_support) / (1 - confidence) if confidence < 1 else 999.0
                    leverage = support - ((antecedent_count / n) * consequent_support)
                    rules.append({
                        "antecedent": list(antecedent),
                        "consequent": list(consequent),
                        "support": round(support, 5),
                        "support_count": int(itemset_count),
                        "confidence": round(confidence, 5),
                        "lift": round(lift, 5),
                        "conviction": round(conviction, 5),
                        "leverage": round(leverage, 5),
                        "antecedent_count": int(antecedent_count),
                        "consequent_count": int(consequent_count),
                    })

    rules.sort(key=lambda rule: (-rule["lift"], -rule["confidence"], -rule["support_count"]))
    frequent_summary = {
        f"frequent_{size}_itemsets": len(counts)
        for size, counts in counts_by_size.items()
    }
    frequent_summary["min_support_count"] = min_count
    return rules, frequent_summary


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3)))


def layer_norm(values: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean = values.mean(axis=-1, keepdims=True)
    variance = np.mean((values - mean) ** 2, axis=-1, keepdims=True)
    normalized = (values - mean) / np.sqrt(variance + eps)
    return normalized * weight + bias


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        payload_start = 8 + header_size
        tensors = {}
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            dtype_name = metadata["dtype"]
            dtype = SAFETENSOR_DTYPES.get(dtype_name)
            if dtype is None:
                raise ValueError(f"Unsupported safetensors dtype {dtype_name} for {name}")
            start, end = metadata["data_offsets"]
            f.seek(payload_start + start)
            raw = f.read(end - start)
            array = np.frombuffer(raw, dtype=dtype).copy().reshape(metadata["shape"])
            if dtype_name == "BF16":
                array = (array.astype(np.uint32) << 16).view(np.float32)
            elif array.dtype != np.float32 and np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float32)
            tensors[name] = array
        for name, array in list(tensors.items()):
            if name.startswith("bert."):
                tensors.setdefault(name[5:], array)
        return tensors


def download_file(url: str, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=90) as response, tmp.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(path)


def ensure_bert_tiny_files(model_dir: Path, allow_download: bool = True) -> None:
    missing = []
    for filename in BERT_TINY_URLS:
        path = model_dir / filename
        min_size = 1_000_000 if filename.endswith(".safetensors") else 100
        if not path.exists() or path.stat().st_size < min_size:
            missing.append(filename)

    if not missing:
        return
    if not allow_download:
        raise FileNotFoundError(f"Missing BERT-Tiny files in {model_dir}: {', '.join(missing)}")

    print(f"Downloading real BERT-Tiny files from Hugging Face: {BERT_TINY_REPO}")
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in missing:
        print(f"  - {filename}")
        download_file(BERT_TINY_URLS[filename], model_dir / filename)


class WordPieceTokenizer:
    def __init__(self, vocab_path: Path) -> None:
        vocab = vocab_path.read_text(encoding="utf-8").splitlines()
        self.vocab = {token: i for i, token in enumerate(vocab)}
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]

    @staticmethod
    def _strip_accents(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def tokenize(self, text: str) -> list[str]:
        text = self._strip_accents(str(text).lower())
        words = re.findall(r"[a-z0-9]+|[^\s\w]", text)
        pieces = []
        for word in words:
            if word in self.vocab:
                pieces.append(word)
                continue
            start = 0
            word_pieces = []
            while start < len(word):
                end = len(word)
                current = None
                while start < end:
                    sub = word[start:end]
                    if start:
                        sub = "##" + sub
                    if sub in self.vocab:
                        current = sub
                        break
                    end -= 1
                if current is None:
                    word_pieces = ["[UNK]"]
                    break
                word_pieces.append(current)
                start = end
            pieces.extend(word_pieces)
        return pieces

    def encode_batch(self, texts: list[str], max_length: int) -> tuple[np.ndarray, np.ndarray]:
        input_ids = np.full((len(texts), max_length), self.pad_id, dtype=np.int64)
        attention = np.zeros((len(texts), max_length), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = ["[CLS]"] + self.tokenize(text)[: max_length - 2] + ["[SEP]"]
            ids = [self.vocab.get(token, self.unk_id) for token in tokens]
            input_ids[i, :len(ids)] = ids
            attention[i, :len(ids)] = 1.0
        return input_ids, attention


class NumpyBertTiny:
    def __init__(self, model_dir: Path, max_length: int) -> None:
        self.model_dir = model_dir
        self.max_length = max(16, min(512, max_length))
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.weights = load_safetensors(model_dir / "model.safetensors")
        self.tokenizer = WordPieceTokenizer(model_dir / "vocab.txt")
        self.hidden_size = int(self.config["hidden_size"])
        self.num_heads = int(self.config["num_attention_heads"])
        self.head_dim = self.hidden_size // self.num_heads
        self.num_layers = int(self.config["num_hidden_layers"])

    def _linear(self, values: np.ndarray, prefix: str) -> np.ndarray:
        weight = self.weights[f"{prefix}.weight"]
        bias = self.weights[f"{prefix}.bias"]
        return values @ weight.T + bias

    def _encoder_layer(self, values: np.ndarray, attention: np.ndarray, layer_index: int) -> np.ndarray:
        prefix = f"encoder.layer.{layer_index}"
        batch, seq_len, _ = values.shape

        query = self._linear(values, f"{prefix}.attention.self.query")
        key = self._linear(values, f"{prefix}.attention.self.key")
        value = self._linear(values, f"{prefix}.attention.self.value")

        def split_heads(x: np.ndarray) -> np.ndarray:
            return x.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        scores = scores + (1.0 - attention[:, None, None, :]) * -10000.0
        probs = softmax(scores, axis=-1)
        context = np.matmul(probs, value).transpose(0, 2, 1, 3).reshape(batch, seq_len, self.hidden_size)

        attention_output = self._linear(context, f"{prefix}.attention.output.dense")
        values = layer_norm(
            values + attention_output,
            self.weights[f"{prefix}.attention.output.LayerNorm.weight"],
            self.weights[f"{prefix}.attention.output.LayerNorm.bias"],
        )

        intermediate = gelu(self._linear(values, f"{prefix}.intermediate.dense"))
        layer_output = self._linear(intermediate, f"{prefix}.output.dense")
        values = layer_norm(
            values + layer_output,
            self.weights[f"{prefix}.output.LayerNorm.weight"],
            self.weights[f"{prefix}.output.LayerNorm.bias"],
        )
        return values

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        outputs = []
        batch_size = max(1, min(batch_size, 128))
        position_ids = np.arange(self.max_length)
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            input_ids, attention = self.tokenizer.encode_batch(batch_texts, self.max_length)
            token_type_ids = np.zeros_like(input_ids)

            values = (
                self.weights["embeddings.word_embeddings.weight"][input_ids]
                + self.weights["embeddings.position_embeddings.weight"][position_ids][None, :, :]
                + self.weights["embeddings.token_type_embeddings.weight"][token_type_ids]
            )
            values = layer_norm(
                values,
                self.weights["embeddings.LayerNorm.weight"],
                self.weights["embeddings.LayerNorm.bias"],
            )

            for layer_index in range(self.num_layers):
                values = self._encoder_layer(values, attention, layer_index)

            mask = attention[:, :, None]
            pooled = (values * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
            outputs.append(pooled)

        embeddings = np.vstack(outputs).astype(np.float32)
        return normalize(embeddings)


def category_prompts() -> list[str]:
    prompts = []
    for label, keywords in CATEGORIES.items():
        prompts.append(f"{label.lower()} repositories about {' '.join(keywords)}")
    return prompts


def try_sentence_transformer(texts: list[str], args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    model_name = args.semantic_model.lower()
    if model_name == "tfidf" or model_name in {"bert-tiny-numpy", "numpy-bert-tiny", BERT_TINY_REPO.lower()}:
        return None, None, "TF-IDF + SVD semantic fallback"
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(args.semantic_model)
        repo_embeddings = model.encode(
            texts,
            batch_size=args.bert_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        label_embeddings = model.encode(
            category_prompts(),
            batch_size=args.bert_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(repo_embeddings), np.asarray(label_embeddings), f"BERT/SentenceTransformer: {args.semantic_model}"
    except Exception as exc:
        print(f"SentenceTransformer unavailable, trying local BERT-Tiny: {exc}")
        return None, None, "SentenceTransformer unavailable"


def try_numpy_bert_tiny(texts: list[str], args: argparse.Namespace) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if args.semantic_model.lower() == "tfidf":
        return None, None, "TF-IDF + SVD semantic fallback"
    try:
        model_dir = Path(args.bert_model_dir)
        ensure_bert_tiny_files(model_dir, allow_download=not args.no_bert_download)
        encoder = NumpyBertTiny(model_dir, max_length=args.bert_max_length)
        repo_embeddings = encoder.encode(texts, batch_size=args.bert_batch_size)
        label_embeddings = encoder.encode(category_prompts(), batch_size=args.bert_batch_size)
        return repo_embeddings, label_embeddings, f"BERT/NumPy BERT-Tiny: {BERT_TINY_REPO}"
    except Exception as exc:
        print(f"Local BERT-Tiny unavailable, using TF-IDF/SVD fallback: {exc}")
        return None, None, "TF-IDF + SVD semantic fallback (BERT-ready)"


def semantic_analysis(repos: list[dict], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    texts = [repo.get("combined_text") or repo.get("name", "") for repo in repos]
    bert_embeddings, label_embeddings, method = try_sentence_transformer(texts, args)
    if bert_embeddings is None or label_embeddings is None:
        bert_embeddings, label_embeddings, method = try_numpy_bert_tiny(texts, args)

    if bert_embeddings is not None and label_embeddings is not None:
        scores = np.matmul(bert_embeddings, label_embeddings.T)
        embeddings = normalize(bert_embeddings)
    else:
        prompts = category_prompts()
        vectorizer = TfidfVectorizer(
            max_features=args.max_features,
            min_df=2,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(texts + prompts)
        repo_matrix = matrix[:len(texts)]
        label_matrix = matrix[len(texts):]
        scores = repo_matrix @ label_matrix.T
        scores = scores.toarray()

        dims = min(args.svd_dims, max(2, min(repo_matrix.shape) - 1))
        svd = TruncatedSVD(n_components=dims, random_state=SEED)
        embeddings = normalize(svd.fit_transform(repo_matrix))

    labels = list(CATEGORIES.keys())
    best_idx = scores.argmax(axis=1)
    sorted_scores = np.sort(scores, axis=1)
    best = sorted_scores[:, -1]
    second = sorted_scores[:, -2] if scores.shape[1] > 1 else np.zeros_like(best)
    raw_conf = 0.55 + np.clip(best - second, 0, 1) * 0.45

    for i, repo in enumerate(repos):
        repo["bert_category"] = labels[int(best_idx[i])]
        repo["bert_confidence"] = round(float(raw_conf[i]), 4)
        repo["semantic_score"] = round(float(best[i]), 4)

    return embeddings, scores, method


def cluster_and_project(repos: list[dict], embeddings: np.ndarray, clusters: int) -> None:
    n = len(repos)
    if n == 0:
        return

    k = max(1, min(clusters, n))
    if k == 1:
        labels = np.zeros(n, dtype=int)
    else:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=SEED,
            batch_size=min(2048, max(128, n)),
            n_init=5,
        )
        labels = model.fit_predict(embeddings)

    if embeddings.shape[1] >= 2:
        try:
            projection = PCA(n_components=2, random_state=SEED).fit_transform(embeddings)
        except Exception:
            projection = TruncatedSVD(n_components=2, random_state=SEED).fit_transform(embeddings)
    else:
        projection = np.column_stack([embeddings[:, 0], np.zeros(n)])

    for i, repo in enumerate(repos):
        repo["cluster_id"] = int(labels[i])
        repo["pca_x"] = round(float(projection[i, 0]), 6)
        repo["pca_y"] = round(float(projection[i, 1]), 6)


def compute_neighbors(repos: list[dict], embeddings: np.ndarray, top_k: int) -> dict[str, list[list[object]]]:
    if len(repos) <= 1:
        return {repo["name"]: [] for repo in repos}
    top_k = max(1, min(top_k, len(repos) - 1))
    nn = NearestNeighbors(n_neighbors=top_k + 1, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings, return_distance=True)

    output = {}
    for i, repo in enumerate(repos):
        items = []
        for dist, idx in zip(distances[i], indices[i]):
            if idx == i:
                continue
            score = max(0.0, 1.0 - float(dist))
            items.append([repos[int(idx)]["name"], round(score, 5)])
            if len(items) >= top_k:
                break
        output[repo["name"]] = items
    return output


def export_json(path: Path, payload: object, indent: int | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)


def build_stats(repos: list[dict], nodes: list[dict], edges: list[dict], rules: list[dict],
                frequent_summary: dict[str, int], embedding_method: str,
                started: float) -> dict:
    category_dist = Counter(repo["bert_category"] for repo in repos)
    language_dist = Counter(repo["language"] for repo in repos)
    era_dist = Counter(repo["era"] for repo in repos)
    popularity_dist = Counter(repo["popularity"] for repo in repos)
    origin_dist = Counter(repo.get("data_origin", "unknown") for repo in repos)
    cluster_dist = Counter(repo["cluster_id"] for repo in repos)
    license_dist = Counter(repo.get("license", "Unknown") for repo in repos)

    top_techs = [node["id"] for node in nodes[:12]]
    trends: dict[str, dict[str, int]] = {tech: defaultdict(int) for tech in top_techs}
    for repo in repos:
        year = str(repo["created_year"])
        for tech in repo["tech_stack"]:
            if tech in trends:
                trends[tech][year] += 1

    top_repos = sorted(repos, key=lambda repo: (-repo["stars"], -repo["forks"], repo["name"]))[:25]
    avg_conf = float(np.mean([repo.get("bert_confidence", 0.0) for repo in repos])) if repos else 0.0
    star_values = [repo["stars"] for repo in repos]
    fork_values = [repo["forks"] for repo in repos]

    top_rule = rules[0] if rules else None
    top_pr = nodes[0] if nodes else None
    insights = []
    if top_pr:
        insights.append(
            f"{top_pr['id']} is the most central technology by PageRank, appearing in {top_pr['count']} repositories."
        )
    if top_rule:
        lhs = " + ".join(top_rule["antecedent"])
        rhs = " + ".join(top_rule["consequent"])
        insights.append(
            f"The strongest association rule is {lhs} -> {rhs} with lift {top_rule['lift']} and confidence {top_rule['confidence']}."
        )
    if category_dist:
        cat, count = category_dist.most_common(1)[0]
        insights.append(f"{cat} is the largest semantic category with {count} repositories.")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(time.time() - started, 2),
        "data_source_distribution": dict(origin_dist.most_common()),
        "embedding_method": embedding_method,
        "total_repos": len(repos),
        "total_technologies": len(nodes),
        "total_edges": len(edges),
        "total_communities": len(set(node["community"] for node in nodes)),
        "total_rules": len(rules),
        "total_categories": len(category_dist),
        "average_bert_confidence": round(avg_conf, 4),
        "stars": {
            "total": int(sum(star_values)),
            "mean": round(float(np.mean(star_values)), 2) if star_values else 0,
            "median": round(float(np.median(star_values)), 2) if star_values else 0,
            "max": int(max(star_values)) if star_values else 0,
        },
        "forks": {
            "total": int(sum(fork_values)),
            "mean": round(float(np.mean(fork_values)), 2) if fork_values else 0,
            "median": round(float(np.median(fork_values)), 2) if fork_values else 0,
            "max": int(max(fork_values)) if fork_values else 0,
        },
        "category_distribution": dict(category_dist.most_common()),
        "language_distribution": dict(language_dist.most_common(25)),
        "era_distribution": dict(era_dist),
        "popularity_distribution": dict(popularity_dist),
        "license_distribution": dict(license_dist.most_common(12)),
        "cluster_distribution": {str(k): v for k, v in cluster_dist.most_common()},
        "frequent_itemsets": frequent_summary,
        "top_pagerank": [{"tech": node["id"], "score": node["pagerank"], "repos": node["count"]} for node in nodes[:20]],
        "top_authorities": [{"tech": node["id"], "score": node["authority"], "repos": node["count"]} for node in sorted(nodes, key=lambda n: -n["authority"])[:15]],
        "top_hubs": [{"tech": node["id"], "score": node["hub"], "repos": node["count"]} for node in sorted(nodes, key=lambda n: -n["hub"])[:15]],
        "top_repositories": [{
            "name": repo["name"],
            "stars": repo["stars"],
            "forks": repo["forks"],
            "language": repo["language"],
            "category": repo["bert_category"],
            "url": repo["html_url"],
        } for repo in top_repos],
        "trends": {tech: dict(sorted(years.items())) for tech, years in trends.items()},
        "insights": insights,
    }


def export_repos(repos: list[dict]) -> list[dict]:
    output = []
    for repo in repos:
        output.append({
            "name": repo["name"],
            "stars": repo["stars"],
            "forks": repo["forks"],
            "watchers": repo.get("watchers", 0),
            "open_issues": repo.get("open_issues", 0),
            "language": repo["language"],
            "topics": repo.get("topics", [])[:10],
            "description": str(repo.get("description") or "")[:260],
            "tech_stack": repo.get("tech_stack", [])[:12],
            "created_year": repo["created_year"],
            "era": repo["era"],
            "popularity": repo["popularity"],
            "bert_category": repo["bert_category"],
            "bert_confidence": repo["bert_confidence"],
            "semantic_score": repo.get("semantic_score", 0),
            "cluster_id": repo["cluster_id"],
            "pca_x": repo["pca_x"],
            "pca_y": repo["pca_y"],
            "license": repo.get("license", "Unknown"),
            "html_url": repo["html_url"],
            "data_origin": repo.get("data_origin", "unknown"),
        })
    output.sort(key=lambda repo: (-repo["stars"], repo["name"]))
    return output


def write_report(stats: dict, nodes: list[dict], rules: list[dict]) -> None:
    top_rules = "\n".join(
        f"| {' + '.join(rule['antecedent'])} | {' + '.join(rule['consequent'])} | {rule['support']} | {rule['confidence']} | {rule['lift']} |"
        for rule in rules[:12]
    )
    top_techs = "\n".join(
        f"| {i + 1} | {node['id']} | {node['pagerank']} | {node['authority']} | {node['hub']} | {node['count']} |"
        for i, node in enumerate(nodes[:12])
    )
    insights = "\n".join(f"- {item}" for item in stats["insights"])

    report = f"""# GitHub Repository Mining - Data Mining Report

Generated: {stats['generated_at']}

## Objective

This project mines GitHub repository data to discover software development trends, influential technologies, common technology stacks, and semantically similar projects.

## Dataset

- Repositories analyzed: {stats['total_repos']:,}
- Unique technologies/topics: {stats['total_technologies']:,}
- Technology graph edges exported: {stats['total_edges']:,}
- Data source distribution: {stats['data_source_distribution']}
- Embedding method: {stats['embedding_method']}

## Preprocessing

Repository descriptions, README snippets, languages, and topics were cleaned and normalized. Topics were merged with programming languages to create a transaction-style `tech_stack` for each repository.

## Association Rule Mining

Apriori was applied to repository technology stacks. The pipeline exports support, support count, confidence, lift, conviction, and leverage.

| Antecedent | Consequent | Support | Confidence | Lift |
|---|---|---:|---:|---:|
{top_rules}

## Link Analysis

Technologies are nodes. Weighted edges represent co-occurrence inside repository tech stacks. PageRank identifies central ecosystem technologies, while HITS identifies hub and authority roles.

| Rank | Technology | PageRank | Authority | Hub | Repositories |
|---:|---|---:|---:|---:|---:|
{top_techs}

## Semantic/BERT Analysis

Repository text is classified into {stats['total_categories']} categories. The default pipeline uses real pretrained BERT-Tiny embeddings through a lightweight NumPy inference path. A SentenceTransformer model can also be supplied with `--semantic-model` when PyTorch is available.

- Average classification confidence: {stats['average_bert_confidence']}
- Largest categories: {list(stats['category_distribution'].items())[:5]}

## Key Insights

{insights}

## Decision Support

The mined rules and graph rankings can be used to recommend technology stacks, identify trending repository domains, and find similar projects for learning or reuse.
"""
    (REPORTS / "project_report.md").write_text(report, encoding="utf-8")

    slides = f"""# Presentation Outline

1. Project goal and grading requirements
2. GitHub data collection strategy
3. Preprocessing and feature engineering
4. Association rule mining with Apriori
5. Link analysis with PageRank and HITS
6. BERT/semantic classification and similarity search
7. Dashboard walkthrough
8. Findings and technology stack recommendations
9. Limitations and future work

Key numbers:
- {stats['total_repos']:,} repositories
- {stats['total_technologies']:,} technologies
- {stats['total_rules']:,} rules
- {stats['total_communities']:,} communities
- {stats['total_categories']:,} semantic categories
"""
    (REPORTS / "presentation_outline.md").write_text(slides, encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    raw_path = Path(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw repositories from {raw_path}")
    raw_repos = load_repos(raw_path)
    repos = preprocess(raw_repos)
    print(f"Preprocessed {len(repos):,} unique repositories")

    print("Building technology graph and running PageRank/HITS")
    nodes, edges = build_graph(repos)
    print(f"Graph: {len(nodes):,} nodes, {len(edges):,} exported edges")

    print("Mining association rules with Apriori")
    rules, frequent_summary = mine_rules(
        repos,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_itemset=args.max_itemset,
    )
    print(f"Rules: {len(rules):,} discovered")

    print("Running semantic/BERT classification and embeddings")
    embeddings, _, embedding_method = semantic_analysis(repos, args)
    print(f"Semantic method: {embedding_method}")

    print("Clustering repositories and projecting embeddings")
    cluster_and_project(repos, embeddings, args.clusters)

    print("Computing nearest-neighbor recommendations")
    similarities = compute_neighbors(repos, embeddings, args.neighbors)

    stats = build_stats(repos, nodes, edges, rules, frequent_summary, embedding_method, started)
    export_json(out_dir / "graph_nodes.json", nodes)
    export_json(out_dir / "graph_edges.json", edges)
    export_json(out_dir / "association_rules.json", rules, indent=1)
    export_json(out_dir / "repos.json", export_repos(repos))
    export_json(out_dir / "similarities.json", similarities)
    export_json(out_dir / "stats.json", stats, indent=1)
    write_report(stats, nodes, rules)

    print("")
    print("Pipeline complete")
    print(f"Repositories: {stats['total_repos']:,}")
    print(f"Technologies:  {stats['total_technologies']:,}")
    print(f"Rules:         {stats['total_rules']:,}")
    print(f"Communities:   {stats['total_communities']:,}")
    print(f"Runtime:       {stats['runtime_seconds']}s")


if __name__ == "__main__":
    main()
