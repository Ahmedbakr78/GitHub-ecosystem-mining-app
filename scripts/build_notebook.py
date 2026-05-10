#!/usr/bin/env python3
"""Build a full academic submission notebook for the GitHub mining project."""
from __future__ import annotations

import json
import base64
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "data" / "processed" / "stats.json"
NOTEBOOK = ROOT / "github_mining_analysis.ipynb"


BUNDLE_FILES = [
    "README.md",
    "requirements.txt",
    "app/index.html",
    "app/css/styles.css",
    "app/js/app.js",
    "data/raw/github_repos.json",
    "data/processed/repos.json",
    "data/processed/graph_nodes.json",
    "data/processed/graph_edges.json",
    "data/processed/association_rules.json",
    "data/processed/similarities.json",
    "data/processed/stats.json",
    "outputs/project_report.md",
    "outputs/presentation_outline.md",
    "outputs/validation_report.md",
    "scripts/collect_data.py",
    "scripts/run_pipeline.py",
    "scripts/validate_project.py",
]


def source(text: str) -> list[str]:
    return text.strip("\n").splitlines(True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def embedded_bundle_b64() -> str:
    bundle = {}
    for rel_path in BUNDLE_FILES:
        path = ROOT / rel_path
        if path.exists():
            bundle[rel_path] = path.read_text(encoding="utf-8")
    payload = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")


def main() -> None:
    stats = {}
    if STATS.exists():
        with STATS.open(encoding="utf-8") as f:
            stats = json.load(f)

    total_repos = stats.get("total_repos", "N/A")
    total_tech = stats.get("total_technologies", "N/A")
    total_edges = stats.get("total_edges", "N/A")
    total_rules = stats.get("total_rules", "N/A")
    total_comm = stats.get("total_communities", "N/A")
    total_categories = stats.get("total_categories", "N/A")
    embedding_method = stats.get("embedding_method", "Run scripts/run_pipeline.py first")
    generated_at = stats.get("generated_at", "Run scripts/run_pipeline.py first")
    project_bundle_b64 = embedded_bundle_b64()

    cells = [
        md(f"""
# GitHub Repository Mining - Advanced Data Mining Project

This notebook is the complete academic artifact for the GitHub Repository Mining project. It covers the full grading rubric: real GitHub data collection, preprocessing, association rule mining, link analysis, visualization, real BERT/SentenceTransformer semantic analysis, recommendations, report conclusions, and presentation notes.

Current processed run:

| Metric | Value |
|---|---:|
| Repositories | {total_repos} |
| Technologies / topics | {total_tech} |
| Graph edges exported | {total_edges} |
| Association rules discovered | {total_rules} |
| Technology communities | {total_comm} |
| Semantic categories | {total_categories} |

Generated at: `{generated_at}`

Embedding method used in this local run: `{embedding_method}`
"""),
        md("""
## Phase 1. Secure GitHub API Runtime Setup

This notebook never stores a GitHub token in source code. At runtime, it can read `GITHUB_TOKEN` from the local environment or securely prompt for a Personal Access Token. Authentication is verified against GitHub's rate-limit endpoint before any live API collection runs.

For non-interactive validation only, set `DM_PROJECT_OFFLINE_VALIDATION=1` before executing the notebook. The notebook will then use the embedded real GitHub API dataset already bundled in the submission.
"""),
        code("""
# @title Secure GitHub API Authentication { display-mode: "form" }
import getpass
import os
import sys

import requests

GITHUB_API_BASE = "https://api.github.com"
OFFLINE_VALIDATION = os.environ.get("DM_PROJECT_OFFLINE_VALIDATION") == "1"
REQUIRE_GITHUB_AUTH = globals().get("REQUIRE_GITHUB_AUTH", not OFFLINE_VALIDATION)
GITHUB_TOKEN = (globals().get("GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
GITHUB_AUTHENTICATED = False
GITHUB_RATE_LIMIT = {}

def github_headers():
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "dm-project-notebook-miner",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

if REQUIRE_GITHUB_AUTH:
    if not GITHUB_TOKEN:
        GITHUB_TOKEN = getpass.getpass("Paste your GitHub Personal Access Token: ").strip()
    if not GITHUB_TOKEN:
        raise SystemExit("GitHub authentication failed: no token supplied.")

    response = requests.get(f"{GITHUB_API_BASE}/rate_limit", headers=github_headers(), timeout=30)
    if response.status_code != 200:
        raise SystemExit(f"GitHub authentication failed with HTTP {response.status_code}. Check the token and rerun.")

    rate_payload = response.json()
    GITHUB_RATE_LIMIT = rate_payload.get("resources", {}).get("core", {})
    GITHUB_AUTHENTICATED = True
    print("GitHub authentication verified.")
    print("Core API remaining requests:", GITHUB_RATE_LIMIT.get("remaining", "unknown"))
    print("Core API reset epoch:", GITHUB_RATE_LIMIT.get("reset", "unknown"))
else:
    print("GitHub authentication skipped for offline validation mode.")
    print("The analysis will use the embedded real GitHub API dataset already saved in the notebook.")
"""),
        md("""
## Phase 2. Interactive Live GitHub Collection GUI

This cell builds the live-collection notebook GUI for target keywords, minimum stars, preferred languages, and repository limit. When authentication is active, the same function can fetch live GitHub Search API metadata with pagination, exponential backoff, language enrichment, README retrieval for the highest-star repositories, and local reproducibility saves.
"""),
        code("""
# @title Live GitHub Search GUI and Collector { display-mode: "form" }
import base64
import json
import time
from pathlib import Path

import pandas as pd
import requests

LIVE_SEARCH_KEYWORDS = globals().get("LIVE_SEARCH_KEYWORDS", "machine learning data science")
LIVE_MIN_STARS = int(globals().get("LIVE_MIN_STARS", 500))
LIVE_LANGUAGES = globals().get("LIVE_LANGUAGES", "Python, JavaScript, TypeScript")
LIVE_MAX_REPOS = int(globals().get("LIVE_MAX_REPOS", 50))
LIVE_README_LIMIT = int(globals().get("LIVE_README_LIMIT", 10))
AUTO_RUN_GITHUB_SEARCH = globals().get("AUTO_RUN_GITHUB_SEARCH", GITHUB_AUTHENTICATED and not OFFLINE_VALIDATION)
LIVE_REPOS_DF = None

def github_api_json(url, params=None, max_attempts=5):
    for attempt in range(max_attempts):
        response = requests.get(url, params=params, headers=github_headers(), timeout=30)
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")

        if response.status_code in {403, 429}:
            if remaining == "0" and reset:
                wait = max(5, int(reset) - int(time.time()) + 3)
            else:
                wait = min(60, 2 ** attempt * 5)
            print(f"GitHub rate/throttle response {response.status_code}. Sleeping {wait}s.")
            time.sleep(wait)
            continue

        if response.status_code >= 500:
            wait = min(60, 2 ** attempt * 3)
            print(f"GitHub server response {response.status_code}. Retrying in {wait}s.")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json(), response.headers

    raise RuntimeError(f"GitHub API request failed after {max_attempts} attempts: {url}")

def fetch_repo_languages(full_name):
    try:
        payload, _ = github_api_json(f"{GITHUB_API_BASE}/repos/{full_name}/languages")
        if isinstance(payload, dict):
            return sorted(payload, key=payload.get, reverse=True)
    except Exception:
        return []
    return []

def fetch_readme_text(full_name):
    try:
        payload, _ = github_api_json(f"{GITHUB_API_BASE}/repos/{full_name}/readme")
        content = payload.get("content", "") if isinstance(payload, dict) else ""
        if not content:
            return ""
        return base64.b64decode(content).decode("utf-8", errors="ignore")[:6000]
    except Exception:
        return ""

def search_live_github_repositories(keywords, min_stars, languages, max_repos, readme_limit=10):
    if not GITHUB_AUTHENTICATED:
        raise RuntimeError("Authenticate with GitHub before live collection.")

    terms = " ".join(str(keywords or "").split())
    language_list = [item.strip() for item in str(languages or "").split(",") if item.strip()]
    queries = []
    if language_list:
        for language in language_list:
            queries.append(f"{terms} language:{language} stars:>={int(min_stars)}")
    else:
        queries.append(f"{terms} stars:>={int(min_stars)}")

    repos_by_name = {}
    for query in queries:
        page = 1
        while len(repos_by_name) < int(max_repos) and page <= 10:
            payload, headers = github_api_json(
                f"{GITHUB_API_BASE}/search/repositories",
                {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": min(100, int(max_repos)),
                    "page": page,
                },
            )
            items = payload.get("items", []) if isinstance(payload, dict) else []
            if not items:
                break
            for item in items:
                full_name = item.get("full_name")
                if not full_name or full_name in repos_by_name:
                    continue
                repos_by_name[full_name] = {
                    "name": full_name,
                    "description": item.get("description") or "",
                    "stars": item.get("stargazers_count", 0),
                    "forks": item.get("forks_count", 0),
                    "watchers": item.get("watchers_count", 0),
                    "open_issues": item.get("open_issues_count", 0),
                    "language": item.get("language") or "Unknown",
                    "languages": [],
                    "topics": item.get("topics") or [],
                    "html_url": item.get("html_url"),
                    "archived": bool(item.get("archived")),
                    "pushed_at": (item.get("pushed_at") or "")[:10],
                    "created_at": (item.get("created_at") or "")[:10],
                    "readme": "",
                    "data_origin": "github_api_live_notebook",
                }
                if len(repos_by_name) >= int(max_repos):
                    break
            page += 1

    records = sorted(repos_by_name.values(), key=lambda repo: (-int(repo["stars"]), repo["name"]))
    for repo in records[: max(0, int(readme_limit))]:
        repo["languages"] = fetch_repo_languages(repo["name"])
        repo["readme"] = fetch_readme_text(repo["name"])

    df = pd.DataFrame(records)
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(out_dir / "notebook_live_github_repos.json", orient="records", indent=1, force_ascii=False)
    df.to_csv(out_dir / "notebook_live_github_repos.csv", index=False)
    print(f"Saved live collection: {out_dir / 'notebook_live_github_repos.json'}")
    print(f"Remaining search quota header: {headers.get('X-RateLimit-Remaining', 'unknown') if records else 'unknown'}")
    return df

try:
    import ipywidgets as widgets
    from IPython.display import display

    keyword_widget = widgets.Text(value=LIVE_SEARCH_KEYWORDS, description="Keywords")
    stars_widget = widgets.IntSlider(value=LIVE_MIN_STARS, min=0, max=5000, step=100, description="Min stars")
    languages_widget = widgets.Text(value=LIVE_LANGUAGES, description="Languages")
    limit_widget = widgets.BoundedIntText(value=LIVE_MAX_REPOS, min=1, max=1000, description="Max repos")
    run_button = widgets.Button(description="Fetch GitHub Data", button_style="primary")
    output = widgets.Output()

    def run_live_collection(_):
        global LIVE_REPOS_DF
        with output:
            output.clear_output()
            LIVE_REPOS_DF = search_live_github_repositories(
                keyword_widget.value,
                stars_widget.value,
                languages_widget.value,
                limit_widget.value,
                LIVE_README_LIMIT,
            )
            display(LIVE_REPOS_DF.head(10))

    run_button.on_click(run_live_collection)
    display(widgets.VBox([keyword_widget, stars_widget, languages_widget, limit_widget, run_button, output]))
except Exception as exc:
    print("Interactive widgets are unavailable in this environment:", exc)

if AUTO_RUN_GITHUB_SEARCH:
    LIVE_REPOS_DF = search_live_github_repositories(
        LIVE_SEARCH_KEYWORDS,
        LIVE_MIN_STARS,
        LIVE_LANGUAGES,
        LIVE_MAX_REPOS,
        LIVE_README_LIMIT,
    )
    display(LIVE_REPOS_DF.head(10))
else:
    print("Live GitHub search was not auto-run.")
    print("Set AUTO_RUN_GITHUB_SEARCH=True after authentication to fetch fresh live data from this cell.")
"""),
        md("""
## Rubric Coverage

| Requirement | Marks | Where it is covered |
|---|---:|---|
| Data Collection & GUI | 1 or 2 | GitHub API collector, embedded real dataset, raw schema, dataset profile, dashboard GUI |
| Data Preprocessing | 1 or 2 | Text cleaning, topic normalization, feature engineering |
| Association Rule Mining | 2 or 3 | Apriori rules, optional FP-Growth, support, confidence, lift, stack recommendations |
| Link Analysis (PageRank/HITS) | 2 or 3 | Technology PageRank/HITS and repository influence graph |
| Visualization | 1 | EDA charts, rule plots, PageRank plots, graph-ready exports |
| Report and presentation | 1 | Final insights, limitations, presentation outline |
| BERT | 5 | Text classification, semantic embeddings, clusters, similarity recommendations |

The dashboard app in `app/` is the GUI deliverable. This notebook is the detailed explanation and analysis deliverable.
"""),
        md("""
## Required Grading Sections

This notebook explicitly includes every requested grading section:

- **Data Collection & GUI**: GitHub collection design, embedded real GitHub API dataset, raw data inspection, dashboard evidence.
- **Data Preprocessing**: cleaning, normalization, transaction construction, quality checks, schema validation.
- **Association Rule Mining**: Apriori rules, support/confidence/lift/conviction/leverage, FP-Growth comparison when available.
- **Visualization**: language/category charts, distribution plots, heatmaps, rule plots, graph charts, semantic cluster plots.
- **Link Analysis (PageRank/HITS)**: technology graph PageRank/HITS and repository influence graph PageRank/HITS.
"""),
        md("""
## 1. Project Architecture and Run Commands

The project can run in two real-data modes:

1. Real GitHub collection mode using `scripts/collect_data.py` with an optional `GITHUB_TOKEN`.
2. One-notebook Colab mode using the embedded real GitHub API dataset already saved inside this notebook.

The same analysis pipeline processes both sources.
"""),
        code("""
from pathlib import Path

PROJECT_ROOT_OVERRIDE = globals().get("PROJECT_ROOT_OVERRIDE", None)
EMBEDDED_REAL_PROJECT_B64 = globals().get("EMBEDDED_REAL_PROJECT_B64", "")

print("This notebook can run as one uploaded file in Colab.")
print("If project data is missing, it will restore embedded real GitHub API data automatically.")
print("Optional override if you have the full project folder:")
print('PROJECT_ROOT_OVERRIDE = "/content/DM_Project"')

print("\\nReal GitHub data commands:")
print("export GITHUB_TOKEN=your_token_here")
print("python3 scripts/collect_data.py --target 30000 --readme-limit 1500")
print("python3 scripts/run_pipeline.py")

print("\\nProcess collected real data:")
print("python3 scripts/run_pipeline.py")
print("python3 scripts/build_notebook.py")
print("python3 scripts/validate_project.py")
print("python3 -m http.server 8080")
print("Open http://127.0.0.1:8080/app/")
"""),
        md("""
## 2. Environment Setup

The notebook mainly reads processed JSON outputs, so it remains fast even for large datasets. For one-file Colab submission, this notebook carries the current real GitHub API dataset and dashboard files inside an embedded bundle.
"""),
        code(f"""
# @title Embedded Real GitHub API Project Bundle {{ display-mode: "form" }}
EMBEDDED_REAL_PROJECT_B64 = \"\"\"{project_bundle_b64}\"\"\"
"""),
        code("""
# @title Environment Setup { display-mode: "form" }
import base64
import json
import math
import os
import re
import subprocess
import sys
import warnings
import zipfile
import zlib
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"jupyter_client\\..*")
warnings.filterwarnings("ignore", message=r".*datetime\\.datetime\\.utcnow\\(\\) is deprecated.*")

try:
    from IPython.display import display, Markdown
except Exception:
    display = print

PROJECT_ROOT_OVERRIDE = globals().get("PROJECT_ROOT_OVERRIDE", None)

def bounded_find(base, relative_marker, max_depth=5):
    base = Path(base)
    if not base.exists():
        return None
    marker_parts = Path(relative_marker).parts
    for root, dirs, files in os.walk(base):
        root_path = Path(root)
        depth = len(root_path.relative_to(base).parts)
        if depth > max_depth:
            dirs[:] = []
            continue
        if marker_parts[-1] not in files:
            continue
        marker = root_path / marker_parts[-1]
        if str(marker).endswith(str(Path(relative_marker))):
            return marker.parents[len(marker_parts) - 1]
    return None

def unpack_project_zip_if_present():
    if not Path("/content").exists():
        return None
    for zip_path in [Path("/content/DM_Project_colab.zip"), Path("/content/DM_Project.zip"), Path("/content/project.zip")]:
        if zip_path.exists():
            target = Path("/content/DM_Project")
            if not (target / "data" / "processed" / "stats.json").exists() and not (target / "scripts" / "run_pipeline.py").exists():
                print(f"Found {zip_path}. Extracting to {target} ...")
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(target)
            nested = target / "DM_Project"
            if (nested / "data" / "processed" / "stats.json").exists() or (nested / "scripts" / "run_pipeline.py").exists():
                return nested
            if (target / "data" / "processed" / "stats.json").exists() or (target / "scripts" / "run_pipeline.py").exists():
                return target
    return None

def upload_project_zip_in_colab():
    if not Path("/content").exists():
        return None
    try:
        from google.colab import files
    except Exception:
        return None

    print("Missing project data. Please upload DM_Project_colab.zip when the upload button appears.")
    uploaded = files.upload()
    if not uploaded:
        return None
    for filename, content in uploaded.items():
        if not filename.endswith(".zip"):
            continue
        zip_path = Path("/content") / filename
        zip_path.write_bytes(content)
        print(f"Uploaded {filename}. Extracting ...")
        return unpack_project_zip_if_present()
    return None

def find_project_root():
    candidates = []
    if PROJECT_ROOT_OVERRIDE:
        candidates.append(Path(PROJECT_ROOT_OVERRIDE).expanduser())
    extracted = unpack_project_zip_if_present()
    if extracted:
        candidates.append(extracted)
    cwd = Path.cwd()
    candidates.extend([cwd, *cwd.parents])
    candidates.extend([
        Path("/content/DM_Project"),
        Path("/content/drive/MyDrive/DM_Project"),
        Path.home() / "Downloads" / "DM_Project",
    ])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "data" / "processed" / "stats.json").exists():
            return candidate
        if (candidate / "scripts" / "run_pipeline.py").exists():
            return candidate

    for base in [Path("/content"), Path("/content/drive/MyDrive")]:
        found = bounded_find(base, "data/processed/stats.json", max_depth=5)
        if found:
            return found
        found = bounded_find(base, "scripts/run_pipeline.py", max_depth=5)
        if found:
            return found
    return cwd

ROOT = find_project_root()
PROCESSED = ROOT / "data" / "processed"
RAW_PATH = ROOT / "data" / "raw" / "github_repos.json"

def refresh_project_root():
    global ROOT, PROCESSED, RAW_PATH
    new_root = find_project_root()
    if new_root != ROOT:
        ROOT = new_root
        PROCESSED = ROOT / "data" / "processed"
        RAW_PATH = ROOT / "data" / "raw" / "github_repos.json"
        print("Updated project root:", ROOT)
    return ROOT

def build_processed_data_if_possible():
    global ROOT, PROCESSED, RAW_PATH
    refresh_project_root()
    stats_path = PROCESSED / "stats.json"
    if stats_path.exists():
        return True

    pipeline = ROOT / "scripts" / "run_pipeline.py"
    if not pipeline.exists():
        return False

    print("Processed data missing. Running scripts/run_pipeline.py ...")
    subprocess.run([sys.executable, str(pipeline)], cwd=str(ROOT), check=True)
    return stats_path.exists()

def extract_embedded_real_project_if_possible():
    # Restore the real GitHub API dataset and GUI files embedded in this notebook.
    global ROOT, PROCESSED, RAW_PATH
    if (PROCESSED / "stats.json").exists():
        return True
    if not EMBEDDED_REAL_PROJECT_B64:
        return False

    target = Path("/content/DM_Project") if Path("/content").exists() else Path.cwd() / "DM_Project_real"
    ROOT = target
    PROCESSED = ROOT / "data" / "processed"
    RAW_PATH = ROOT / "data" / "raw" / "github_repos.json"
    ROOT.mkdir(parents=True, exist_ok=True)

    try:
        payload = zlib.decompress(base64.b64decode(EMBEDDED_REAL_PROJECT_B64.encode("ascii"))).decode("utf-8")
        bundle = json.loads(payload)
    except Exception as exc:
        print("Could not restore embedded real project bundle:", exc)
        return False

    for rel_path, content in bundle.items():
        out_path = ROOT / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")

    print("Embedded real GitHub API project restored:", PROCESSED / "stats.json")
    return (PROCESSED / "stats.json").exists()

def require_file(path, purpose):
    path = Path(path)
    if path.exists():
        return path
    refresh_project_root()
    retry = PROCESSED / path.name
    if retry.exists():
        return retry
    if path.name == RAW_PATH.name and RAW_PATH.exists():
        return RAW_PATH
    build_processed_data_if_possible()
    retry = PROCESSED / path.name
    if retry.exists():
        return retry
    if path.name == RAW_PATH.name and RAW_PATH.exists():
        return RAW_PATH
    if path.exists():
        return path
    extract_embedded_real_project_if_possible()
    retry = PROCESSED / path.name
    if retry.exists():
        return retry
    if path.name == RAW_PATH.name and RAW_PATH.exists():
        return RAW_PATH
    message = f\"\"\"Missing file for {purpose}: {path}

Current working directory: {Path.cwd()}
Detected project root: {ROOT}

Fix options:
1. In local Jupyter, open the notebook from the project root:
   /home/ahmed/Downloads/DM_Project

2. If scripts are present but processed data is missing, run:
   python3 scripts/collect_data.py --target 30000 --readme-limit 1500
   python3 scripts/run_pipeline.py
   python3 scripts/build_notebook.py

This notebook can restore embedded real GitHub API project data automatically. If this failed, rerun the setup cell.
\"\"\"
    raise FileNotFoundError(message)

def load_json(name):
    return json.loads(require_file(PROCESSED / name, name).read_text(encoding="utf-8"))

plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25
pd.set_option("display.max_colwidth", 90)

print("Environment ready")
print("Project root:", ROOT)
print("Processed data:", PROCESSED)
print("Raw data:", RAW_PATH)
"""),
        md("""
## 2.1 Real BERT Model Refresh

This cell guarantees that the semantic analysis is backed by a real pretrained BERT model. It runs `phob0s/bert-tiny` through the lightweight NumPy inference path in `scripts/run_pipeline.py`, embedding repository descriptions, README snippets, topics, and languages before rebuilding the processed JSON files.
"""),
        code("""
# @title Run Real BERT-Tiny Refresh { display-mode: "form" }
RUN_REAL_BERT_REFRESH = globals().get("RUN_REAL_BERT_REFRESH", True)
REAL_BERT_MODEL = globals().get("REAL_BERT_MODEL", "bert-tiny-numpy")

def ensure_real_bert_outputs():
    if not RUN_REAL_BERT_REFRESH:
        print("Real BERT refresh skipped because RUN_REAL_BERT_REFRESH=False.")
        return
    extract_embedded_real_project_if_possible()
    stats_path = PROCESSED / "stats.json"
    current_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    method = current_stats.get("embedding_method", "")
    if method.startswith("BERT/"):
        print("Real BERT outputs already available:", method)
        return

    pipeline = ROOT / "scripts" / "run_pipeline.py"
    if not pipeline.exists():
        raise FileNotFoundError(f"Cannot run real BERT refresh because {pipeline} is missing.")

    print("Running real BERT model:", REAL_BERT_MODEL)
    subprocess.run(
        [sys.executable, str(pipeline), "--semantic-model", REAL_BERT_MODEL, "--bert-max-length", "96"],
        cwd=str(ROOT),
        check=True,
    )
    refreshed = json.loads((PROCESSED / "stats.json").read_text(encoding="utf-8"))
    print("Real BERT refresh complete.")
    print("Embedding method:", refreshed["embedding_method"])

ensure_real_bert_outputs()
"""),
        md("""
## 3. Data Collection - Web Mining from GitHub

The real collector uses GitHub's REST API and shards search queries by topic, language, and star ranges because GitHub Search caps each query at 1,000 results. For every repository it stores:

- repository full name and URL
- programming language
- topics
- stars, forks, watchers, issues
- created / pushed dates
- license and size
- description and optional README snippet

The current local data source distribution is shown below.
"""),
        code("""
try:
    stats = load_json("stats.json")
except FileNotFoundError as exc:
    print("Could not load stats.json yet.")
    print("Current ROOT:", ROOT)
    print("Current PROCESSED:", PROCESSED)
    print("")
    print("In Colab: upload DM_Project_colab.zip when prompted, then rerun this cell if needed.")
    print("Or run: PROJECT_ROOT_OVERRIDE = '/content/DM_Project' and rerun the setup cell.")
    raise

display(pd.Series(stats["data_source_distribution"], name="repositories"))
print("Generated at:", stats["generated_at"])
print("Embedding method:", stats["embedding_method"])
print("Loaded from:", require_file(PROCESSED / "stats.json", "stats.json"))
"""),
        code("""
# Inspect raw file size and a raw record.
raw_path = require_file(RAW_PATH, "raw GitHub repository data")
raw_size_mb = raw_path.stat().st_size / (1024 * 1024)
print(f"Raw dataset size: {raw_size_mb:.2f} MB")

with raw_path.open(encoding="utf-8") as f:
    raw_sample = json.load(f)[0]

display(pd.DataFrame([raw_sample]).T.rename(columns={0: "sample_value"}).head(25))
"""),
        md("""
### Real GitHub Collector Design

This is the command used for real GitHub data collection. It is intentionally not executed automatically inside the notebook because it needs a token and can take time due to API rate limits.
"""),
        code("""
print("GITHUB_TOKEN=... python3 scripts/collect_data.py --target 30000 --readme-limit 1500")

# Collector implementation is in scripts/collect_data.py.
# It uses:
# - topic and language query shards
# - star buckets
# - deduplication by full repository name
# - optional README retrieval
# - rate-limit sleep handling
"""),
        md("""
## 4. Load Processed Data

The pipeline exports compact JSON files for the app and notebook:

- `repos.json`: cleaned repository records and semantic labels
- `graph_nodes.json`: technology PageRank, HITS, community scores
- `graph_edges.json`: weighted technology co-occurrence edges
- `association_rules.json`: mined Apriori rules
- `similarities.json`: nearest-neighbor repository recommendations
- `stats.json`: global dashboard metrics
"""),
        code("""
repos = pd.read_json(require_file(PROCESSED / "repos.json", "processed repositories"))
nodes = pd.read_json(require_file(PROCESSED / "graph_nodes.json", "graph nodes"))
edges = pd.read_json(require_file(PROCESSED / "graph_edges.json", "graph edges"))
rules = pd.read_json(require_file(PROCESSED / "association_rules.json", "association rules"))
similarities = load_json("similarities.json")

print(f"Repositories: {len(repos):,}")
print(f"Technologies: {len(nodes):,}")
print(f"Technology edges: {len(edges):,}")
print(f"Association rules exported: {len(rules):,}")
print(f"Similarity index entries: {len(similarities):,}")

display(repos.head(8))
"""),
        md("""
## 4.1 Automated Project Health Check

This section verifies that the notebook, processed data, raw data, dashboard, report, and scripts are all present. It is useful for submission because it proves the project is complete, not just a single isolated notebook.
"""),
        code("""
required_files = {
    "raw_data": (RAW_PATH, True),
    "stats": (PROCESSED / "stats.json", True),
    "repos": (PROCESSED / "repos.json", True),
    "graph_nodes": (PROCESSED / "graph_nodes.json", True),
    "graph_edges": (PROCESSED / "graph_edges.json", True),
    "association_rules": (PROCESSED / "association_rules.json", True),
    "similarities": (PROCESSED / "similarities.json", True),
    "dashboard_html": (ROOT / "app" / "index.html", False),
    "dashboard_js": (ROOT / "app" / "js" / "app.js", False),
    "dashboard_css": (ROOT / "app" / "css" / "styles.css", False),
    "pipeline": (ROOT / "scripts" / "run_pipeline.py", False),
    "collector": (ROOT / "scripts" / "collect_data.py", False),
    "report": (ROOT / "outputs" / "project_report.md", False),
    "presentation": (ROOT / "outputs" / "presentation_outline.md", False),
}

health = []
for name, (path, required) in required_files.items():
    health.append({
        "artifact": name,
        "required_for_notebook": required,
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else 0,
        "path": str(path),
    })

health_df = pd.DataFrame(health)
display(health_df)

missing_required = health_df[health_df["required_for_notebook"] & ~health_df["exists"]]
assert missing_required.empty, "Some core notebook data artifacts are missing."
assert len(repos) == stats["total_repos"], "repos.json row count does not match stats.json"
assert len(nodes) == stats["total_technologies"], "graph_nodes.json count does not match stats.json"
assert len(edges) == stats["total_edges"], "graph_edges.json count does not match stats.json"
if (~health_df["exists"]).any():
    print("Core notebook data check passed. Some full-project files are optional in notebook-only Colab mode.")
else:
    print("Full project health check passed.")
"""),
        md("""
## 4.2 Schema Validation

The following checks verify that the processed data contains the columns needed by every rubric item and the dashboard.
"""),
        code("""
required_repo_columns = {
    "name", "stars", "forks", "language", "topics", "description",
    "tech_stack", "created_year", "era", "popularity",
    "bert_category", "bert_confidence", "cluster_id", "pca_x", "pca_y", "html_url"
}
required_node_columns = {"id", "pagerank", "hub", "authority", "community", "count"}
required_rule_columns = {"antecedent", "consequent", "support", "support_count", "confidence", "lift"}

schema_checks = pd.DataFrame([
    {
        "table": "repos",
        "required_columns": len(required_repo_columns),
        "missing": sorted(required_repo_columns - set(repos.columns)),
    },
    {
        "table": "graph_nodes",
        "required_columns": len(required_node_columns),
        "missing": sorted(required_node_columns - set(nodes.columns)),
    },
    {
        "table": "association_rules",
        "required_columns": len(required_rule_columns),
        "missing": sorted(required_rule_columns - set(rules.columns)),
    },
])
display(schema_checks)
assert schema_checks["missing"].apply(len).sum() == 0, "Schema validation failed."
print("Schema validation passed.")
"""),
        md("""
## 5. Data Understanding

We first profile repository metadata to understand language distribution, repository popularity, project age, and category balance.
"""),
        code("""
summary = pd.DataFrame({
    "metric": [
        "repositories",
        "technologies",
        "technology_edges",
        "communities",
        "association_rules",
        "semantic_categories",
        "total_stars",
        "mean_stars",
        "median_stars",
        "max_stars",
        "total_forks",
    ],
    "value": [
        stats["total_repos"],
        stats["total_technologies"],
        stats["total_edges"],
        stats["total_communities"],
        stats["total_rules"],
        stats["total_categories"],
        stats["stars"]["total"],
        stats["stars"]["mean"],
        stats["stars"]["median"],
        stats["stars"]["max"],
        stats["forks"]["total"],
    ],
})
display(summary)
"""),
        code("""
display(repos[["stars", "forks", "watchers", "open_issues", "created_year", "bert_confidence"]].describe().T)

missing = repos.isna().mean().sort_values(ascending=False).head(12)
display(pd.DataFrame({"missing_ratio": missing}))
"""),
        code("""
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

pd.Series(stats["language_distribution"]).head(15).sort_values().plot(
    kind="barh", ax=axes[0, 0], title="Top programming languages"
)
pd.Series(stats["category_distribution"]).sort_values().plot(
    kind="barh", ax=axes[0, 1], title="BERT/semantic category distribution"
)
pd.Series(stats["era_distribution"]).plot(
    kind="bar", ax=axes[1, 0], title="Repository creation era", rot=0
)
pd.Series(stats["popularity_distribution"]).plot(
    kind="bar", ax=axes[1, 1], title="Popularity buckets", rot=0
)

plt.tight_layout()
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

repos["stars"].clip(upper=repos["stars"].quantile(0.99)).plot(
    kind="hist", bins=40, ax=axes[0], title="Stars distribution clipped at 99th percentile"
)
repos["forks"].clip(upper=repos["forks"].quantile(0.99)).plot(
    kind="hist", bins=40, ax=axes[1], title="Forks distribution clipped at 99th percentile"
)

axes[0].set_xlabel("Stars")
axes[1].set_xlabel("Forks")
plt.tight_layout()
"""),
        md("""
## 5.1 Data Quality and Cleaning Evidence

A strong data mining project should show that the collected data is usable. These checks measure duplicate names, valid technology stacks, valid numeric values, and text coverage.
"""),
        code("""
quality_metrics = {
    "duplicate_repository_names": int(repos["name"].duplicated().sum()),
    "missing_descriptions": int((repos["description"].fillna("").str.len() == 0).sum()),
    "empty_tech_stacks": int(repos["tech_stack"].apply(lambda x: len(x) if isinstance(x, list) else 0).eq(0).sum()),
    "negative_stars": int((repos["stars"] < 0).sum()),
    "negative_forks": int((repos["forks"] < 0).sum()),
    "invalid_confidence": int(((repos["bert_confidence"] < 0) | (repos["bert_confidence"] > 1)).sum()),
    "unique_languages": int(repos["language"].nunique()),
    "unique_categories": int(repos["bert_category"].nunique()),
}

quality_df = pd.DataFrame(
    [{"metric": key, "value": value} for key, value in quality_metrics.items()]
)
display(quality_df)

assert quality_metrics["duplicate_repository_names"] == 0
assert quality_metrics["empty_tech_stacks"] == 0
assert quality_metrics["negative_stars"] == 0
assert quality_metrics["negative_forks"] == 0
assert quality_metrics["invalid_confidence"] == 0
print("Data quality checks passed.")
"""),
        code("""
category_language = pd.crosstab(repos["bert_category"], repos["language"])
top_langs = repos["language"].value_counts().head(10).index
heatmap_data = category_language[top_langs]

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heatmap_data.values, aspect="auto", cmap="YlGnBu")
ax.set_xticks(range(len(top_langs)))
ax.set_xticklabels(top_langs, rotation=45, ha="right")
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)
ax.set_title("Category x Language heatmap")
plt.colorbar(im, ax=ax, label="Repository count")
plt.tight_layout()
"""),
        md("""
## 6. Data Preprocessing

Preprocessing converts raw GitHub metadata into mining-ready features:

1. Clean description and README text.
2. Normalize topic aliases such as `ml` to `machine-learning`.
3. Remove weak generic tags such as `tool`, `framework`, and `awesome`.
4. Merge topics with programming language to form a transaction-style `tech_stack`.
5. Create year, era, popularity, and semantic text features.

The production implementation is in `scripts/run_pipeline.py`.
"""),
        code("""
STOPWORDS = {
    "awesome", "list", "tool", "tools", "resource", "resources",
    "tutorial", "book", "course", "open-source", "hacktoberfest",
    "cli", "api", "sdk", "library", "framework", "platform",
    "cloud-native", "serverless", "real-time", "high-performance",
    "scalable", "production-ready", "enterprise"
}

ALIASES = {
    "ai": "artificial-intelligence",
    "ml": "machine-learning",
    "dl": "deep-learning",
    "nlp": "natural-language-processing",
    "cv": "computer-vision",
    "llm": "large-language-model",
    "genai": "generative-ai",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
}

def clean_text(value, limit=None):
    text = "" if value is None else str(value)
    text = re.sub(r"```[\\s\\S]*?```", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\\s\\+\\-\\#\\.]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip().lower()
    return text[:limit] if limit else text

def normalize_topic(topic):
    value = str(topic or "").lower().strip().replace("_", "-")
    value = re.sub(r"\\s+", "-", value)
    return ALIASES.get(value, value)

def normalize_topics(topics):
    output = []
    for topic in topics if isinstance(topics, list) else []:
        topic = normalize_topic(topic)
        if topic and topic not in STOPWORDS and len(topic) > 2:
            output.append(topic)
    return sorted(set(output))

print(clean_text("Fast <b>React</b> framework for AI apps! https://example.com"))
print(normalize_topics(["ML", "awesome", "k8s", "React Native"]))
"""),
        code("""
display(repos[["name", "language", "topics", "tech_stack", "era", "popularity"]].sample(10, random_state=7))
"""),
        md("""
## 7. Technology Transaction Dataset

For association rule mining, every repository is treated as one transaction:

`transaction(repo) = normalized_topics(repo) union {primary_language(repo)}`

This lets us discover technology stacks that frequently appear together.
"""),
        code("""
transactions = repos["tech_stack"].apply(lambda x: x if isinstance(x, list) else []).tolist()
transaction_lengths = pd.Series([len(t) for t in transactions])

print(f"Total transactions: {len(transactions):,}")
print(f"Average items per transaction: {transaction_lengths.mean():.2f}")

item_counts = Counter(item for tx in transactions for item in tx)
top_items = pd.DataFrame(item_counts.most_common(20), columns=["technology", "count"])
top_items["support"] = top_items["count"] / len(transactions)
display(top_items)

transaction_lengths.plot(kind="hist", bins=20, title="Items per repository transaction", figsize=(7, 4))
plt.xlabel("Number of technologies in transaction")
plt.tight_layout()
"""),
        md("""
## 8. Association Rule Mining - Apriori

For a rule `A -> B`:

- Support = count(A union B) / N
- Confidence = support(A union B) / support(A)
- Lift = confidence(A -> B) / support(B)
- Conviction measures implication strength
- Leverage measures how much more often A and B occur together than expected by independence

The exported rules are mined by the pipeline and sorted by lift, confidence, and support count.
"""),
        code("""
display(rules[[
    "antecedent", "consequent", "support", "support_count",
    "confidence", "lift", "conviction", "leverage"
]].head(20))

display(pd.Series(stats["frequent_itemsets"], name="count"))
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

scatter = axes[0].scatter(
    rules["support"], rules["confidence"],
    s=rules["lift"].clip(upper=25) * 10,
    c=rules["lift"], cmap="viridis", alpha=0.7
)
axes[0].set_title("Association rules: support vs confidence")
axes[0].set_xlabel("Support")
axes[0].set_ylabel("Confidence")
plt.colorbar(scatter, ax=axes[0], label="Lift")

top_rule_labels = (
    rules.head(12)["antecedent"].apply(lambda x: " + ".join(x))
    + " -> "
    + rules.head(12)["consequent"].apply(lambda x: " + ".join(x))
)
axes[1].barh(range(len(top_rule_labels)), rules.head(12)["lift"])
axes[1].set_yticks(range(len(top_rule_labels)))
axes[1].set_yticklabels(top_rule_labels, fontsize=8)
axes[1].invert_yaxis()
axes[1].set_title("Top association rules by lift")
axes[1].set_xlabel("Lift")

plt.tight_layout()
"""),
        md("""
## 8.1 Rule Quality Interpretation

High lift alone is not enough. A useful rule should also have enough support count to matter. The table below groups rules into practical decision-support tiers.
"""),
        code("""
def rule_tier(row):
    if row["lift"] >= 10 and row["confidence"] >= 0.8 and row["support_count"] >= stats["frequent_itemsets"]["min_support_count"]:
        return "strong stack signal"
    if row["lift"] >= 3 and row["confidence"] >= 0.5:
        return "useful recommendation"
    return "exploratory"

rules = rules.copy()
rules["tier"] = rules.apply(rule_tier, axis=1)
display(rules["tier"].value_counts().rename_axis("tier").reset_index(name="rules"))

display(rules[[
    "antecedent", "consequent", "support_count", "confidence", "lift", "tier"
]].head(25))
"""),
        code("""
def format_rule_side(value):
    if isinstance(value, (list, tuple, set)):
        return " + ".join(map(str, value))
    if pd.isna(value):
        return ""
    return str(value)

if "tier" not in rules.columns:
    rules = rules.copy()
    rules["tier"] = rules.apply(rule_tier, axis=1)

plot_candidates = rules[rules["tier"] == "strong stack signal"].copy()
plot_title = "Strong stack rules by repository support count"

if plot_candidates.empty:
    plot_candidates = rules[rules["tier"] == "useful recommendation"].copy()
    plot_title = "Useful stack recommendations by repository support count"

if plot_candidates.empty:
    plot_candidates = rules.copy()
    plot_title = "Top association rules by repository support count"

top_stack_rules = (
    plot_candidates
    .sort_values(["support_count", "lift", "confidence"], ascending=False)
    .head(15)
    .copy()
)
top_stack_rules["rule"] = (
    top_stack_rules["antecedent"].apply(format_rule_side)
    + " -> "
    + top_stack_rules["consequent"].apply(format_rule_side)
)

display(top_stack_rules[["rule", "support_count", "confidence", "lift", "tier"]])

fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top_stack_rules))))
if top_stack_rules.empty:
    ax.text(0.5, 0.5, "No association rules available to plot", ha="center", va="center")
    ax.set_axis_off()
else:
    plot_df = top_stack_rules.iloc[::-1]
    ax.barh(plot_df["rule"], plot_df["support_count"])
    ax.set_title(plot_title)
    ax.set_xlabel("Repository support count")
    ax.set_ylabel("")
plt.tight_layout()
"""),
        code("""
def recommend_stack(seed_terms, rules_df, top_n=10):
    seed = set(seed_terms)
    candidates = []
    for _, row in rules_df.iterrows():
        antecedent = set(row["antecedent"])
        consequent = set(row["consequent"])
        if seed.issubset(antecedent):
            candidates.append({
                "given": " + ".join(row["antecedent"]),
                "recommend": " + ".join(row["consequent"]),
                "support_count": row.get("support_count", 0),
                "confidence": row["confidence"],
                "lift": row["lift"],
            })
    if not candidates:
        return pd.DataFrame(columns=["given", "recommend", "support_count", "confidence", "lift"])
    return pd.DataFrame(candidates).sort_values(["lift", "confidence"], ascending=False).head(top_n)

seed_a = rules.iloc[0]["antecedent"][:1]
seed_b = rules.iloc[min(10, len(rules) - 1)]["antecedent"][:1]

print("Seed stack:", seed_a)
display(recommend_stack(seed_a, rules))

print("Seed stack:", seed_b)
display(recommend_stack(seed_b, rules))
"""),
        md("""
### Optional FP-Growth Comparison

Apriori is implemented in the pipeline because it is clear and easy to explain. FP-Growth is an alternative that can be faster for very large transaction datasets. The optional cell below runs only if `mlxtend` is installed.
"""),
        code("""
try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import fpgrowth

    sample_transactions = transactions[:5000]
    encoder = TransactionEncoder()
    one_hot = encoder.fit(sample_transactions).transform(sample_transactions)
    tx_df = pd.DataFrame(one_hot, columns=encoder.columns_)
    fp = fpgrowth(tx_df, min_support=0.02, use_colnames=True).sort_values("support", ascending=False)
    display(fp.head(15))
except Exception as exc:
    print("FP-Growth comparison skipped:", exc)
"""),
        md("""
## 9. Graph Construction

The project uses graph mining in two ways:

1. Technology co-occurrence graph: nodes are technologies/topics/languages, weighted edges mean the technologies appear in the same repositories.
2. Repository similarity graph: nodes are repositories, weighted edges mean repositories share technologies.

The dashboard uses the technology graph because it is readable and directly supports technology trend decisions. The repository graph below is included to satisfy the influential repository interpretation of the task.
"""),
        code("""
display(nodes[["id", "pagerank", "authority", "hub", "community", "tribe", "count"]].head(20))
display(edges.head(10))
"""),
        md("""
## 9.1 Strongest Technology Relationships

The highest-weight edges reveal technologies that repeatedly occur in the same repositories. These relationships are useful for stack suggestions and ecosystem understanding.
"""),
        code("""
top_edges = edges.sort_values("weight", ascending=False).head(25).copy()
top_edges["relationship"] = top_edges["source"] + " + " + top_edges["target"]
display(top_edges[["relationship", "weight"]])

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_edges["relationship"].iloc[::-1], top_edges["weight"].iloc[::-1])
ax.set_title("Strongest technology co-occurrence edges")
ax.set_xlabel("Shared repository count")
plt.tight_layout()
"""),
        code("""
community_summary = (
    nodes.groupby(["community", "tribe"])
    .agg(
        technologies=("id", "count"),
        total_repo_mentions=("count", "sum"),
        max_pagerank=("pagerank", "max"),
    )
    .sort_values(["total_repo_mentions", "max_pagerank"], ascending=False)
    .reset_index()
)
display(community_summary.head(20))
"""),
        code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

nodes.head(15).sort_values("pagerank").plot(
    kind="barh", x="id", y="pagerank", ax=axes[0], legend=False, title="Top PageRank technologies"
)
nodes.sort_values("authority", ascending=False).head(15).sort_values("authority").plot(
    kind="barh", x="id", y="authority", ax=axes[1], legend=False, title="Top HITS authorities"
)
nodes.sort_values("hub", ascending=False).head(15).sort_values("hub").plot(
    kind="barh", x="id", y="hub", ax=axes[2], legend=False, title="Top HITS hubs"
)

plt.tight_layout()
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

edges["weight"].plot(kind="hist", bins=30, ax=axes[0], title="Technology edge weight distribution")
nodes["community"].value_counts().sort_index().plot(kind="bar", ax=axes[1], title="Technology community sizes")
axes[0].set_xlabel("Co-occurrence weight")
axes[1].set_xlabel("Community ID")
plt.tight_layout()
"""),
        md("""
## 10. Link Analysis - PageRank and HITS

PageRank ranks nodes that are connected to other important nodes. HITS separates nodes into:

- Hubs: nodes that connect to many authoritative nodes
- Authorities: nodes that are pointed to by strong hubs

For an undirected co-occurrence graph, hub and authority values can be similar, but they are still useful for ranking central ecosystem technologies.
"""),
        code("""
def weighted_pagerank(node_count, weighted_edges, damping=0.85, iterations=60):
    adj = defaultdict(dict)
    degree = defaultdict(float)
    for src, dst, weight in weighted_edges:
        adj[src][dst] = adj[src].get(dst, 0.0) + weight
        adj[dst][src] = adj[dst].get(src, 0.0) + weight
        degree[src] += weight
        degree[dst] += weight

    ranks = np.full(node_count, 1.0 / node_count)
    for _ in range(iterations):
        new = np.full(node_count, (1 - damping) / node_count)
        dangling = ranks[[i for i in range(node_count) if degree.get(i, 0) == 0]].sum()
        new += damping * dangling / node_count
        for node in range(node_count):
            for nb, weight in adj.get(node, {}).items():
                if degree[nb]:
                    new[node] += damping * ranks[nb] * weight / degree[nb]
        ranks = new / new.sum()
    return ranks

def weighted_hits(node_count, weighted_edges, iterations=40):
    adj = defaultdict(dict)
    for src, dst, weight in weighted_edges:
        adj[src][dst] = adj[src].get(dst, 0.0) + weight
        adj[dst][src] = adj[dst].get(src, 0.0) + weight

    hubs = np.ones(node_count)
    auth = np.ones(node_count)
    for _ in range(iterations):
        auth_new = np.zeros(node_count)
        for node in range(node_count):
            auth_new[node] = sum(hubs[nb] * weight for nb, weight in adj.get(node, {}).items())
        auth = auth_new / (np.linalg.norm(auth_new) or 1)

        hubs_new = np.zeros(node_count)
        for node in range(node_count):
            hubs_new[node] = sum(auth[nb] * weight for nb, weight in adj.get(node, {}).items())
        hubs = hubs_new / (np.linalg.norm(hubs_new) or 1)
    return hubs, auth
"""),
        md("""
### Repository Influence Graph

To avoid a dense graph with millions of edges, this notebook builds a repository graph on the most starred repositories and caps each technology bucket. This keeps the graph interpretable while still using shared technologies as the relationship signal.
"""),
        code("""
top_repo_graph = repos.nlargest(2500, "stars").reset_index(drop=True)
repo_stacks = top_repo_graph["tech_stack"].apply(lambda x: x if isinstance(x, list) else []).tolist()

tech_to_repo_ids = defaultdict(list)
for idx, stack in enumerate(repo_stacks):
    for tech in stack:
        tech_to_repo_ids[tech].append(idx)

pair_counts = Counter()
for tech, ids in tech_to_repo_ids.items():
    ids = sorted(ids, key=lambda i: -top_repo_graph.loc[i, "stars"])[:250]
    for a, b in combinations(ids, 2):
        pair_counts[(a, b)] += 1

repo_weighted_edges = [
    (a, b, weight)
    for (a, b), weight in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))[:5000]
]

repo_pr = weighted_pagerank(len(top_repo_graph), repo_weighted_edges)
repo_hub, repo_auth = weighted_hits(len(top_repo_graph), repo_weighted_edges)

repo_influence = top_repo_graph[["name", "stars", "forks", "language", "bert_category"]].copy()
repo_influence["repo_pagerank"] = repo_pr
repo_influence["repo_hub"] = repo_hub
repo_influence["repo_authority"] = repo_auth
repo_influence = repo_influence.sort_values("repo_pagerank", ascending=False)

display(repo_influence.head(20))
print(f"Repository graph nodes: {len(top_repo_graph):,}")
print(f"Repository graph edges: {len(repo_weighted_edges):,}")
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

repo_influence.head(15).sort_values("repo_pagerank").plot(
    kind="barh", x="name", y="repo_pagerank", ax=axes[0], legend=False,
    title="Influential repositories by PageRank"
)
repo_influence.head(50).plot(
    kind="scatter", x="repo_authority", y="repo_hub",
    s=repo_influence.head(50)["stars"].clip(upper=50000) / 800,
    ax=axes[1], title="Repository HITS: authority vs hub"
)

plt.tight_layout()
"""),
        md("""
## 11. BERT-based Text Classification

The pipeline uses a real pretrained BERT-Tiny model (`phob0s/bert-tiny`) to embed repository descriptions, README snippets, topics, and languages. The BERT embeddings drive category classification, clustering, and similarity recommendations. A SentenceTransformer model can still be supplied with `--semantic-model`, but the default notebook path runs real BERT without requiring PyTorch.
"""),
        code("""
print("Semantic embedding method used by the latest pipeline run:")
print(stats["embedding_method"])

display(repos[["name", "description", "bert_category", "bert_confidence", "cluster_id"]].head(12))
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

repos["bert_confidence"].plot(kind="hist", bins=15, ax=axes[0], title="BERT/semantic confidence")
axes[0].set_xlabel("Confidence")

pd.Series(stats["cluster_distribution"]).sort_index(key=lambda s: s.astype(int)).plot(
    kind="bar", ax=axes[1], title="Embedding cluster distribution", rot=0
)
axes[1].set_xlabel("Cluster ID")

plt.tight_layout()
"""),
        code("""
# Real BERT-Tiny embedding demonstration using the same NumPy inference path as the pipeline.
try:
    import importlib.util

    sample_texts = (
        repos["description"].fillna("")
        + " "
        + repos["tech_stack"].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    ).head(24).tolist()

    pipeline_path = ROOT / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", pipeline_path)
    run_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_pipeline)

    model_dir = ROOT / "models" / "bert-tiny"
    run_pipeline.ensure_bert_tiny_files(model_dir)
    model = run_pipeline.NumpyBertTiny(model_dir, max_length=96)
    sample_embeddings = model.encode(sample_texts, batch_size=12)
    print("BERT embedding sample shape:", sample_embeddings.shape)
    print("Model:", run_pipeline.BERT_TINY_REPO)
except Exception as exc:
    print("Real BERT-Tiny demo skipped in this environment:", exc)
    print("The latest processed method is:", stats["embedding_method"])
"""),
        md("""
## 11.1 Classification Evaluation When Labels Exist

The real GitHub API does not provide ground-truth topic labels. If labels are available in a future curated dataset, this section computes a sanity-check evaluation; for pure GitHub API data, it reports that no ground-truth labels are present.
"""),
        code("""
with require_file(RAW_PATH, "raw GitHub repository data").open(encoding="utf-8") as f:
    raw_records = json.load(f)

raw_labels = pd.DataFrame([
    {"name": item.get("name"), "source_category": item.get("category")}
    for item in raw_records
    if item.get("name") and item.get("category")
])

source_to_semantic = {
    "ai_ml": "AI/ML Infrastructure",
    "web_frontend": "Web Development",
    "web_backend": "Backend & APIs",
    "data_science": "Data Science",
    "devops": "DevOps & Cloud",
    "mobile": "Mobile Development",
    "security": "Security",
    "blockchain": "Blockchain & Web3",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
    "genai": "Generative AI",
    "database": "Database & Storage",
    "gamedev": "Game Development",
}

if raw_labels.empty:
    print("No generator labels available. This is expected for real GitHub API data.")
else:
    eval_df = repos.merge(raw_labels, on="name", how="inner")
    eval_df["expected_category"] = eval_df["source_category"].map(source_to_semantic)
    eval_df = eval_df.dropna(subset=["expected_category"])
    eval_df["correct"] = eval_df["expected_category"].eq(eval_df["bert_category"])
    accuracy = eval_df["correct"].mean()
    print(f"Label sanity-check accuracy: {accuracy:.3f} on {len(eval_df):,} generated-label repositories")
    display(pd.crosstab(eval_df["expected_category"], eval_df["bert_category"]).head(20))
"""),
        md("""
## 12. Semantic Similarity and Recommendation

Repository recommendations are based on nearest-neighbor cosine similarity over the semantic embedding space. This supports the task requirement to recommend similar projects or technologies.
"""),
        code("""
selected_repo = repos.iloc[0]["name"]
print("Selected repository:", selected_repo)

recommended = pd.DataFrame(
    similarities[selected_repo],
    columns=["similar_repository", "cosine_similarity"]
)
recommended = recommended.merge(
    repos[["name", "language", "bert_category", "stars"]],
    left_on="similar_repository",
    right_on="name",
    how="left"
).drop(columns=["name"])

display(recommended)
"""),
        code("""
def show_recommendations(repo_name, top_n=8):
    if repo_name not in similarities:
        return pd.DataFrame()
    out = pd.DataFrame(similarities[repo_name][:top_n], columns=["similar_repository", "similarity"])
    return out.merge(
        repos[["name", "language", "bert_category", "stars", "description"]],
        left_on="similar_repository",
        right_on="name",
        how="left"
    ).drop(columns=["name"])

example_repo = repos.sample(1, random_state=12).iloc[0]["name"]
print("Example:", example_repo)
display(show_recommendations(example_repo))
"""),
        md("""
## 13. Trend Analysis and Decision Support

The final objective is not only to run algorithms, but to interpret the results. The cells below identify popular technologies, strong stack combinations, and high-level recommendations.
"""),
        code("""
trend_data = stats.get("trends", {})
trend_df = pd.DataFrame(trend_data).fillna(0).sort_index()
display(trend_df.tail(10))

if not trend_df.empty:
    trend_df.plot(figsize=(14, 5), title="Technology emergence trends")
    plt.xlabel("Created year")
    plt.ylabel("Repository count")
    plt.tight_layout()
"""),
        code("""
top_language = pd.Series(stats["language_distribution"]).idxmax()
top_category = pd.Series(stats["category_distribution"]).idxmax()
top_tech = nodes.iloc[0]["id"]
top_rule = rules.iloc[0]

decision_points = [
    f"Most common language: {top_language}",
    f"Largest semantic category: {top_category}",
    f"Most central technology by PageRank: {top_tech}",
    f"Strongest stack rule: {' + '.join(top_rule['antecedent'])} -> {' + '.join(top_rule['consequent'])}",
    "Use association rules to recommend complementary technologies.",
    "Use PageRank/HITS to prioritize central ecosystem skills.",
    "Use semantic similarity to find comparable projects for learning or reuse.",
]

for point in decision_points:
    print("-", point)
"""),
        md("""
## 14. Dashboard GUI

The dashboard is a complete GUI for the project. It includes:

- KPI overview
- filters for category, language, popularity, and search
- language/category/era charts
- technology trend chart
- PageRank and HITS charts
- D3 co-occurrence network graph
- association rule visualization
- BERT classification diagnostics
- PCA cluster map
- recommendation engine
- paginated large-data repository explorer
"""),
        code("""
print("Start the dashboard from the project root:")
print("python3 -m http.server 8080")
print("Open: http://127.0.0.1:8080/app/")
"""),
        md("""
## 14.1 Grading Evidence Matrix

This matrix maps every requested rubric item to concrete files and computed outputs.
"""),
        code("""
grading_evidence = pd.DataFrame([
    {
        "rubric_item": "Data Collection and GUI",
        "evidence": "GitHub API collector, large raw JSON dataset, dashboard app",
        "file_or_output": "scripts/collect_data.py, data/raw/github_repos.json, app/index.html",
        "status": "complete",
    },
    {
        "rubric_item": "Data Preprocessing",
        "evidence": "cleaned text, normalized topics, tech_stack, era/popularity features",
        "file_or_output": "scripts/run_pipeline.py, data/processed/repos.json",
        "status": "complete",
    },
    {
        "rubric_item": "Association Rule Mining",
        "evidence": f"{stats['total_rules']:,} Apriori rules with support/confidence/lift",
        "file_or_output": "data/processed/association_rules.json",
        "status": "complete",
    },
    {
        "rubric_item": "Link Analysis",
        "evidence": f"PageRank/HITS over {stats['total_technologies']:,} technology nodes and repository influence graph",
        "file_or_output": "data/processed/graph_nodes.json",
        "status": "complete",
    },
    {
        "rubric_item": "Visualization",
        "evidence": "Notebook plots, Chart.js charts, D3 network, PCA projection",
        "file_or_output": "github_mining_analysis.ipynb, app/",
        "status": "complete",
    },
    {
        "rubric_item": "BERT",
        "evidence": "Real pretrained BERT-Tiny embeddings, classification, clustering, recommendations",
        "file_or_output": "scripts/run_pipeline.py, data/processed/similarities.json",
        "status": "complete",
    },
    {
        "rubric_item": "Report and Presentation",
        "evidence": "Markdown report and presentation outline",
        "file_or_output": "outputs/project_report.md, outputs/presentation_outline.md",
        "status": "complete",
    },
])

display(grading_evidence)
"""),
        md("""
## 15. Report-Ready Results

These are the main results to include in the written report and slides.
"""),
        code("""
print("Main metrics")
print("------------")
print(f"Repositories: {stats['total_repos']:,}")
print(f"Technologies: {stats['total_technologies']:,}")
print(f"Graph edges: {stats['total_edges']:,}")
print(f"Communities: {stats['total_communities']:,}")
print(f"Rules: {stats['total_rules']:,}")
print(f"Categories: {stats['total_categories']:,}")
print(f"Average semantic confidence: {stats['average_bert_confidence']}")

print("\\nKey insights")
print("------------")
for insight in stats.get("insights", []):
    print("-", insight)
"""),
        md("""
## 16. Limitations and Future Work

Limitations:

- GitHub API collection depends on token quota and rate limits.
- GitHub API collection without a token is smaller because public unauthenticated requests are rate-limited.
- The default real BERT-Tiny path downloads a compact Hugging Face model; larger SentenceTransformer models can improve embedding quality when compute is available.
- Repository relationships based on shared technologies can be dense, so the notebook caps graph edges for interpretability.

Future work:

- Collect real data nightly with GitHub Actions.
- Add temporal trend forecasting.
- Add community detection with Louvain or Leiden.
- Add a real vector database for semantic search.
- Use full README embeddings with a fine-tuned BERT classifier.
"""),
        md("""
## 17. Presentation Outline

1. Problem statement and project objective
2. Data collection from GitHub
3. Dataset profile and preprocessing
4. Association rule mining with Apriori
5. Link analysis with PageRank and HITS
6. BERT-based classification and semantic similarity
7. Dashboard demonstration
8. Key insights and decision support
9. Limitations and future improvements
10. Conclusion
"""),
        md("""
## 18. Conclusion

This project provides an end-to-end data mining workflow for GitHub repository mining. It combines web mining, transaction mining, graph mining, semantic text mining, visualization, and recommendations into a single reproducible project with both a notebook and an interactive app.
"""),
    ]

    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"cell-{index:03d}"

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with NOTEBOOK.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Full notebook written to {NOTEBOOK}")
    print(f"Cells: {len(cells)}")


if __name__ == "__main__":
    main()
