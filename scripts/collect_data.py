#!/usr/bin/env python3
"""Collect real GitHub repository data with the GitHub REST API.

Recommended:
    GITHUB_TOKEN=ghp_xxx python3 scripts/collect_data.py --target 30000 --readme-limit 1500

The GitHub Search API has rate limits and caps each query at 1,000 results, so
this script shards collection across topics and star ranges.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "github_repos.json"
API = "https://api.github.com"

TOPICS = [
    "machine-learning",
    "deep-learning",
    "artificial-intelligence",
    "nlp",
    "computer-vision",
    "generative-ai",
    "llm",
    "data-science",
    "react",
    "vue",
    "angular",
    "nextjs",
    "typescript",
    "javascript",
    "python",
    "go",
    "rust",
    "java",
    "flutter",
    "react-native",
    "android",
    "ios",
    "devops",
    "docker",
    "kubernetes",
    "terraform",
    "security",
    "blockchain",
    "web3",
    "database",
    "postgresql",
    "redis",
    "game-development",
]

STAR_BUCKETS = [
    "stars:100..499",
    "stars:500..999",
    "stars:1000..4999",
    "stars:5000..19999",
    "stars:>=20000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect real GitHub repositories.")
    parser.add_argument("--target", type=int, default=30000, help="Target number of unique repositories.")
    parser.add_argument("--per-query", type=int, default=900, help="Maximum repositories per query shard.")
    parser.add_argument("--readme-limit", type=int, default=1500,
                        help="Fetch README snippets for this many highest-starred repositories. Use 0 to skip.")
    parser.add_argument("--readme-save-every", type=int, default=25,
                        help="Save progress after this many README fetch attempts.")
    parser.add_argument("--out", default=str(OUT), help="Output JSON path.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay between API calls.")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"), help="GitHub token, or set GITHUB_TOKEN.")
    parser.add_argument("--append", action="store_true", help="Append/deduplicate with an existing output file.")
    parser.add_argument("--topic", action="append", help="Restrict collection to one or more topics.")
    return parser.parse_args()


def headers(token: str | None) -> dict[str, str]:
    base = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "dm-project-github-miner",
    }
    if token:
        base["Authorization"] = f"Bearer {token}"
    return base


def request_json(url: str, params: dict[str, Any] | None, token: str | None, sleep: float) -> Any:
    while True:
        response = requests.get(url, params=params, headers=headers(token), timeout=30)
        if response.status_code in {403, 429}:
            reset = response.headers.get("X-RateLimit-Reset")
            remaining = response.headers.get("X-RateLimit-Remaining")
            if remaining == "0" and reset:
                wait = max(5, int(reset) - int(time.time()) + 3)
                print(f"Rate limit reached. Sleeping {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            print(f"GitHub throttled request ({response.status_code}). Sleeping 60s...", file=sys.stderr)
            time.sleep(60)
            continue
        if response.status_code == 422:
            return {"items": []}
        response.raise_for_status()
        if sleep:
            time.sleep(sleep)
        return response.json()


def request_readme(full_name: str, token: str | None, sleep: float) -> str:
    url = f"{API}/repos/{full_name}/readme"
    try:
        payload = request_json(url, None, token, sleep)
    except Exception:
        return ""
    content = payload.get("content", "") if isinstance(payload, dict) else ""
    if not content:
        return ""
    try:
        return base64.b64decode(content).decode("utf-8", errors="ignore")[:4000]
    except Exception:
        return ""


def repo_record(item: dict[str, Any]) -> dict[str, Any]:
    license_info = item.get("license") or {}
    owner = item.get("owner") or {}
    return {
        "name": item.get("full_name"),
        "stars": item.get("stargazers_count", 0),
        "forks": item.get("forks_count", 0),
        "watchers": item.get("watchers_count", 0),
        "open_issues": item.get("open_issues_count", 0),
        "language": item.get("language") or "Unknown",
        "topics": item.get("topics") or [],
        "description": item.get("description") or "",
        "readme": "",
        "created_at": (item.get("created_at") or "")[:10],
        "updated_at": (item.get("updated_at") or "")[:10],
        "pushed_at": (item.get("pushed_at") or "")[:10],
        "license": license_info.get("spdx_id") or "Unknown",
        "size_kb": item.get("size", 0),
        "html_url": item.get("html_url"),
        "homepage": item.get("homepage") or "",
        "default_branch": item.get("default_branch") or "main",
        "archived": bool(item.get("archived")),
        "owner_type": owner.get("type") or "Unknown",
        "data_origin": "github_api",
    }


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {repo["name"]: repo for repo in data if repo.get("name")}


def save(path: Path, repos: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(repos.values(), key=lambda repo: (-int(repo.get("stars", 0)), repo["name"]))
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=1, ensure_ascii=False)


def query_plan(topics: list[str]) -> list[str]:
    queries = []
    for topic in topics:
        for bucket in STAR_BUCKETS:
            queries.append(f"topic:{topic} {bucket}")
    queries.extend([
        "stars:>=100000",
        "stars:50000..99999",
        "stars:20000..49999",
        "language:Python stars:1000..20000",
        "language:JavaScript stars:1000..20000",
        "language:TypeScript stars:1000..20000",
        "language:Go stars:1000..20000",
        "language:Rust stars:1000..20000",
    ])
    return queries


def collect(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    out_path = Path(args.out)
    repos = load_existing(out_path) if args.append else {}
    topics = args.topic if args.topic else TOPICS
    plan = query_plan(topics)

    if not args.token:
        print("Warning: no GITHUB_TOKEN found. Unauthenticated GitHub collection is heavily rate-limited.", file=sys.stderr)

    for query in plan:
        if len(repos) >= args.target:
            break
        pages = min(10, max(1, (args.per_query + 99) // 100))
        progress = tqdm(range(1, pages + 1), desc=query[:42], ncols=100)
        for page in progress:
            if len(repos) >= args.target:
                break
            payload = request_json(
                f"{API}/search/repositories",
                {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 100,
                    "page": page,
                },
                args.token,
                args.sleep,
            )
            items = payload.get("items", []) if isinstance(payload, dict) else []
            if not items:
                break
            for item in items:
                record = repo_record(item)
                if record["name"]:
                    repos[record["name"]] = record
            progress.set_postfix(unique=len(repos))
        save(out_path, repos)

    if args.readme_limit > 0:
        ordered = sorted(repos.values(), key=lambda repo: -int(repo.get("stars", 0)))[:args.readme_limit]
        for i, repo in enumerate(tqdm(ordered, desc="README snippets", ncols=100), start=1):
            if repo.get("readme"):
                continue
            repo["readme"] = request_readme(repo["name"], args.token, args.sleep)
            if args.readme_save_every > 0 and i % args.readme_save_every == 0:
                save(out_path, repos)
        save(out_path, repos)

    return repos


def main() -> None:
    args = parse_args()
    repos = collect(args)
    print(f"Collected {len(repos):,} unique repositories -> {args.out}")
    print("Next: python3 scripts/run_pipeline.py")


if __name__ == "__main__":
    main()
