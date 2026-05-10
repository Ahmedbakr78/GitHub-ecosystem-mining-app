# GitHub Repository Mining - Advanced Data Mining Project

This project is a complete GitHub ecosystem mining app for the required topics:

- Web mining and GitHub repository data collection
- Data preprocessing and feature engineering
- Association rule mining with Apriori
- Link analysis with PageRank and HITS
- Real BERT text classification and repository similarity using pretrained BERT-Tiny weights
- Interactive visualization dashboard
- Report and presentation deliverables

## Current Project Output

The latest processed run uses real repositories collected from the GitHub REST API with a token supplied through the `GITHUB_TOKEN` environment variable. The token is not stored in source code.

| Metric | Value |
|---|---:|
| Repositories | 5,049 |
| README snippets | 50 |
| Technologies / topics | 10,470 |
| Graph edges exported | 1,500 |
| Association rules discovered | 238 |
| Technology communities | 14 |
| Semantic categories | 13 |

With a `GITHUB_TOKEN`, the collector can collect larger real datasets from GitHub API shards across topics, languages, and star ranges.

## Quick Start

Run everything from the project root:

```bash
python3 scripts/collect_data.py --target 1000 --readme-limit 50
python3 scripts/run_pipeline.py
python3 -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080/app/
```

## Running the Notebook in Colab or Jupyter

For Google Colab, you can upload only:

```text
github_mining_analysis.ipynb
```

Then run the notebook from the first cell. If the project files are not present, the notebook restores the embedded real GitHub API dataset, processed files, reports, and dashboard files under `/content/DM_Project/`.

Important for the BERT requirement: run the `Real BERT Model Refresh` cell. It runs a real pretrained BERT-Tiny model (`phob0s/bert-tiny`) over repository descriptions, README snippets, topics, and languages, then updates `stats.json` to a real `BERT/NumPy BERT-Tiny` run.

For the full local project, open the notebook from the project root so it can use the richer generated files in `data/processed/`.

If you also upload the full project folder to Colab, put it at `/content/DM_Project`, then run this in the first notebook cell if needed:

```python
PROJECT_ROOT_OVERRIDE = "/content/DM_Project"
```

Uploading a project zip is optional because the notebook can now run alone with embedded real data.

If processed files are missing, run:

```bash
python3 scripts/collect_data.py --target 1000 --readme-limit 50
python3 scripts/run_pipeline.py
python3 scripts/build_notebook.py
```

## Real GitHub Data Collection

Use this when you want the dataset to come directly from GitHub:

```bash
export GITHUB_TOKEN=your_token_here
python3 scripts/collect_data.py --target 30000 --readme-limit 1500
python3 scripts/run_pipeline.py
python3 scripts/build_notebook.py
python3 scripts/validate_project.py
```

Notes:

- GitHub Search API limits each query to 1,000 results, so the collector shards across topics and star buckets.
- A token is strongly recommended for large collection.
- README fetching is limited by `--readme-limit` to avoid wasting API quota.
- Never commit your token into the project files. Keep it only in your terminal environment.

### How to Get a GitHub Token

1. Open GitHub.
2. Click your profile picture.
3. Go to `Settings`.
4. Go to `Developer settings`.
5. Open `Personal access tokens`.
6. Choose `Fine-grained tokens`.
7. Click `Generate new token`.
8. Set a short expiration, for example 7 or 30 days.
9. For public repository mining, no private repository access is needed. Keep permissions minimal/read-only.
10. Copy the token once and use it in the terminal:

```bash
export GITHUB_TOKEN=ghp_or_github_pat_your_token_here
python3 scripts/collect_data.py --target 30000 --readme-limit 1500
```

Official GitHub documentation: https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token

## Validation

Run this before submission:

```bash
python3 scripts/validate_project.py
```

The validator checks data collection, preprocessing, association rules, PageRank/HITS, visualization, report/presentation files, and BERT/semantic similarity. It writes `outputs/validation_report.md`.

## BERT / Semantic Analysis

The notebook includes a `Real BERT Model Refresh` cell. It runs `phob0s/bert-tiny` from Hugging Face using the lightweight NumPy BERT inference path in `scripts/run_pipeline.py`, then rebuilds the processed outputs before the analysis sections load them. This avoids a heavy PyTorch install while still using real pretrained BERT weights.

Local command:

```bash
python3 scripts/run_pipeline.py --semantic-model bert-tiny-numpy
```

The dashboard shows the active embedding method in the BERT section. The latest processed output uses `BERT/NumPy BERT-Tiny: phob0s/bert-tiny`.

## Project Structure

```text
DM_Project/
|-- app/
|   |-- index.html
|   |-- css/styles.css
|   `-- js/app.js
|-- data/
|   |-- raw/github_repos.json
|   `-- processed/
|       |-- repos.json
|       |-- graph_nodes.json
|       |-- graph_edges.json
|       |-- association_rules.json
|       |-- similarities.json
|       `-- stats.json
|-- outputs/
|   |-- project_report.md
|   `-- presentation_outline.md
|-- scripts/
|   |-- collect_data.py
|   |-- run_pipeline.py
|   |-- validate_project.py
|   `-- build_notebook.py
|-- github_mining_analysis.ipynb
|-- requirements.txt
`-- README.md
```

## Algorithms

| Requirement | Implementation |
|---|---|
| Data collection | GitHub REST API collector plus embedded real GitHub API notebook bundle |
| Preprocessing | Text cleaning, topic normalization, language/topic tech stacks |
| Association rules | Apriori itemsets up to size 3 or 4 with support, confidence, lift, conviction, leverage |
| Link analysis | Weighted technology co-occurrence graph, PageRank, HITS, label propagation communities |
| Visualization | Chart.js charts, D3 network graph, PCA cluster map, repository explorer |
| BERT | Real pretrained BERT-Tiny embeddings using `phob0s/bert-tiny` safetensors and NumPy inference |
| Recommendations | Nearest-neighbor cosine similarity over semantic embeddings |

## Grading Checklist

| Rubric item | Where to find it |
|---|---|
| Data Collection and GUI | `scripts/collect_data.py`, `app/`, embedded notebook project bundle |
| Data Preprocessing | `scripts/run_pipeline.py`, preprocessing section |
| Association Rule Mining | `data/processed/association_rules.json`, dashboard Association section |
| Link Analysis | `data/processed/graph_nodes.json`, dashboard Link Analysis section |
| Visualization | Dashboard charts, D3 graph, PCA plot |
| BERT | `scripts/run_pipeline.py`, dashboard BERT section |
| Report and Presentation | `outputs/project_report.md`, `outputs/presentation_outline.md` |

## Deliverables

- Interactive app: `app/index.html`
- Processed data: `data/processed/`
- Real GitHub API raw dataset: `data/raw/github_repos.json`
- Report: `outputs/project_report.md`
- Presentation outline: `outputs/presentation_outline.md`

## License

MIT License - Academic Project 2026
