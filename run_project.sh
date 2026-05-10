#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

PORT="${PORT:-8080}"
RUN_NOTEBOOK=1
RUN_VALIDATE=1
RUN_SERVER=1
REBUILD_NOTEBOOK=1

usage() {
  cat <<'USAGE'
GitHub Repository Mining professional runner

Usage:
  ./run_project.sh [options]

Options:
  --port PORT          Start dashboard from this port, default 8080.
  --skip-build         Do not rebuild github_mining_analysis.ipynb.
  --skip-notebook      Do not execute the notebook.
  --skip-validate      Do not run scripts/validate_project.py.
  --no-server          Run checks only, do not start the dashboard server.
  --quick              Same as --skip-build --skip-notebook.
  -h, --help           Show this help.

Default workflow:
  1. Check required files.
  2. Rebuild notebook from scripts/build_notebook.py.
  3. Execute notebook in offline validation mode.
  4. Run the rubric validator.
  5. Serve dashboard at http://127.0.0.1:8080/app/ or next free port.

Notes:
  - No GitHub token is stored here.
  - Notebook execution uses DM_PROJECT_OFFLINE_VALIDATION=1 so CI-style checks do
    not stop for a token prompt.
  - For fresh live GitHub collection, open the notebook and paste your token in
    the secure runtime prompt.
USAGE
}

log() {
  printf '\n\033[1;34m==>\033[0m %s\n' "$*"
}

ok() {
  printf '\033[1;32mOK\033[0m %s\n' "$*"
}

warn() {
  printf '\033[1;33mWARN\033[0m %s\n' "$*"
}

fail() {
  printf '\033[1;31mERROR\033[0m %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      [[ $# -ge 2 ]] || fail "--port needs a value."
      PORT="$2"
      shift 2
      ;;
    --skip-build)
      REBUILD_NOTEBOOK=0
      shift
      ;;
    --skip-notebook)
      RUN_NOTEBOOK=0
      shift
      ;;
    --skip-validate)
      RUN_VALIDATE=0
      shift
      ;;
    --no-server)
      RUN_SERVER=0
      shift
      ;;
    --quick)
      REBUILD_NOTEBOOK=0
      RUN_NOTEBOOK=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

require_file() {
  [[ -e "$1" ]] || fail "Missing required file or directory: $1"
}

check_required_files() {
  log "Checking project structure"
  require_file "github_mining_analysis.ipynb"
  require_file "requirements.txt"
  require_file "app/index.html"
  require_file "app/js/app.js"
  require_file "app/css/styles.css"
  require_file "scripts/build_notebook.py"
  require_file "scripts/run_pipeline.py"
  require_file "scripts/validate_project.py"
  require_file "data/raw/github_repos.json"
  require_file "data/processed/repos.json"
  require_file "data/processed/graph_nodes.json"
  require_file "data/processed/graph_edges.json"
  require_file "data/processed/association_rules.json"
  require_file "data/processed/similarities.json"
  require_file "data/processed/stats.json"
  require_file "models/bert-tiny/config.json"
  require_file "models/bert-tiny/vocab.txt"
  require_file "models/bert-tiny/model.safetensors"
  ok "Project files are present."
}

check_python_modules() {
  log "Checking Python runtime"
  "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

required = [
    "pandas",
    "numpy",
    "sklearn",
    "matplotlib",
    "requests",
    "ipywidgets",
    "nbformat",
    "nbconvert",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python modules:", ", ".join(missing))
    print("Install them with: python3 -m pip install -r requirements.txt nbconvert nbformat")
    sys.exit(1)
print("Python executable:", sys.executable)
print("Required runtime modules are available.")
PY
}

rebuild_notebook() {
  if [[ "$REBUILD_NOTEBOOK" -eq 0 ]]; then
    warn "Skipping notebook rebuild."
    return
  fi

  log "Rebuilding notebook"
  "$PYTHON_BIN" scripts/build_notebook.py
  ok "Notebook rebuilt."
}

execute_notebook() {
  if [[ "$RUN_NOTEBOOK" -eq 0 ]]; then
    warn "Skipping notebook execution."
    return
  fi

  log "Executing notebook top-to-bottom"
  mkdir -p /tmp/dm_project_jupyter/config /tmp/dm_project_jupyter/data /tmp/dm_project_jupyter/runtime
  DM_PROJECT_OFFLINE_VALIDATION=1 \
  JUPYTER_CONFIG_DIR=/tmp/dm_project_jupyter/config \
  JUPYTER_DATA_DIR=/tmp/dm_project_jupyter/data \
  JUPYTER_RUNTIME_DIR=/tmp/dm_project_jupyter/runtime \
  "$PYTHON_BIN" -m nbconvert \
    --to notebook \
    --execute \
    --inplace \
    github_mining_analysis.ipynb \
    --ExecutePreprocessor.timeout=900
  ok "Notebook executed with no blocking token prompt."
}

validate_project() {
  if [[ "$RUN_VALIDATE" -eq 0 ]]; then
    warn "Skipping project validation."
    return
  fi

  log "Running rubric validator"
  "$PYTHON_BIN" scripts/validate_project.py
  ok "Validation complete."
}

print_project_summary() {
  log "Project summary"
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

stats = json.loads(Path("data/processed/stats.json").read_text(encoding="utf-8"))
print(f"Repositories: {stats.get('total_repos'):,}")
print(f"Technologies/topics: {stats.get('total_technologies'):,}")
print(f"Graph edges: {stats.get('total_edges'):,}")
print(f"Association rules: {stats.get('total_rules'):,}")
print(f"Semantic categories: {stats.get('total_categories'):,}")
print(f"BERT method: {stats.get('embedding_method')}")
print(f"Data sources: {stats.get('data_source_distribution')}")
PY
}

find_free_port() {
  "$PYTHON_BIN" - "$PORT" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for port in range(start, start + 50):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit(f"No free port found from {start} to {start + 49}")
PY
}

start_dashboard() {
  if [[ "$RUN_SERVER" -eq 0 ]]; then
    warn "Server disabled by --no-server."
    return
  fi

  log "Starting dashboard server"
  local selected_port
  selected_port="$(find_free_port)"

  if [[ "$selected_port" != "$PORT" ]]; then
    warn "Port $PORT is busy. Using $selected_port instead."
  fi

  echo
  echo "Dashboard URL:"
  echo "  http://127.0.0.1:${selected_port}/app/"
  echo
  echo "Notebook file:"
  echo "  $ROOT_DIR/github_mining_analysis.ipynb"
  echo
  echo "Press Ctrl+C to stop the dashboard server."
  echo

  "$PYTHON_BIN" -m http.server "$selected_port" --bind 127.0.0.1
}

main() {
  check_required_files
  check_python_modules
  rebuild_notebook
  execute_notebook
  validate_project
  print_project_summary
  start_dashboard
}

main "$@"
