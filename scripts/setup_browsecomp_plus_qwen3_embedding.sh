#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BCP_REPO_DIR="${BCP_REPO_DIR:-$ROOT_DIR/external/BrowseComp-Plus}"
CONDA_ENV="${BCP_CONDA_ENV:-$ROOT_DIR/.conda/browsecomp-py310}"
PYTHON_BIN="$CONDA_ENV/bin/python"

mkdir -p "$ROOT_DIR/external"

if [[ ! -d "$BCP_REPO_DIR/.git" ]]; then
  git clone https://github.com/texttron/BrowseComp-Plus "$BCP_REPO_DIR"
else
  git -C "$BCP_REPO_DIR" pull --ff-only
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  conda create -y -p "$CONDA_ENV" python=3.10
fi

"$PYTHON_BIN" -m pip install -U pip "setuptools<82" wheel
"$PYTHON_BIN" -m pip install -e "$ROOT_DIR/agent"
"$PYTHON_BIN" -m pip install \
  "datasets>=4.0.0" \
  "huggingface_hub>=0.25.0" \
  "transformers>=4.53.2" \
  "accelerate>=1.9.0" \
  "peft>=0.16.0" \
  "faiss-cpu>=1.11.0.post1" \
  "fastmcp==2.9.2" \
  "qwen-omni-utils==0.0.8" \
  "python-dotenv>=1.1.1" \
  "rich>=14.0.0" \
  "torchvision" \
  "tqdm>=4.67.1"
if ! "$PYTHON_BIN" -c "import tevatron" >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install "git+https://github.com/texttron/tevatron.git@main"
fi

mkdir -p "$BCP_REPO_DIR/data" "$BCP_REPO_DIR/topics-qrels" "$BCP_REPO_DIR/indexes"

if [[ -x "$CONDA_ENV/bin/hf" ]]; then
  "$CONDA_ENV/bin/hf" download Tevatron/browsecomp-plus-indexes \
    --repo-type=dataset \
    --include="qwen3-embedding-8b/*" \
    --local-dir "$BCP_REPO_DIR/indexes"
else
  "$CONDA_ENV/bin/huggingface-cli" download Tevatron/browsecomp-plus-indexes \
    --repo-type=dataset \
    --include="qwen3-embedding-8b/*" \
    --local-dir "$BCP_REPO_DIR/indexes"
fi

(
  cd "$BCP_REPO_DIR"
  "$PYTHON_BIN" scripts_build_index/decrypt_dataset.py \
    --output data/browsecomp_plus_decrypted.jsonl \
    --generate-tsv topics-qrels/queries.tsv
)

QUERY_ROWS="$(wc -l < "$BCP_REPO_DIR/topics-qrels/queries.tsv" | tr -d ' ')"
INDEX_SHARDS="$(find "$BCP_REPO_DIR/indexes/qwen3-embedding-8b" -name '*.pkl' | wc -l | tr -d ' ')"

echo "browsecomp_repo=$BCP_REPO_DIR"
echo "python=$PYTHON_BIN"
echo "queries=$BCP_REPO_DIR/topics-qrels/queries.tsv rows=$QUERY_ROWS"
echo "index_dir=$BCP_REPO_DIR/indexes/qwen3-embedding-8b shards=$INDEX_SHARDS"
