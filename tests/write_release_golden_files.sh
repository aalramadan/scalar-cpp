#!/usr/bin/env bash
# Writes the golden files of ctest that depend on the SCALAR release: the encoder cases, the model rows, the
# CRF cases, and the tagger answers. The splitter files depend on Spiral alone, as PORT.md describes.
# Usage: tests/write_release_golden_files.sh <scanl_tagger checkout at the release> <build folder>
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <scanl_tagger checkout at the release> <build folder>" >&2
    exit 2
fi
# shellcheck source=tools/scalar_release.sh
source "$(dirname "$0")/../tools/scalar_release.sh"
checkout="$(cd "$1" && pwd)" build="$(cd "$2" && pwd)"
require_installed_release
model="$(model_folder)"
corpus="$scalar_repo/tools/identifiers_tag.tsv"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cd "$scalar_repo/tests"
PYTHONHASHSEED=0 python3 write_golden.py --scalar "$checkout" --model "$model" --corpus "$corpus"
PYTHONHASHSEED=0 SCALAR_MODEL="$model" SCALAR_CORPUS="$corpus" python3 write_golden_files.py \
    --corpus-out "$work/model_reference.tsv" --golden-dir data/model
"$scalar_repo/tools/compare_with_python.sh" "$build" "$corpus" "$work/comparison" data/tagger/custom_words.txt
python3 write_tagger_golden.py --answers "$work/comparison/python.jsonl" --out data/tagger/golden_answers.jsonl
