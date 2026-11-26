#!/usr/bin/env bash
# Tags a corpus with the installed Python SCALAR and with scalarTagCorpus of a build, then compares the
# answers word by word. The Python SCALAR must be the release that cmake/scalarRelease.cmake names.
# Usage: tools/compare_with_python.sh <build folder> <corpus.tsv> <output folder> [word list]...
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <build folder> <corpus.tsv> <output folder> [word list]..." >&2
    exit 2
fi
# shellcheck source=tools/scalar_release.sh
source "$(dirname "$0")/scalar_release.sh"
build="$1" corpus="$2" out="$3"
shift 3
require_installed_release

mkdir -p "$out"
PYTHONHASHSEED=0 python3 "$scalar_repo/tools/tag_with_python_scalar.py" --model "$(model_folder)" --corpus "$corpus" \
    --out "$out/python.jsonl" --en "$@"
"$build/scalarTagCorpus" "$corpus" "$out/port.jsonl" "$(nproc)" "$@"
python3 "$scalar_repo/tools/compare_tag_answers.py" --first "$out/python.jsonl" --second "$out/port.jsonl" \
    --out "$out/comparison.json"
