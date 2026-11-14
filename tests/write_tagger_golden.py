"""Write the golden answers of the tagger tests from the answers of the Python SCALAR.

The input is the JSON Lines output of tools/tag_with_python_scalar.py on a corpus. The golden
file keeps every failed answer and every answer without words, the first answers with a word
flagged DD, and every n-th other answer of each context until each context has its share.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

PER_CONTEXT = 60
WITH_DIGITS = 100


def has_digits(answer: dict) -> bool:
    """Return whether a tagged answer holds a word flagged DD."""
    return any(next(iter(word.values()))["dictionary"] == "DD" for word in answer.get("words", []))


def select(answers: list) -> list:
    """Return the edge cases and an even share of the other answers of each context."""
    edge = [answer for answer in answers if "error" in answer or not answer["words"]]
    digits = [answer for answer in answers if answer not in edge and has_digits(answer)][:WITH_DIGITS]
    by_context = defaultdict(list)
    for answer in answers:
        if answer not in edge and not has_digits(answer):
            by_context[answer["context"]].append(answer)
    chosen = []
    for context in sorted(by_context):
        rows = by_context[context]
        step = max(1, len(rows) // PER_CONTEXT)
        chosen.extend(rows[::step][:PER_CONTEXT])
    return edge + digits + chosen


def main() -> None:
    """Read the Python answers and write the golden subset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True, help="output of tools/tag_with_python_scalar.py")
    parser.add_argument("--out", type=Path, required=True, help="golden file, such as data/tagger/golden_answers.jsonl")
    args = parser.parse_args()
    with open(args.answers) as handle:
        answers = [json.loads(line) for line in handle]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        handle.writelines(json.dumps(answer, sort_keys=True) + "\n" for answer in select(answers))


if __name__ == "__main__":
    main()
