"""Compare two sets of SCALAR answers for the same rows, word by word.

Each input is a JSON Lines file in one of two forms. Answers of a SCALAR server, saved with
the HTTP status and the body, hold the words in the body. The answers of
tag_with_python_scalar.py and of scalarTagCorpus hold the words or the error. The output
is one JSON file with the counts of rows that agree, rows whose split, tags, or dictionary
flags differ, rows that fail on one side, the tag changes word by word, the counts per
context, and the first differing rows as examples. A row is its name, its context, and its
type, language, and system where the answers hold them.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

EXAMPLES = 40
OPTIONAL_COLUMNS = ("type", "language", "system")


def normalize(record: dict) -> tuple:
    """Return ("words", ((word, tag, flag), ...)) or ("error", text) for one answer."""
    if "status" in record:
        if record["status"] != 200:
            return ("error", f"HTTP {record['status']}")
        record = json.loads(record["body"])
    if "error" in record:
        return ("error", record["error"])
    words = []
    for entry in record["words"]:
        ((word, value),) = entry.items()
        words.append((word, value["tag"], value["dictionary"]))
    return ("words", tuple(words))


def load_answers(path: Path) -> dict:
    """Return the normalized answer of each row of a JSON Lines file."""
    answers = {}
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            key = (record["name"], record["context"], *(record.get(name, "") for name in OPTIONAL_COLUMNS))
            answers[key] = normalize(record)
    return answers


def compare(first: dict, second: dict) -> dict:
    """Return the counts of agreement and difference between two sets of answers."""
    keys = sorted(first.keys() & second.keys())
    rows = Counter()
    tag_changes = Counter()
    by_context = {}
    examples = []
    words_compared = words_differing = 0
    for key in keys:
        (kind_a, value_a), (kind_b, value_b) = first[key], second[key]
        context = by_context.setdefault(key[1], Counter())
        context["rows"] += 1
        if kind_a == "error" or kind_b == "error":
            outcome = {("error", "error"): "error_in_both", ("error", "words"): "error_in_first_only",
                       ("words", "error"): "error_in_second_only"}[(kind_a, kind_b)]
        elif [w for w, _, _ in value_a] != [w for w, _, _ in value_b]:
            outcome = "different_split"
        else:
            words_compared += len(value_a)
            changed = [(a[1], b[1]) for a, b in zip(value_a, value_b) if a[1] != b[1]]
            words_differing += len(changed)
            tag_changes.update(f"{a}->{b}" for a, b in changed)
            if changed:
                outcome = "different_tags"
            elif [f for _, _, f in value_a] != [f for _, _, f in value_b]:
                outcome = "different_flags"
            else:
                outcome = "identical"
        rows[outcome] += 1
        context[outcome] += 1
        if outcome not in ("identical", "error_in_both") and len(examples) < EXAMPLES:
            examples.append({"row": dict(zip(("name", "context", *OPTIONAL_COLUMNS), key)), "outcome": outcome,
                             "first": first[key][1], "second": second[key][1]})
    return {
        "rows_in_first": len(first),
        "rows_in_second": len(second),
        "rows_compared": len(keys),
        "rows": dict(sorted(rows.items())),
        "words_compared": words_compared,
        "words_with_different_tag": words_differing,
        "tag_changes": dict(tag_changes.most_common()),
        "by_context": {name: dict(sorted(counts.items())) for name, counts in sorted(by_context.items())},
        "examples": examples,
    }


def main() -> None:
    """Compare two answer files and write the counts as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"first": str(args.first), "second": str(args.second),
              **compare(load_answers(args.first), load_answers(args.second))}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["rows"]), result["words_with_different_tag"], "of", result["words_compared"],
          "words differ")
    only_first = result["rows_in_first"] - result["rows_compared"]
    only_second = result["rows_in_second"] - result["rows_compared"]
    if only_first or only_second:
        print(f"{only_first} rows only in --first and {only_second} rows only in --second, not compared")


if __name__ == "__main__":
    main()
