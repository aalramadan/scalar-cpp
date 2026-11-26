"""Tag every row of a corpus with the installed Python SCALAR, as listen() of its server does.

Each worker process loads TaggingBackend of the scalar_tagger package on the CPU with one torch
thread, splits each name with spiral.ronin, tags its words with the context of the row and with
the type, language, and system of the row where the corpus has these columns, and sets the
dictionary flag as dictionary_lookup does from the bundled English words and the given en.txt
files. The answers go to a JSON Lines file with the inputs of the row and either the words of the
server answer or the error. Run it with a fixed PYTHONHASHSEED, since the Ronin splitter depends
on the iteration order of a set of strings.
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from scalar_tagger.tagging_backend import TaggingBackend, load_english_words
from scalar_tagger.version import __version__

OPTIONAL_COLUMNS = ("type", "language", "system")
_STATE = {}


def _start_worker(model: str, en_files: list) -> None:
    """Load the tagger and the dictionary once per worker process."""
    torch.set_num_threads(1)
    words = load_english_words(extra_words_path="")
    for path in en_files:
        with open(path) as handle:
            words.update(line.strip().lower() for line in handle if line.strip())
    _STATE["backend"] = TaggingBackend(model, local=True, device="cpu", english_words=words)


def _flag(word: str) -> str:
    """Return the dictionary flag of tag_identifier.dictionary_lookup without a word list."""
    if word.lower() in _STATE["backend"].english_words:
        return "DW"
    if word.isnumeric():
        return "DD"
    return "UC"


def _tag(row: dict) -> dict:
    """Return the server answer for one row, or the error it raises."""
    try:
        tagged = _STATE["backend"].tag_identifier(row["name"], row["context"], type_str=row.get("type", ""),
                                                  language=row.get("language", ""),
                                                  system_name=row.get("system", ""))
        words = [{word: {"dictionary": _flag(word), "tag": tag}} for word, tag in zip(tagged["tokens"], tagged["tags"])]
        return {**row, "words": words}
    except Exception as error:  # noqa: BLE001, the server answers HTTP 500 for any exception
        return {**row, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    """Tag the corpus in parallel and write the answers in corpus order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="model folder")
    parser.add_argument("--en", type=Path, nargs="*", default=[], help="the files of en.txt, in order")
    parser.add_argument("--corpus", type=Path, required=True, help="TSV with the columns name and context")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        raise SystemExit("set PYTHONHASHSEED, since the splitter depends on it")

    # The fields are read as written, without escapes, as scalarTagCorpus reads them
    with open(args.corpus, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        columns = ["name", "context"] + [name for name in OPTIONAL_COLUMNS if name in (reader.fieldnames or [])]
        rows = [{name: row[name] for name in columns} for row in reader]
    print(f"SCALAR {__version__}, {len(rows)} rows, columns {', '.join(columns)}", flush=True)
    init = (str(args.model.resolve()), [str(path) for path in args.en])
    with ProcessPoolExecutor(args.workers, initializer=_start_worker, initargs=init) as pool, \
            open(args.out, "w") as output:
        for count, answer in enumerate(pool.map(_tag, rows, chunksize=64), start=1):
            output.write(json.dumps(answer, sort_keys=True) + "\n")
            if count % 5000 == 0 or count == len(rows):
                print(f"progress {count}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
