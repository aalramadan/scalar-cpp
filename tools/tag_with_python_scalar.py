"""Tag every row of a corpus with the Python SCALAR of a given checkout, as its server does.

Each worker process loads the checkout's DistilBertTagger on the CPU with one torch thread,
splits each name with spiral.ronin, tags the words in the row's context with empty type,
language, and system name (nameChecker sends none), and sets the dictionary flag from the
NLTK words corpus and en.txt, read with strip().lower() as the server of nameChecker reads
it. The answers go to a JSON Lines file with the name, the context, and either the words of
the server answer or the error. Run it with a fixed PYTHONHASHSEED, since the Ronin splitter
depends on the iteration order of a set of strings.
"""

import argparse
import csv
import importlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nltk
import torch

_STATE = {}


def _start_worker(checkout: str, model: str, en_files: list) -> None:
    """Load the tagger, the splitter, and the dictionary once per worker process."""
    torch.set_num_threads(1)
    # The checkout is known only at run time, therefore its modules load by name
    sys.path.insert(0, checkout)
    ronin = importlib.import_module("spiral.ronin")
    DistilBertTagger = importlib.import_module("src.lm_based_tagger.distilbert_tagger").DistilBertTagger

    words = {word.lower() for word in nltk.corpus.words.words()}
    for path in en_files:
        with open(path) as handle:
            words.update(line.strip().lower() for line in handle)
    _STATE.update(tagger=DistilBertTagger(model, local=True), split=ronin.split, words=words)


def _flag(word: str) -> str:
    """Return the dictionary flag of tag_identifier.dictionary_lookup without a word list."""
    if word.lower() in _STATE["words"]:
        return "DW"
    if word.isnumeric():
        return "DD"
    return "UC"


def _tag(row: tuple) -> dict:
    """Return the server answer for one name and context, or the error it raises."""
    name, context = row
    try:
        words = _STATE["split"](name)
        tags = _STATE["tagger"].tag_identifier(tokens=words, context=context, type_str="", language="",
                                               system_name="")
        answer = [{word: {"dictionary": _flag(word), "tag": tag}} for word, tag in zip(words, tags)]
        return {"name": name, "context": context, "words": answer}
    except Exception as error:  # noqa: BLE001, the server answers HTTP 500 for any exception
        return {"name": name, "context": context, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    """Tag the corpus in parallel and write the answers in corpus order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True, help="SCALAR source folder")
    parser.add_argument("--model", type=Path, required=True, help="model folder")
    parser.add_argument("--en", type=Path, nargs="+", required=True, help="the files of en.txt, in order")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") is None:
        raise SystemExit("set PYTHONHASHSEED, since the splitter depends on it")

    # The fields are read as written, without escapes, as scalarTagCorpus reads them
    with open(args.corpus, newline="") as handle:
        rows = [(row["name"], row["context"])
                for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)]
    init = (str(args.checkout.resolve()), str(args.model.resolve()), [str(path) for path in args.en])
    with ProcessPoolExecutor(args.workers, initializer=_start_worker, initargs=init) as pool, \
            open(args.out, "w") as output:
        for count, answer in enumerate(pool.map(_tag, rows, chunksize=64), start=1):
            output.write(json.dumps(answer, sort_keys=True) + "\n")
            if count % 2000 == 0 or count == len(rows):
                print(f"progress {count}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
