"""Write the Ronin splits of Spiral for the long tokens of tests/data/tagger/long_tokens.tsv.

The small-stack test of the tagger splits these tokens on a thread with a stack of 256 KiB.
"""
import argparse

from spiral import ronin

CASES = ["a" * 100, "a" * 500, "a" * 1500, "getvalue" * 100, "Get" + "value" * 150, "x" * 300 + "y" * 300, "a" * 1999]


def main():
    """Split each case with Ronin and write one tab-separated line per case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "w", newline="\n") as handle:
        handle.writelines(case + "\t" + "\t".join(ronin.split(case)) + "\n" for case in CASES)


if __name__ == "__main__":
    main()
