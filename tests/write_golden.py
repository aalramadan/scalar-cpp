"""Write the golden files of the C++ encoder tests from the upstream SCALAR Python code.

The words of each identifier come from spiral.ronin.split, as the SCALAR server splits a name,
and the files store them, therefore the C++ tests need neither Python nor the splitter port.
ronin.split depends on the hash seed for a few names, such as TLS_ECDH_anon_WITH_3DES_EDE_CBC_SHA,
therefore the script runs only with PYTHONHASHSEED=0 to write the same files each time.

* corpus_words.tsv: the first letter of the context and the words of each row of identifiers_tag.tsv.
* corpus_digests.bin: per row, the digest of the legacy encoding, of the encoding of the default
  features with an empty type, language, and system name, and of the encoding of the default
  features with the type, language, and system name of a row of input/tagger_data.tsv, taken in
  turn; little-endian, 32 bits each.
* tagger_rows.tsv: context, type, language, system name, digest of the encoding of the default
  features, and the words of each row of input/tagger_data.tsv.
* stress_cases.tsv: edge cases with their full expected ids and positions, in the case format.
* ratio_cases.tsv: SequenceMatcher(None, a, b).ratio() for pairs of strings.
* float_sums.tsv: sum() of lists of floats, mostly cvr ratios whose plain running sum differs.
* feature_lists.tsv: the features of the model, the legacy features, and normalize_selected_features
  for feature lists.
"""

import argparse
import difflib
import random
import struct
import sys
import time
from pathlib import Path

from spiral import ronin

from scalar_reference import (
    CONTEXTS,
    FEATURE_SETS,
    ReferenceEncoder,
    case_line,
    digest,
    escape,
    read_tag_corpus,
    read_tagger_rows,
)

PRINTABLE = [chr(code) for code in range(0x20, 0x7F)]
CONTROL = [chr(code) for code in list(range(0x20)) + [0x7F]]
LETTERS = "abcdefghijklmnopqrstuvwxyz"

TYPES = [
    "int", "unsigned int", "const char *", "std::vector<int>", "std::map<std::string, int>&", "int[]", "uint8_t",
    "u_int32_t", "_int_t", "int_t", "size_t", "ssize_t", "ptrdiff_t", "xsize_t", "my_size_t", "size_t*", "int64",
    "uint", "iint8", "uint16", "Integer", "boolean", "GLboolean", "boolean_t", "void", "void *", "float", "double&",
    "enum Color", "MyEnum", "Callback", "std::function<void()>", "Runnable", "List<String>", "String[]", "char16_t",
    "wchar_t", "T", "", "   ", "static const unsigned long long", "signed char", "struct Foo", "class Bar", "union U",
    "volatile int", "Iterator<T>", "QString", "NSArray", "u8", "i32", "usize", "int__t", "int32_t*", "auto",
    "decltype(x)", "a->b", "a::b::C", "Foo.Bar", "foo-bar", "int\n", "uint8_t\t", "intx_t", "u_int_t", "_t",
    "Map<K, V>*", "array", "TextBuffer", "rune", "Number", "func", "lambda", "unsignedness", "constant",
    "2DPoint", "Vec2D", "32Bit", "3dVector", "x 64Bit", "int32Value",
]
LANGUAGES = ["C++", "C", "C#", "Java", " java ", "c++++", "C#++", "##", "Objective-C", "F#", "", "PYTHON", "c\t#", "+++"]
SYSTEMS = [
    "drill", "Apache Drill", "rigraph", "gimp.idents", "getHTTP", "get", "ge", "g", LETTERS * 10, "ab" * 150, "x" * 300,
    "getValue", "  getvalue  ", "QuantLib", "facebook-repo-ds2", "a_b-c.d", "::", "123", "v8", "Value", "valueGetter",
    "get" + "e" * 200, "".join(LETTERS[i % 7] for i in range(240)), "GETVALUE", "\tget\x1c", "3DEngine", "4Chan",
]
SYSTEM_WORDS = [["get", "Value"], ["drill", "Bit"], ["Quant", "Lib"], ["value"], ["v", "8"], ["x"], [], [""], ["ge", "t"],
                ["3", "D", "Engine"], ["4", "Chan"], ["2", "D", "Point"]]


def stress_cases():
    """Yield (group, feature set, (context, type, language, system name), words) for the edge cases."""
    for index, char in enumerate(PRINTABLE):
        context = CONTEXTS[index % len(CONTEXTS)]
        yield "stress/printable", "legacy", (context, "", "", ""), ["x" + char + "y", char, char * 3]
        yield "stress/printable", "default", (context, "T" + char + "int", char + "C++", "sys" + char), [char, "get" + char + "Value", "2"]
    for char in CONTROL:
        yield "stress/control", "legacy", ("FUNCTION", "", "", ""), ["a" + char + "b", char + "x"]
        yield "stress/control", "legacy", ("FUNCTION", "", "", ""), [char]
        yield "stress/control", "default", (char + "CLASS" + char, char + "int" + char, char + "Java", char + "app" + char), ["get" + char, char + "Value"]
    for length in (99, 100, 101, 102, 150, 250):
        word = (LETTERS * 10)[:length]
        yield "stress/long", "legacy", ("DECLARATION", "", "", ""), [word]
        yield "stress/long", "legacy", ("DECLARATION", "", "", ""), ["get", word.upper(), "x"]
        yield "stress/long", "default", ("ATTRIBUTE", "", "", ""), [("Ab1" * 100)[:length]]
        yield "stress/long", "legacy", ("PARAMETER", "", "", ""), ["9" * length]
        yield "stress/long", "legacy", ("PARAMETER", "", "", ""), [word + "_" + word[:50]]
        yield "stress/long", "default", ("FUNCTION", "", "", ""), [word[: length // 2] + " " + word[length // 2:]]
    for word in ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]", "a[CLS]b", "[cls]", "[[MASK]]", "[MASK][SEP]", "[UNK",
                 "UNK]", "[ CLS]", "x[PAD]y[PAD]", "[CLS][CLS]", "[MASK]2", "[SEP"]:
        yield "stress/special", "legacy", ("FUNCTION", "", "", ""), [word]
        yield "stress/special", "default", ("CLASS", "[CLS]", "[SEP]", "[PAD]"), ["get", word]
    for words in ([], [""], [" "], ["\t"], ["a", ""], ["", "a"], ["a", " ", "b"], ["get", "\x00"], ["  value  "], [" 2 "],
                  ["a b"], ["\x1c2\x1f"]):
        for feature_set in FEATURE_SETS:
            yield "stress/empty", feature_set, ("FUNCTION", "", "", ""), list(words)
    for words in (["fMatcher"], ["f", "Matcher"], ["b", "Force", "16", "bpp"], ["m", "value"], ["ABC"], ["abC"],
                  ["XMLHttp"], ["_x"], ["x_"], ["ab_c"], ["AB_"], ["aBc"], ["lpsz", "Name"], ["a", "B"], ["Ab"], ["abcD"],
                  ["abcdE"], ["a_"], ["A"], ["fMa"], ["fM"], ["f", "MA"]):
        for feature_set in FEATURE_SETS:
            yield "stress/hungarian", feature_set, ("ATTRIBUTE", "", "", ""), words
    for words in (["2"], ["get", "2", "set"], ["a2b"], ["x", "2"], ["2", "x"], ["12", "2", "2"], [" 2"], ["4", "2", "u"],
                  ["2", "2"], ["Two", "2", "3"], ["x", "2", "9"]):
        yield "stress/digit", "default", ("FUNCTION", "", "", ""), words
    for words in (["classes"], ["ties"], ["boxes"], ["status"], ["boss"], ["bus"], ["buses"], ["uses"], ["values", "keys"],
                  ["policies", "item"], ["get", "items", "Count"], ["is", "has", "was"], ["glass"], ["https"], ["Matches"],
                  ["axis"], ["virus"], ["IDS"], ["xs"], ["dies"], ["ies"], [" files "], ["Wishes"], ["heroes"], ["quizzes"]):
        yield "stress/plural", "default", ("DECLARATION", "", "", ""), words
    for words in (["aeiou"], ["xyz"], ["rhythm"], ["a"], ["bcdfg", "aei"], ["x1"], ["Strength"], ["queue"], ["a", "e"],
                  ["get", "Value", "Str"], ["bcd", "a", "x"], ["eye", "ccc"]):
        for feature_set in FEATURE_SETS:
            yield "stress/cvr", feature_set, ("PARAMETER", "", "", ""), words
    for index, type_str in enumerate(TYPES):
        words = [["get", "Value"], ["int", "Count"], ["size"], ["string", "Builder"], ["callback"], ["vec", "Map"]][index % 6]
        yield "stress/type", "default", (CONTEXTS[index % len(CONTEXTS)], type_str, "C++", ""), words
    for type_str, words in (("2DPoint", ["2", "D", "Point"]), ("32Bit", ["32", "Bit"]), ("Vec2D", ["vec2", "D"])):
        yield "stress/type", "default", ("PARAMETER", type_str, "C++", ""), words
    for language in LANGUAGES:
        yield "stress/language", "default", ("CLASS", "", language, ""), ["Widget"]
    for index, system in enumerate(SYSTEMS):
        for words in SYSTEM_WORDS[index % 3::3]:
            yield "stress/system", "default", ("FUNCTION", "", "Java", system), words


def truncation_cases(reference):
    """Return the cases on each side of the 512-id limit, for word lists of growing length."""
    cases = []
    shapes = [["a"], ["xqzvbnmlkj"], ["get", "Xqzvbnmlkjhgfdsa"], ["Value", "a" * 40]]
    for feature_set in FEATURE_SETS:
        for shape in shapes:
            previous = None
            for count in range(1, 140):
                words = (shape * count)[:count]
                result = reference.encode(feature_set, words, "FUNCTION", "", "", "")
                near_limit = result is not None and len(result[0]) >= 509
                first_error = result is None and previous is not None
                if near_limit or first_error:
                    cases.append(("stress/truncation", feature_set, ("FUNCTION", "", "", ""), words))
                if result is None and previous is None:
                    break
                previous = result
    return cases


def write_corpus(reference, corpus_path, tagger_rows, out_dir):
    """Write corpus_words.tsv and corpus_digests.bin for every row of identifiers_tag.tsv."""
    rows = read_tag_corpus(corpus_path)
    with open(out_dir / "corpus_words.tsv", "w", newline="\n", encoding="ascii") as words_file, \
            open(out_dir / "corpus_digests.bin", "wb") as digest_file:
        for index, (name, context) in enumerate(rows):
            if context not in CONTEXTS:
                raise ValueError(f"unexpected context {context!r}")
            words = ronin.split(name)
            _, _, type_str, language, system = tagger_rows[index % len(tagger_rows)]
            results = [
                reference.encode("legacy", words, context, "", "", ""),
                reference.encode("default", words, context, "", "", ""),
                reference.encode("default", words, context, type_str, language, system),
            ]
            words_file.write("\t".join([context[0]] + [escape(word) for word in words]) + "\n")
            digest_file.write(struct.pack("<3I", *(digest(result) for result in results)))
    return len(rows)


def write_tagger_rows(reference, tagger_rows, out_dir):
    """Write tagger_rows.tsv with the encoding of the default features of each training row of SCALAR."""
    with open(out_dir / "tagger_rows.tsv", "w", newline="\n", encoding="ascii") as handle:
        for name, context, type_str, language, system in tagger_rows:
            words = ronin.split(name)
            result = reference.encode("default", words, context, type_str, language, system)
            fields = [escape(context), escape(type_str), escape(language), escape(system), f"{digest(result):08x}"]
            handle.write("\t".join(fields + [escape(word) for word in words]) + "\n")


def write_stress(reference, out_dir):
    """Write stress_cases.tsv and return its number of cases."""
    cases = list(stress_cases()) + truncation_cases(reference)
    with open(out_dir / "stress_cases.tsv", "w", newline="\n", encoding="ascii") as handle:
        for group, feature_set, row, words in cases:
            result = reference.encode(feature_set, words, *row)
            handle.write(case_line(group, feature_set, row, words, result) + "\n")
    return len(cases)


def ratio_pairs():
    """Yield (a, b) pairs for SequenceMatcher, with long b strings that trigger autojunk."""
    rng = random.Random(20261106)
    alphabet = "abcde"
    for system in SYSTEMS:
        for word in ("get", "value", "drill", "x", "", "ab", "getvalue"):
            yield word, system
    for index in range(600):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        length = rng.randint(195, 420) if index % 4 == 0 else rng.randint(0, 20)
        b = "".join(rng.choice(alphabet[: rng.randint(1, 5)]) for _ in range(length))
        yield a, b


def write_ratios(out_dir):
    """Write ratio_cases.tsv with the ratio of difflib in its shortest round-trip form."""
    with open(out_dir / "ratio_cases.tsv", "w", newline="\n", encoding="ascii") as handle:
        for a, b in ratio_pairs():
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            handle.write(f"{escape(a)}\t{escape(b)}\t{ratio!r}\n")


def float_lists():
    """Yield lists of floats for sum(): cvr ratios whose plain running sum differs from sum(), and edge values."""
    rng = random.Random(1106)
    ratios = [c / (v + 1e-5) for c in range(41) for v in range(41) if c or v]
    kept = 0
    while kept < 150:
        values = [rng.choice(ratios) for _ in range(rng.randint(3, 12))]
        running = 0.0
        for value in values:
            running += value
        if running != sum(values):
            kept += 1
            yield values
    yield from ([-0.0], [-0.0, -0.0], [0.0, -0.0], [1e308, 1e308], [1e308, 1e308, -1e308], [float("inf"), -float("inf")],
                [0.1] * 10, [1.0, 1e100, 1.0, -1e100], [5e-324, 5e-324], [2.5])


def write_float_sums(out_dir):
    """Write float_sums.tsv: the values and their sum() in the shortest round-trip form."""
    with open(out_dir / "float_sums.tsv", "w", newline="\n", encoding="ascii") as handle:
        handle.writelines(" ".join(map(repr, values)) + "\t" + repr(sum(values)) + "\n" for values in float_lists())


def write_feature_lists(reference, out_dir):
    """Write feature_lists.tsv: each list and its normalized form or the message of its ValueError."""
    preprocessing = reference.preprocessing
    lists = [
        None, [], ["context"], [" cvr ", "context", "context"], ["hungarian_legacy", "digit"], ["bogus"],
        ["zeta", "alpha", "context", "zeta"], ["", "  "], list(preprocessing.LEGACY_FEATURES),
        list(preprocessing.AVAILABLE_FEATURES), list(reversed(preprocessing.AVAILABLE_FEATURES)), ["Context"],
        ["sys_sim", "\tlanguage\n"], ["hungarian", "hungarian_legacy"], ["\x1ctype\x1f", "type_overlap"],
    ]
    with open(out_dir / "feature_lists.tsv", "w", newline="\n", encoding="ascii") as handle:
        handle.write("\t".join(["model"] + reference.model_features) + "\n")
        handle.write("\t".join(["legacy"] + reference.features["legacy"]) + "\n")
        for features in lists:
            try:
                outcome = ["ok"] + preprocessing.normalize_selected_features(features)
            except ValueError as error:
                outcome = ["error", escape(str(error))]
            given = ["none"] if features is None else ["list", str(len(features))] + [escape(name) for name in features]
            handle.write("\t".join(given + outcome) + "\n")


def main():
    """Parse the paths and write every golden file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scalar", type=Path, required=True, help="checkout of scanl_tagger, for input/tagger_data.tsv")
    parser.add_argument("--model", type=Path, required=True, help="model folder")
    parser.add_argument("--corpus", type=Path, required=True, help="names and contexts, such as tools/identifiers_tag.tsv")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data" / "encoder")
    args = parser.parse_args()
    if sys.flags.hash_randomization:
        raise SystemExit("run with PYTHONHASHSEED=0, since ronin.split depends on the hash seed")

    started = time.perf_counter()
    reference = ReferenceEncoder(args.model)
    tagger_rows = read_tagger_rows(args.scalar / "input" / "tagger_data.tsv")
    args.out.mkdir(parents=True, exist_ok=True)
    corpus_rows = write_corpus(reference, args.corpus, tagger_rows, args.out)
    write_tagger_rows(reference, tagger_rows, args.out)
    stress_count = write_stress(reference, args.out)
    write_ratios(args.out)
    write_float_sums(args.out)
    write_feature_lists(reference, args.out)
    print(f"corpus rows {corpus_rows}, tagger rows {len(tagger_rows)}, stress cases {stress_count}, "
          f"{time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()
