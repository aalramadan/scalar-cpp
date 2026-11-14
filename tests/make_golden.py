"""Write the golden files of the C++ splitter tests from Spiral dff5373 and NLTK 3.10.3.

  data/splitter/stem_golden.tsv     word, then SnowballStemmer('english').stem(word)
  data/splitter/split_golden.tsv    identifier, then each token of ronin.split(identifier)
  data/splitter/rescale_golden.tsv  count, then the bits of math.pow(count, dict_word_exponent) and of
                                    math.pow(count, normal_exponent), for each count above low_freq_cutoff
  data/splitter/alt_golden.tsv      n, then the bits of math.pow(n, alt_exponent), for n from 1 to 200

The words and identifiers are stratified samples: inputs are grouped by the features that select
the code paths of the stemmer and the splitter, and the sample takes one input of each group in
turn. Every corpus string that holds a longer one of two competing terms with numbers joins the
splitter sample. The splits come from Spiral under the longest-first order of the terms with
numbers (python_reference.use_longest_first_terms), the rule of the C++ port.

Usage: SCALAR_SPLIT_CORPUS=<one identifier per line> python3 tests/make_golden.py
"""

import collections
import math
import pathlib
import random
import re
import struct

from nltk.stem import SnowballStemmer
from spiral import ronin, simple_splitters

import python_reference as reference

DATA = pathlib.Path(__file__).resolve().parent / 'data' / 'splitter'
STEM_SAMPLE = 20000
SPLIT_SAMPLE = 20000
ALT_LIMIT = 200


def float_bits(value):
    """The 64 bits of a double as 16 hex digits."""
    return f'{struct.unpack("<Q", struct.pack("<d", value))[0]:016x}'


def stratified(groups, size, seed):
    """Take one item of each group in turn, the groups and their items in a seeded random order."""
    rng = random.Random(seed)
    queues = []
    for key in sorted(groups):
        items = sorted(set(groups[key]))
        rng.shuffle(items)
        queues.append(collections.deque(items))
    rng.shuffle(queues)
    sample = []
    while len(sample) < size and queues:
        for queue in list(queues):
            if len(sample) == size:
                break
            sample.append(queue.popleft())
            if not queue:
                queues.remove(queue)
    return sample


def write_lines(name, rows):
    """Write escaped rows and report the size of the file."""
    path = DATA / name
    path.write_text(''.join('\t'.join(reference.escape_field(f) for f in row) + '\n' for row in rows), encoding='ascii')
    print(f'{name}: {len(rows)} lines, {path.stat().st_size} bytes')


def stem_rows(corpus_tokens):
    """A stratified sample of words of each source with their NLTK stems."""
    stemmer = SnowballStemmer('english')
    frequency_keys, dictionary = reference.spiral_tables()
    sources = {
        'dictionary': dictionary,
        'frequency': frequency_keys,
        'corpus': corpus_tokens,
        'stress': reference.stress_words(seed=11, count=60000),
    }
    groups = collections.defaultdict(list)
    for source, words in sources.items():
        for word in words:
            stem = stemmer.stem(word)
            changed = stem != word.lower()
            groups[(source, min(len(word), 12), word.lower()[-2:], stem[-2:], changed)].append(word)
    special = list(stemmer.stemmer._EnglishStemmer__special_words)
    words = sorted(set(stratified(groups, STEM_SAMPLE - 3 * len(special), seed=12)))
    words += [variant for word in special for variant in (word, word.upper(), "'" + word)]
    return [(word, stemmer.stem(word)) for word in words]


def split_features(identifier, tokens):
    """Features that select the code paths of heuristic_split and of Ronin."""
    initial = simple_splitters.heuristic_split(identifier)
    return (
        min(len(identifier) // 6, 6),
        min(len(tokens), 6),
        any(c.isdigit() for c in identifier),
        bool(re.search(r'[a-z][A-Z]', identifier)),
        bool(re.search(r'[A-Z][a-z]', identifier)),
        any(c in reference.DELIMITERS for c in identifier),
        any(not c.isalnum() and c not in reference.DELIMITERS for c in identifier),
        bool(simple_splitters._exceptions_re.search(identifier)),
        len(tokens) - len(initial),
    )


def split_rows():
    """A stratified sample of corpus and stress identifiers with their Ronin splits."""
    corpus = reference.corpus_identifiers()
    frequency_keys, _ = reference.spiral_tables()
    words = [k for k in frequency_keys if k.isalpha()]
    sources = {
        'corpus': corpus,
        'printable': reference.stress_identifiers(seed=1, count=40000, words=words, alphabet='printable'),
        'ascii': reference.stress_identifiers(seed=2, count=8000, words=words, alphabet='ascii'),
    }
    longer_terms = sorted({long for _, long in reference.competing_terms()})
    competing = [s for s in corpus if any(term in s.lower() for term in longer_terms)]
    groups = collections.defaultdict(list)
    splits = {}
    for source, identifiers in sources.items():
        for identifier in identifiers:
            tokens = ronin.split(identifier)
            splits[identifier] = tokens
            groups[(source,) + split_features(identifier, tokens)].append(identifier)
    sample = sorted(set(stratified(groups, SPLIT_SAMPLE - len(competing), seed=13)) | set(competing))
    corpus_tokens = sorted({token.lower() for identifier in corpus for token in splits[identifier]} |
                           {token for identifier in corpus for token in splits[identifier]})
    return [(identifier, *splits[identifier]) for identifier in sample], corpus_tokens


def rescale_rows():
    """Python's math.pow for every count that Ronin rescales and for the lengths of splits."""
    splitter = ronin.Ronin()
    splitter.init()
    counts = sorted({c for c in splitter._frequencies.values() if c > splitter._low_freq_cutoff})
    rescale = [(str(c), float_bits(math.pow(c, splitter._dict_word_exponent)), float_bits(math.pow(c, splitter._normal_exponent)))
               for c in counts]
    alt = [(str(n), float_bits(math.pow(n, splitter._alt_exponent))) for n in range(1, ALT_LIMIT + 1)]
    return rescale, alt


def main():
    """Write every golden file."""
    reference.use_longest_first_terms()
    ronin.init()
    DATA.mkdir(parents=True, exist_ok=True)
    splits, corpus_tokens = split_rows()
    write_lines('split_golden.tsv', splits)
    write_lines('stem_golden.tsv', stem_rows(corpus_tokens))
    rescale, alt = rescale_rows()
    write_lines('rescale_golden.tsv', rescale)
    write_lines('alt_golden.tsv', alt)


if __name__ == '__main__':
    main()
