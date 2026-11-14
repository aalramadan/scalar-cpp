"""Shared parts of the Python side of the splitter tests: line format, inputs, and the reference order.

The line format matches tests/escapedLines.hpp: fields split by tabs, a backslash written as \\\\,
and each byte below 0x20, 0x7F, and each byte of 0x80 or more of the UTF-8 text written as \\xHH.

Python orders the alternatives of simple_splitters._exceptions_re by the hash order of a set,
which depends on PYTHONHASHSEED. use_longest_first_terms() rebuilds that regex with the longest
alternatives first, the order under which re picks the longest term at each position, the rule of
the C++ port. Every other line of Spiral and NLTK runs unchanged.
"""

import gzip
import os
import pathlib
import pickle
import random
import re

from spiral import constants, ronin, simple_splitters

# One identifier per line, such as the corpus of the shipped golden files (PORT.md)
SPLIT_CORPUS = pathlib.Path(os.environ.get('SCALAR_SPLIT_CORPUS', 'identifiers_split.txt'))

DELIMITERS = '$~_.:/@'
OPERATORS = '+-*/<>=!&|^~%()[]{},;?\'"#`\\'
UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LOWER = 'abcdefghijklmnopqrstuvwxyz'
DIGITS = '0123456789'
HEX = '0123456789abcdefABCDEF'


def escape_field(text):
    """Escape one field as tests/escapedLines.hpp does."""
    out = []
    for byte in text.encode('utf-8', 'surrogatepass'):
        if byte == 0x5C:
            out.append('\\\\')
        elif byte < 0x20 or byte >= 0x7F:
            out.append(f'\\x{byte:02x}')
        else:
            out.append(chr(byte))
    return ''.join(out)


def unescape_fields(line):
    """Split an escaped line at tabs and decode each field."""
    fields = []
    for field in line.split('\t'):
        raw = bytearray()
        i = 0
        while i < len(field):
            if field[i] != '\\':
                raw.extend(field[i].encode('utf-8'))
                i += 1
            elif field[i + 1] == '\\':
                raw.append(0x5C)
                i += 2
            else:
                raw.append(int(field[i + 2:i + 4], 16))
                i += 4
        fields.append(raw.decode('utf-8', 'surrogatepass'))
    return fields


def use_longest_first_terms():
    """Rebuild _exceptions_re with the longest alternatives first, the C++ rule for competing terms."""
    ordered = sorted(constants.common_terms_with_numbers, key=lambda term: (-len(term), term))
    simple_splitters._exceptions_re = re.compile(r'(' + '|'.join(ordered) + ')', re.IGNORECASE)


def competing_terms():
    """Pairs of terms with numbers that match at one position with different lengths."""
    literals = sorted(t for t in constants.common_terms_with_numbers if re.fullmatch(r'[a-z0-9]+', t))
    patterns = sorted(t for t in constants.common_terms_with_numbers if t not in literals)
    pairs = [(short, long) for short in literals for long in literals if long != short and long.startswith(short)]
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        stem = re.match(r'[a-z0-9]*', pattern).group(0)
        for literal in literals:
            if regex.match(literal) or literal.startswith(stem) or stem.startswith(literal):
                pairs.append((literal, pattern))
    return pairs


def corpus_identifiers():
    """The identifier strings of the split corpus, 268,480 for the shipped golden files."""
    return SPLIT_CORPUS.read_text(encoding='utf-8').split('\n')[:-1]


def spiral_tables():
    """Frequency keys before the merge and the dictionary, as the pickles of Spiral hold them."""
    with gzip.open(ronin._DEFAULT_FREQ_PICKLE, 'rb') as pickle_file:
        frequencies = pickle.load(pickle_file)
    with gzip.open(ronin._DICTIONARY_PICKLE, 'rb') as pickle_file:
        dictionary = pickle.load(pickle_file)
    return sorted(frequencies), sorted(dictionary)


def _term_instance(rng, term):
    """A string that a term with numbers matches, in a random case."""
    def digits(match):
        return ''.join(rng.choice(DIGITS) for _ in range(int(match.group(1)) + rng.randrange(3)))
    return _random_case(rng, re.sub(r'\[0-9\]\{(\d+),\}', digits, term))


def _hex_number(rng):
    """A hexadecimal literal such as 0x1fL."""
    return '0' + rng.choice('xX') + ''.join(rng.choice(HEX) for _ in range(rng.randint(1, 6))) + rng.choice(['', 'l', 'L'])


def _random_case(rng, text):
    style = rng.randrange(4)
    if style == 0:
        return text.upper()
    if style == 1:
        return text.capitalize()
    if style == 2:
        return ''.join(c.upper() if rng.random() < 0.3 else c for c in text)
    return text


def stress_identifiers(seed, count, words, alphabet):
    """Distinct random identifiers of runs of letters, digits, delimiters, spaces, operators, and terms.

    With alphabet 'printable' every character is of 0x20 to 0x7E; with 'ascii' runs of any byte of
    0x00 to 0x7F join in; with 'unicode' runs of non-ASCII letters, digits, and spaces join in.
    """
    rng = random.Random(seed)
    terms = sorted(constants.common_terms_with_numbers)
    affixes = sorted(constants.prefixes | constants.suffixes)
    affixes = [a for a in affixes if ' ' not in a]
    runs = [
        (8, lambda: ''.join(rng.choice(LOWER) for _ in range(rng.randint(1, 8)))),
        (6, lambda: ''.join(rng.choice(UPPER) for _ in range(rng.randint(1, 6)))),
        (5, lambda: ''.join(rng.choice(DIGITS) for _ in range(rng.randint(1, 5)))),
        (3, lambda: ''.join(rng.choice(DELIMITERS) for _ in range(rng.randint(1, 2)))),
        (2, lambda: ' ' * rng.randint(1, 2)),
        (3, lambda: ''.join(rng.choice(OPERATORS) for _ in range(rng.randint(1, 3)))),
        (10, lambda: _random_case(rng, rng.choice(words))),
        (5, lambda: _term_instance(rng, rng.choice(terms))),
        (3, lambda: _random_case(rng, rng.choice(affixes))),
        (2, lambda: _hex_number(rng)),
        (2, lambda: '0' + ''.join(rng.choice('01234567') for _ in range(rng.randint(1, 5)))),
    ]
    if alphabet == 'ascii':
        runs.append((4, lambda: ''.join(chr(rng.randrange(0x80)) for _ in range(rng.randint(1, 3)))))
    elif alphabet == 'unicode':
        extra = 'ÀÉÎõüßÆøŁžıİſK²٣  ’éèñ中文'
        runs.append((6, lambda: ''.join(rng.choice(extra) for _ in range(rng.randint(1, 3)))))
    weights = [w for w, _ in runs]
    makers = [m for _, m in runs]
    found = set()
    result = []
    while len(result) < count:
        text = ''.join(rng.choices(makers, weights)[0]() for _ in range(rng.randint(1, 6)))
        if text not in found:
            found.add(text)
            result.append(text)
    return result


def stress_words(seed, count):
    """Distinct random words that reach each step of the English stemmer, with odd characters mixed in."""
    rng = random.Random(seed)
    endings = ['', 's', "'s", "'s'", "'", 'sses', 'ied', 'ies', 'us', 'ss', 'eedly', 'ingly', 'edly', 'eed',
               'ing', 'ed', 'ation', 'ational', 'tional', 'ization', 'fulness', 'ousness', 'iveness', 'biliti',
               'bli', 'lessli', 'entli', 'alism', 'aliti', 'ousli', 'iviti', 'fulli', 'enci', 'anci', 'abli',
               'izer', 'ator', 'alli', 'logi', 'ogi', 'li', 'cli', 'alize', 'icate', 'iciti', 'ative', 'ical',
               'ness', 'ful', 'ement', 'ance', 'ence', 'able', 'ible', 'ment', 'ant', 'ent', 'ism', 'ate', 'iti',
               'ous', 'ive', 'ize', 'ion', 'sion', 'tion', 'al', 'er', 'ic', 'e', 'l', 'll', 'y', 'ly', 'at',
               'bl', 'iz', 'bb', 'tt', 'ying', 'yed']
    starts = ['', '', '', 'gener', 'commun', 'arsen', "'", 'y', 'Y', 'ye', 'ay', 'oy', 're', 'un']
    pieces = 'bcdfghjklmnpqrstvwxz'
    vowels = 'aeiouy'
    odd = "'-0123456789_#ABCYZ"
    found = set()
    result = []
    while len(result) < count:
        core = ''.join(rng.choice(vowels if rng.random() < 0.4 else pieces) for _ in range(rng.randint(0, 7)))
        word = rng.choice(starts) + core + rng.choice(endings)
        if rng.random() < 0.15:
            position = rng.randrange(len(word) + 1)
            word = word[:position] + rng.choice(odd) + word[position:]
        if rng.random() < 0.1:
            word = word.upper()
        if word not in found:
            found.add(word)
            result.append(word)
    return result
