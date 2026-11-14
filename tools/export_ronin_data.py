"""Regenerate the Ronin tables of data/ronin from the installed Spiral package.

Needs Spiral of the cnewman fork at commit dff537320c15849c10e583968036df2d966eddee,
the splitter that SCALAR scalar-3-0 calls, and checks that commit before it writes.
It takes the tables from a Ronin object after the default init(), as SCALAR uses
them, and writes each one as sorted UTF-8 text with one entry per line.

  frequencies.txt                token<TAB>count after the lowercase max-merge of init()
  dictionary.txt                 the words of data/dictionary.pklz
  prefixes.txt                   constants.prefixes
  suffixes.txt                   constants.suffixes
  special_computing_terms.txt    constants.special_computing_terms
  common_terms_with_numbers.txt  constants.common_terms_with_numbers
  common_suffix_numbers.txt      constants.common_suffix_numbers

It also checks that the parameters of init() and the patterns of simple_splitters
equal the values that src/ronin.cpp and src/heuristicSplitter.cpp hold.

Usage: python3 tools/export_ronin_data.py [output folder, default data/ronin]
"""

import importlib.metadata
import inspect
import json
import pathlib
import re
import sys

from spiral import constants, ronin, simple_splitters

SPIRAL_COMMIT = 'dff537320c15849c10e583968036df2d966eddee'

INIT_DEFAULTS = {
    'exact_case': False,
    'low_freq_cutoff': 340,
    'length_cutoff': 2,
    'short_min_freq': 286540,
    'normal_exponent': 0.1527,
    'dict_word_exponent': 0.1228,
    'camel_bias': 8.6319,
    'recognition_bias': 0.0000003635,
    'alt_exponent': 1.2165,
}

SPLITTER_PATTERNS = {
    '_delimiter_chars': '$~_.:/@',
    '_camel_case': r'((?<=[a-z])[A-Z])',
    '_numbers_re': r'(0x[0-9a-f]+l?|0[1-7]{2,}|\d+)',
}

# A term with numbers is a regular expression of literal letters and digits, character
# classes of ranges, and {m}, {m,}, or {m,n} quantifiers, the subset that the C++ port parses.
TERM_GRAMMAR = re.compile(r'(?:(?:[a-z0-9]|\[(?:[a-z0-9](?:-[a-z0-9])?)+\])(?:\{\d+(?:,\d*)?\})?)+')


def check_spiral_commit():
    """Stop unless the installed Spiral comes from the pinned commit."""
    direct_url = importlib.metadata.distribution('spiral').read_text('direct_url.json')
    commit = json.loads(direct_url or '{}').get('vcs_info', {}).get('commit_id')
    if commit != SPIRAL_COMMIT:
        sys.exit(f'Spiral must come from commit {SPIRAL_COMMIT}, found {commit}')


def check_constants_of_port():
    """Stop if init() defaults or splitter patterns differ from the values of the C++ port."""
    defaults = {name: parameter.default
                for name, parameter in inspect.signature(ronin.Ronin.init).parameters.items()
                if name in INIT_DEFAULTS}
    if defaults != INIT_DEFAULTS:
        sys.exit(f'Ronin.init() defaults differ from the C++ port: {defaults}')
    found = {
        '_delimiter_chars': simple_splitters._delimiter_chars,
        '_camel_case': simple_splitters._camel_case.pattern,
        '_numbers_re': simple_splitters._numbers_re.pattern,
    }
    if found != SPLITTER_PATTERNS:
        sys.exit(f'simple_splitters patterns differ from the C++ port: {found}')
    if simple_splitters._numbers_re.flags & re.IGNORECASE == 0:
        sys.exit('simple_splitters._numbers_re lost re.IGNORECASE')
    unparsed = [term for term in constants.common_terms_with_numbers if not TERM_GRAMMAR.fullmatch(term)]
    if unparsed:
        sys.exit(f'terms outside the pattern subset of the C++ port: {unparsed}')


def write_table(path, rows):
    """Write one row per line, its fields joined by tabs, after checking that no field breaks the format."""
    rows = list(rows)
    broken = [row for row in rows if any(not field or any(c in field for c in '\t\n\r') for field in row)]
    if broken:
        sys.exit(f'{path.name}: fields that are empty or hold a tab or a line break: {broken[:5]}')
    path.write_text(''.join('\t'.join(row) + '\n' for row in rows), encoding='utf-8')
    print(f'{path.name}: {len(rows)} entries, {path.stat().st_size} bytes')


def main():
    """Export every table that the C++ Ronin loads."""
    check_spiral_commit()
    check_constants_of_port()
    output = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parent.parent / 'data' / 'ronin'
    output.mkdir(parents=True, exist_ok=True)

    splitter = ronin.Ronin()
    splitter.init()
    frequencies = splitter._frequencies
    if any(key != key.lower() for key in frequencies):
        sys.exit('the frequency table holds a key that is not lowercase after init()')
    if any(not isinstance(count, int) or count <= 0 for count in frequencies.values()):
        sys.exit('the frequency table holds a count that is not a positive integer')

    write_table(output / 'frequencies.txt', ((key, str(frequencies[key])) for key in sorted(frequencies)))
    tables = {
        'dictionary.txt': splitter._dictionary,
        'prefixes.txt': constants.prefixes,
        'suffixes.txt': constants.suffixes,
        'special_computing_terms.txt': constants.special_computing_terms,
        'common_terms_with_numbers.txt': constants.common_terms_with_numbers,
        'common_suffix_numbers.txt': constants.common_suffix_numbers,
    }
    for name, entries in tables.items():
        write_table(output / name, ((entry,) for entry in sorted(entries)))

if __name__ == '__main__':
    main()
