"""Pin scalar-cpp to the SCALAR release of a scanl_tagger checkout.

Writes cmake/scalarRelease.cmake and src/scalarRelease.hpp from the checkout and the Hugging Face model
of the release, checks the bundled English words against the NLTK words of the port, and lists the files
that still name the previous release. It reads the Python files without importing them.
"""

import argparse
import ast
import functools
import gzip
import hashlib
import io
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CMAKE_FILE = REPO / "cmake" / "scalarRelease.cmake"
HEADER_FILE = REPO / "src" / "scalarRelease.hpp"
PREPROCESSING = "scalar_tagger/lm_based_tagger/distilbert_preprocessing.py"
CRF = "scalar_tagger/lm_based_tagger/distilbert_crf.py"
CLI = "scalar_tagger/cli.py"
VERSION = "scalar_tagger/version.py"
BUNDLED_WORDS = "scalar_tagger/data/english_words.txt.gz"

# The constants that the C++ code reads, each with its Python file and, for a class attribute, its class
HEADER_CONSTANTS = [
    (CRF, "DistilBertCRFForTokenClassification", "FORBIDDEN_TRANSITIONS"),
    (PREPROCESSING, None, "CONTEXT_MAP"),
    (PREPROCESSING, None, "AVAILABLE_FEATURES"),
    (PREPROCESSING, None, "DEFAULT_FEATURES"),
    (PREPROCESSING, None, "INTERNAL_FEATURES"),
    (PREPROCESSING, None, "FEATURE_ORDER"),
    (PREPROCESSING, None, "LEGACY_FEATURES"),
    (PREPROCESSING, None, "TYPE_BOOL_TOKENS"),
    (PREPROCESSING, None, "TYPE_VOID_TOKENS"),
    (PREPROCESSING, None, "TYPE_INT_TOKENS"),
    (PREPROCESSING, None, "TYPE_FLOAT_TOKENS"),
    (PREPROCESSING, None, "TYPE_CHAR_TOKENS"),
    (PREPROCESSING, None, "TYPE_STRING_TOKENS"),
    (PREPROCESSING, None, "TYPE_CONTAINER_TOKENS"),
    (PREPROCESSING, None, "TYPE_FUNCTION_TOKENS"),
    (PREPROCESSING, None, "TYPE_QUALIFIER_TOKENS"),
    (PREPROCESSING, None, "NON_PLURAL_S_SUFFIX_STOPLIST"),
]

# The files of a model folder that the port reads, in download order; the others of the model stay out
REQUIRED_MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
OPTIONAL_MODEL_FILES = ("vocab.txt", "special_tokens_map.json", "added_tokens.json")
NLTK_WORD_FILES = ("words/en", "words/en-basic")
SAFE_BUILTINS = {"set": set, "frozenset": frozenset, "list": list, "tuple": tuple, "dict": dict, "sorted": sorted}


@functools.cache
def assigned_constants(path: Path, class_name: str | None) -> dict:
    """Evaluate the assignments of a module or a class body that need no import or call of the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    body = tree.body
    if class_name is not None:
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
        if not classes:
            sys.exit(f"{path}: no class {class_name}")
        body = classes[0].body
    values = {}
    for node in body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, expression = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            name, expression = node.target.id, node.value
        else:
            continue
        code = compile(ast.Expression(expression), str(path), "eval")
        # Only expressions of the release checkout run here, with no builtins beyond SAFE_BUILTINS
        try:
            values[name] = eval(code, {"__builtins__": SAFE_BUILTINS, **values})
        except (NameError, AttributeError, TypeError, KeyError, IndexError, ValueError):
            pass  # an assignment that needs the module at run time, such as a call of os.path
    return values


def constant(checkout: Path, file: str, class_name: str | None, name: str):
    """Return one constant of the checkout, or stop with the place where it is missing."""
    values = assigned_constants(checkout / file, class_name)
    if name not in values:
        owner = f"{file}, class {class_name}" if class_name else file
        sys.exit(f"{owner} no longer assigns {name}; port the change and update HEADER_CONSTANTS")
    return values[name]


def git(checkout: Path, *arguments: str) -> str:
    """Return the output of a git command in the checkout."""
    return subprocess.run(["git", "-C", str(checkout), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def fetch(url: str) -> bytes:
    """Return the body of a public URL."""
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def model_files(repo: str, revision: str) -> list[tuple[str, str]]:
    """Return the name and the SHA-256 of each model file that the port reads, from the Hugging Face API."""
    info = json.loads(fetch(f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"))
    siblings = {entry["rfilename"]: entry for entry in info.get("siblings", [])}
    missing = [name for name in REQUIRED_MODEL_FILES if name not in siblings]
    if missing:
        sys.exit(f"{repo}@{revision} lacks {', '.join(missing)}, which the port reads")
    pinned = []
    for name in REQUIRED_MODEL_FILES[:-1] + OPTIONAL_MODEL_FILES + REQUIRED_MODEL_FILES[-1:]:
        if name not in siblings:
            continue
        lfs = siblings[name].get("lfs")
        digest = lfs["sha256"] if lfs else hashlib.sha256(fetch(f"https://huggingface.co/{repo}/resolve/{revision}/{name}")).hexdigest()
        pinned.append((name, digest))
    return pinned


def cmake_value(text: str, name: str) -> str:
    """Return the quoted value of a set() of the CMake file."""
    match = re.search(rf'^set\({name} +"([^"]*)"\)', text, re.MULTILINE)
    if match is None:
        sys.exit(f"{CMAKE_FILE} has no set({name} ...)")
    return match.group(1)


def cmake_text(tag: str, commit: str, repo: str, revision: str, files: list, nltk_url: str, nltk_sha256: str) -> str:
    """Return cmake/scalarRelease.cmake for a release."""
    width = max(len(name) for name, _ in files)
    file_lines = "\n".join(f"    {name.ljust(width)} {digest}" for name, digest in files)
    return f"""# @file scalarRelease.cmake
# The SCALAR release that scalar-cpp ports, written by tools/update_release.py as PORT.md describes

# The Python SCALAR that the port follows, which tools/compare_with_python.sh requires installed
set(SCALAR_UPSTREAM_TAG    "{tag}")
set(SCALAR_UPSTREAM_COMMIT "{commit}")

# Its release model, RELEASE_MODEL and RELEASE_REVISION of scalar_tagger/cli.py, with the SHA-256 of
#   each file of the model that the port reads
set(SCALAR_MODEL_REPO     "{repo}")
set(SCALAR_MODEL_REVISION "{revision}")
set(SCALAR_MODEL_FILES
{file_lines}
)

# The NLTK words corpus, the source of {BUNDLED_WORDS} that the dictionary
#   flag reads, pinned to its nltk_data commit
set(NLTK_WORDS_URL    "{nltk_url}")
set(NLTK_WORDS_SHA256 "{nltk_sha256}")
"""


def camel_case(name: str) -> str:
    """Return the C++ name of a Python constant, such as typeBoolTokens for TYPE_BOOL_TOKENS."""
    first, *rest = name.lower().split("_")
    return first + "".join(part.capitalize() for part in rest)


def quoted(text: str) -> str:
    """Return a C++ string literal of an ASCII string."""
    if not text.isascii() or any(c in text for c in '"\\\n\r\t'):
        sys.exit(f"the constant value {text!r} needs escapes, which the generator does not write")
    return f'"{text}"'


def wrapped(items: list[str], indent: str = "    ", width: int = 112) -> str:
    """Return the items joined by commas over lines of at most the given width."""
    lines, line = [], indent
    for item in items:
        piece = item + ","
        if line.strip() and len(line) + 1 + len(piece) > width:
            lines.append(line.rstrip())
            line = indent
        line += piece if not line.strip() else " " + piece
    lines.append(line.rstrip())
    return "\n".join(lines)


def cpp_constant(file: str, class_name: str | None, name: str, value) -> str:
    """Return the C++ declaration of one Python constant: a list keeps its order, a set is sorted."""
    origin = f"{name} of {file}" if class_name is None else f"{class_name}.{name} of {file}"
    if isinstance(value, dict):
        value = list(value.items())
    elif isinstance(value, (set, frozenset)):
        value = sorted(value)
        origin += ", sorted"
    if not isinstance(value, (list, tuple)) or not value:
        sys.exit(f"{origin} is no non-empty list, set, or dict")
    strings = [item for item in value if isinstance(item, str)]
    pairs = [item for item in value if isinstance(item, tuple) and len(item) == 2 and all(isinstance(part, str) for part in item)]
    if len(strings) == len(value):
        kind, items = "std::string_view", [quoted(item) for item in strings]
    elif len(pairs) == len(value):
        kind = "std::pair<std::string_view, std::string_view>"
        items = [f"{{{quoted(first)}, {quoted(second)}}}" for first, second in pairs]
    else:
        sys.exit(f"{origin} holds values other than strings or pairs of strings")
    return f"// {origin}\ninline constexpr {kind} {camel_case(name)}[] = {{\n{wrapped(items)}\n}};\n"


def header_text(tag: str, declarations: list[str]) -> str:
    """Return src/scalarRelease.hpp for a release."""
    body = "\n".join(declarations)
    return f"""// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file scalarRelease.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// The constants of SCALAR {tag} that the port reads, under the camelCase form of their Python names.
// tools/update_release.py writes this file from a checkout of the release; edit the script, never the file.

#ifndef SCALAR_RELEASE_HPP
#define SCALAR_RELEASE_HPP

#include <string_view>
#include <utility>

namespace scalar::release {{

{body}
}} // namespace scalar::release

#endif
"""


def nltk_words(url: str, sha256: str) -> set[str]:
    """Return the words that the port reads from the pinned NLTK corpus, lowercased as ScalarTagger does."""
    data = fetch(url)
    if hashlib.sha256(data).hexdigest() != sha256:
        sys.exit(f"{url} no longer has the SHA-256 of {CMAKE_FILE.name}")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        text = "".join(archive.read(name).decode("utf-8") for name in NLTK_WORD_FILES)
    return {line.lower() for line in text.splitlines() if line.rstrip()}


def write_crlf(path: Path, text: str) -> bool:
    """Write the text with CRLF line ends, as the other sources of the folder; return whether it changed."""
    data = text.replace("\n", "\r\n").encode("ascii")
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    return True


def main() -> None:
    """Pin the release of the checkout and report what still names the previous one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="scanl_tagger checked out at the release tag")
    checkout = parser.parse_args().checkout.resolve()

    try:
        tag = git(checkout, "describe", "--tags", "--exact-match", "HEAD")
    except subprocess.CalledProcessError:
        sys.exit(f"{checkout} is not checked out at a tag")
    commit = git(checkout, "rev-parse", "HEAD^{commit}")
    version = constant(checkout, VERSION, None, "__version__")
    if f"v{version}" != tag:
        sys.exit(f"the tag {tag} differs from __version__ {version} of {VERSION}")
    repo = constant(checkout, CLI, None, "RELEASE_MODEL")
    revision = constant(checkout, CLI, None, "RELEASE_REVISION")

    previous = CMAKE_FILE.read_text(encoding="ascii").replace("\r\n", "\n")
    nltk_url = cmake_value(previous, "NLTK_WORDS_URL")
    nltk_sha256 = cmake_value(previous, "NLTK_WORDS_SHA256")
    old = {name: cmake_value(previous, name) for name in ("SCALAR_UPSTREAM_TAG", "SCALAR_MODEL_REPO", "SCALAR_MODEL_REVISION")}

    files = model_files(repo, revision)
    declarations = [cpp_constant(file, class_name, name, constant(checkout, file, class_name, name))
                    for file, class_name, name in HEADER_CONSTANTS]
    for path, text in ((CMAKE_FILE, cmake_text(tag, commit, repo, revision, files, nltk_url, nltk_sha256)),
                       (HEADER_FILE, header_text(tag, declarations))):
        print(f"{path.relative_to(REPO)}: {'written' if write_crlf(path, text) else 'unchanged'}")
    print(f"SCALAR {tag} ({commit[:7]}), model {repo} at {revision[:7]}: {', '.join(name for name, _ in files)}")

    with gzip.open(checkout / BUNDLED_WORDS, "rt", encoding="utf-8") as handle:
        bundled = set(handle.read().split())
    port = nltk_words(nltk_url, nltk_sha256)
    if bundled != port:
        sys.exit(f"{BUNDLED_WORDS} holds {len(bundled - port)} words that the NLTK words lack and lacks "
                 f"{len(port - bundled)} of them; the dictionary of the port must follow the release")
    print(f"the dictionary: {BUNDLED_WORDS} holds the {len(port)} NLTK words of the port")

    stale = [value for value in (old["SCALAR_UPSTREAM_TAG"], old["SCALAR_MODEL_REPO"], old["SCALAR_MODEL_REVISION"][:7])
             if value not in (tag, repo, revision[:7])]
    if stale:
        arguments = ["grep", "-n", "-I", "-F"] + [part for value in stale for part in ("-e", value)]
        arguments += ["--", ".", ":!tests/data", ":!tools/identifiers_tag.tsv"]
        hits = subprocess.run(["git", "-C", str(REPO), *arguments], check=False, capture_output=True, text=True).stdout
        print("files that still name the previous release:\n" + (hits or "none\n"), end="")


if __name__ == "__main__":
    main()
