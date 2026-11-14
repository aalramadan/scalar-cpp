"""Reference encodings from the upstream SCALAR Python code for the tests of the C++ encoder.

ReferenceEncoder loads DistilBertTagger of scanl_tagger scalar-3-0 as the SCALAR server does
and then replaces its network with a recorder. tag_identifier therefore runs steps 1 to 3 as
upstream code, and the recorder keeps the input ids, the attention mask, and the word mask of
the forward pass. The module also writes the case format that the C++ tests read.
"""

import csv
import importlib
import struct
import sys

TOO_LONG = "Identifier is too long for the model input."
FEATURE_SETS = ("legacy", "all10")
CONTEXTS = ("FUNCTION", "CLASS", "ATTRIBUTE", "DECLARATION", "PARAMETER")


def make_recorder(base, captured):
    """Return a stand-in for the CRF network that records its inputs and predicts label 0."""

    class Recorder(base):
        """Passes the isinstance check of tag_identifier for the CRF model."""

        def __init__(self):  # pylint: disable=super-init-not-called
            """Hold no weights."""

        def __call__(self, input_ids, attention_mask, word_mask):
            captured.append((input_ids[0].tolist(), attention_mask[0].tolist(), word_mask[0].tolist()))
            return {"predictions": [[0] * input_ids.shape[1]]}

    return Recorder()


class ReferenceEncoder:
    """Steps 1 to 3 of DistilBertTagger.tag_identifier, run as upstream code."""

    def __init__(self, scalar_dir, model_dir):
        # The SCALAR checkout is a read-only input, therefore its modules load without bytecode caches
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(scalar_dir))
        tagger_module = importlib.import_module("src.lm_based_tagger.distilbert_tagger")
        crf_module = importlib.import_module("src.lm_based_tagger.distilbert_crf")
        preprocessing = importlib.import_module("src.lm_based_tagger.distilbert_preprocessing")
        self.tagger = tagger_module.DistilBertTagger(str(model_dir), local=True, device="cpu")
        if getattr(self.tagger.model.config, "selected_features", None) is not None:
            raise AssertionError("the checkpoint saves a feature list; the legacy set does not apply")
        self.features = {
            "legacy": list(self.tagger.selected_features),
            "all10": preprocessing.normalize_selected_features(preprocessing.AVAILABLE_FEATURES),
        }
        self.preprocessing = preprocessing
        self.captured = []
        self.tagger.model = make_recorder(crf_module.DistilBertCRFForTokenClassification, self.captured)

    def encode(self, feature_set, words, context, type_str, language, system_name):
        """Return (input ids, word positions), or None when tag_identifier raises the too-long ValueError."""
        self.tagger.selected_features = self.features[feature_set]
        self.captured.clear()
        try:
            self.tagger.tag_identifier(list(words), context, type_str, language, system_name)
        except ValueError as error:
            if str(error) != TOO_LONG:
                raise
            return None
        input_ids, attention_mask, word_mask = self.captured[0]
        if any(value != 1 for value in attention_mask):
            raise AssertionError("the attention mask holds padding")
        return input_ids, [index for index, flag in enumerate(word_mask) if flag]


def escape(text):
    """Escape a field of the case format: backslash, tab, newline, and the other control bytes."""
    parts = []
    for char in text:
        code = ord(char)
        if code > 0x7F:
            raise ValueError(f"the cases hold ASCII only: {text!r}")
        if char == "\\":
            parts.append("\\\\")
        elif char == "\t":
            parts.append("\\t")
        elif char == "\n":
            parts.append("\\n")
        elif code < 0x20 or code == 0x7F:
            parts.append(f"\\x{code:02x}")
        else:
            parts.append(char)
    return "".join(parts)


def case_line(group, feature_set, row, words, result):
    """One line of the case format.

    The fields are the group, the feature set, the context, the type, the language, the system
    name, the expected input ids, the expected word positions, and then one field per word. The
    two expected fields hold ! for the too-long error.
    """
    if result is None:
        ids = positions = "!"
    else:
        ids = " ".join(map(str, result[0]))
        positions = " ".join(map(str, result[1]))
    fields = [escape(group), feature_set] + [escape(value) for value in row] + [ids, positions]
    return "\t".join(fields + [escape(word) for word in words])


def digest(result):
    """FNV-1a hash (32 bits) of the input ids, a separator, and the word positions, or of the too-long marker."""
    values = [0xFFFFFFFE] if result is None else result[0] + [0xFFFFFFFF] + result[1]
    value_hash = 0x811C9DC5
    for byte in struct.pack(f"<{len(values)}I", *values):
        value_hash = ((value_hash ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value_hash


def read_tag_corpus(path):
    """Return (name, context) of each row of identifiers_tag.tsv."""
    with open(path, newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))
    if rows[0] != ["name", "context", "source"]:
        raise ValueError(f"unexpected header in {path}")
    return [(name, context) for name, context, _ in rows[1:]]


def read_tagger_rows(path):
    """Return (name, context, type, language, system name) of each row of input/tagger_data.tsv of SCALAR."""
    with open(path, newline="", encoding="ascii") as handle:
        return [
            (row["ORIGINAL_NAME"], row["CONTEXT"], row["TYPE"], row["LANGUAGE"], row["SYSTEM_NAME"])
            for row in csv.DictReader(handle, delimiter="\t")
        ]
