"""Write the reference files of the C++ port of the SCALAR DistilBERT and CRF model.

For each row of identifiers_tag.tsv, the script splits the name with spiral.ronin.split and tags the
words with DistilBertTagger.tag_identifier of SCALAR (scanl_tagger, branch scalar-3-0). A forward hook
on the upstream DistilBertCRFForTokenClassification records the input ids and the word mask that
tag_identifier builds and the emissions of the model. Each row of a reference file holds these inputs,
the emissions at the word positions, the labels of tag_identifier, and the Viterbi margin between the
best and the second-best label sequence. Rows that raise the too-long error keep the status too-long.

The script writes the full corpus reference, the golden subset for ctest, the cases of the CRF test,
which compare the C++ Viterbi decoder with torchcrf on synthetic emissions, and the kernel cases, which
hold inputs and outputs of the torch kernels for GELU, the layer norm, a linear layer, and the attention.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from spiral import ronin
from torchcrf import CRF

# The checkout of scanl_tagger, the model folder, and the corpus come from the environment, since the
# module of the checkout loads at import time (PORT.md, Upgrading SCALAR)
UPSTREAM_DIR = os.environ.get("SCALAR_CHECKOUT", "")
MODEL_DIR = os.environ.get("SCALAR_MODEL", "")
CORPUS = os.environ.get("SCALAR_CORPUS", "")
if not (UPSTREAM_DIR and MODEL_DIR and CORPUS):
    raise SystemExit("set SCALAR_CHECKOUT, SCALAR_MODEL, and SCALAR_CORPUS")
sys.path.insert(0, UPSTREAM_DIR)

# pylint: disable=wrong-import-position
from src.lm_based_tagger.distilbert_tagger import DistilBertTagger

GOLDEN_LOWEST_MARGINS = 150
GOLDEN_LONGEST = 20
GOLDEN_TOTAL = 1000
CRF_RANDOM_CASES = 300
CRF_TIE_CASES = 300
KERNEL_SEED = 20261106


def parse_arguments():
    """Read the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--corpus-out", help="path of the full corpus reference")
    parser.add_argument("--golden-dir", help="folder of the golden rows and the CRF cases; needs --corpus-out")
    parser.add_argument("--kernel-cases", help="path of the kernel cases")
    parser.add_argument("--threads", type=int, help="torch threads; omitted, torch keeps its default")
    parser.add_argument("--limit", type=int, help="tag only the first rows of the corpus")
    arguments = parser.parse_args()
    if arguments.golden_dir and not arguments.corpus_out:
        parser.error("--golden-dir needs --corpus-out")
    if not (arguments.corpus_out or arguments.kernel_cases):
        parser.error("name --corpus-out or --kernel-cases")
    return arguments


def read_corpus(limit):
    """Rows of the corpus as (row index, name, context), the header excluded."""
    with open(CORPUS, encoding="utf-8") as corpus:
        lines = corpus.read().split("\n")[1:]
    rows = [line.split("\t") for line in lines if line]
    rows = [(index, fields[0], fields[1]) for index, fields in enumerate(rows)]
    return rows if limit is None else rows[:limit]


def viterbi_margin(emissions, crf):
    """Score of the best label sequence minus that of the second best, in float64."""
    start = crf.start_transitions.detach().double().numpy()
    end = crf.end_transitions.detach().double().numpy()
    transitions = crf.transitions.detach().double().numpy()
    scores = emissions.astype(np.float64)
    best = start + scores[0]
    second = np.full_like(best, -np.inf)
    for step in range(1, len(scores)):
        candidates = np.concatenate([best[:, None] + transitions, second[:, None] + transitions]) + scores[step]
        ordered = np.sort(candidates, axis=0)
        best, second = ordered[-1], ordered[-2]
    final = np.sort(np.concatenate([best + end, second + end]))
    return float(final[-1] - final[-2])


def float_text(values):
    """Space-separated values that round-trip as float32."""
    return " ".join(f"{float(value):.9g}" for value in values)


def model_fields(captured, tags, label2id, crf):
    """Input ids, word positions, labels, emissions, and margin of one call that the forward hook recorded."""
    positions = torch.nonzero(captured["kwargs"]["word_mask"][0]).flatten()
    emissions = captured["output"]["logits"][0][positions].float().numpy()
    return {"status": "ok", "ids": captured["kwargs"]["input_ids"][0].tolist(), "positions": positions.tolist(),
            "labels": [label2id[tag] for tag in tags], "emissions": emissions,
            "margin": viterbi_margin(emissions, crf) if tags else float("nan")}


def tag_corpus(tagger, rows):
    """Reference record of each row, as a dict."""
    crf = tagger.model._constrained_crf()  # pylint: disable=protected-access
    label2id = {label: index for index, label in tagger.id2label.items()}
    captured = {}
    tagger.model.register_forward_hook(
        lambda module, args, kwargs, output: captured.update(kwargs=kwargs, output=output), with_kwargs=True)
    records = []
    started = time.perf_counter()
    for count, (index, name, context) in enumerate(rows, 1):
        words = ronin.split(name)
        record = {"row": index, "name": name, "context": context, "words": words}
        captured.clear()
        try:
            tags = tagger.tag_identifier(tokens=words, context=context, type_str="", language="", system_name="")
        except ValueError:
            record["status"] = "too-long"
        else:
            record.update(model_fields(captured, tags, label2id, crf))
        records.append(record)
        if count % 2000 == 0:
            print(f"{count} of {len(rows)} rows, {time.perf_counter() - started:.0f} s", flush=True)
    return records


def write_records(path, records, header):
    """One tab-separated line per row, after the header comment lines."""
    with open(path, "w", encoding="utf-8") as out:
        out.writelines(f"# {line}\n" for line in header)
        out.write("# row\tname\tcontext\tstatus\tinput ids\tword positions\tlabels\temissions\tmargin\n")
        for record in records:
            fields = [str(record["row"]), record["name"], record["context"], record["status"]]
            if record["status"] == "ok":
                fields += [" ".join(map(str, record["ids"])), " ".join(map(str, record["positions"])),
                           " ".join(map(str, record["labels"])), float_text(record["emissions"].ravel()),
                           f"{record['margin']:.9g}"]
            else:
                fields += [""] * 5
            out.write("\t".join(fields) + "\n")


def golden_subset(records):
    """The rows with the lowest margins, the longest rows, the rows without words, and a random sample.

    Rows with the status too-long stay out, since the model never sees their input.
    """
    tagged = [record for record in records if record["status"] == "ok"]
    with_words = [record for record in tagged if record["labels"]]
    chosen = {record["row"] for record in sorted(with_words, key=lambda r: r["margin"])[:GOLDEN_LOWEST_MARGINS]}
    chosen |= {record["row"] for record in sorted(tagged, key=lambda r: -len(r["ids"]))[:GOLDEN_LONGEST]}
    chosen |= {record["row"] for record in tagged if not record["labels"]}
    remaining = [record["row"] for record in tagged if record["row"] not in chosen]
    random.Random(20261106).shuffle(remaining)
    chosen |= set(remaining[:max(0, GOLDEN_TOTAL - len(chosen))])
    return [record for record in records if record["row"] in chosen]


def crf_block(crf, generator, integer_scores):
    """Lines of one CRF and its cases: random float32 emissions, or small integers that make ties."""
    tags = crf.num_tags
    lines = [f"crf {tags}", "start " + float_text(crf.start_transitions.detach().numpy()),
             "end " + float_text(crf.end_transitions.detach().numpy()),
             "transitions " + float_text(crf.transitions.detach().numpy().ravel())]
    for case in range(CRF_TIE_CASES if integer_scores else CRF_RANDOM_CASES):
        length = 1 + case % 12
        if integer_scores:
            emissions = torch.from_numpy(generator.integers(-2, 3, size=(1, length, tags)).astype(np.float32))
        else:
            emissions = torch.from_numpy((generator.standard_normal((1, length, tags)) * 4).astype(np.float32))
        path = crf.decode(emissions)[0]
        lines.append(f"case {length} " + float_text(emissions.numpy().ravel()) + " path " + " ".join(map(str, path)))
    return lines


def write_crf_cases(path, model):
    """CRF cases with the constrained transitions of the model and with integer scores that tie."""
    generator = np.random.default_rng(20261106)
    lines = crf_block(model._constrained_crf(), generator, False)  # pylint: disable=protected-access
    tied = CRF(model.crf.num_tags, batch_first=True)
    with torch.no_grad():
        tied.start_transitions.copy_(torch.from_numpy(generator.integers(-1, 2, size=tied.num_tags).astype(np.float32)))
        tied.end_transitions.copy_(torch.from_numpy(generator.integers(-1, 2, size=tied.num_tags).astype(np.float32)))
        tied.transitions.copy_(torch.from_numpy(
            generator.integers(-1, 2, size=(tied.num_tags, tied.num_tags)).astype(np.float32)))
    lines += crf_block(tied, generator, True)
    with open(path, "w", encoding="utf-8") as out:
        out.write("# CRF cases: emissions of length x tags in row-major order and the torchcrf 0.7.2 path\n")
        out.write("\n".join(lines) + "\n")


def bits_text(values):
    """Space-separated float32 bit patterns in hexadecimal, which keep every value exactly."""
    return " ".join(f"{bits:08x}" for bits in np.ascontiguousarray(values, dtype=np.float32).ravel().view(np.uint32))


def kernel_block(name, sizes, arrays):
    """Lines of one kernel case: its name and sizes, then one line per array."""
    lines = [f"kernel {name} " + " ".join(map(str, sizes))]
    lines += [f"array {np.asarray(array).size} " + bits_text(array) for array in arrays]
    return lines


def heads_side_by_side(tensor):
    """A [1, heads, length, head size] tensor as length rows of all heads, the layout of the linear layers."""
    return tensor[0].transpose(0, 1).reshape(tensor.shape[2], -1)


def kernel_lines(generator):
    """Lines of the GELU, layer norm, linear, and attention cases."""
    special = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 0.5, -0.5, 5.0, -5.0, 10.0, -10.0, 13.2, -13.2, 30.0, -30.0])
    gelu_input = torch.cat([torch.randn(4082, generator=generator) * 3, special])
    lines = kernel_block("gelu", [gelu_input.numel()], [gelu_input, F.gelu(gelu_input)])
    norm_input = torch.randn(6, 768, generator=generator) * 2 + 0.3
    gamma = torch.randn(768, generator=generator) * 0.5 + 1
    beta = torch.randn(768, generator=generator) * 0.1
    lines += kernel_block("layernorm", [6, 768],
                          [norm_input, gamma, beta, F.layer_norm(norm_input, (768,), gamma, beta, 1e-12)])
    linear_input = torch.randn(36, 512, generator=generator)
    weight = torch.randn(32, 512, generator=generator) * 0.05
    bias = torch.randn(32, generator=generator) * 0.1
    lines += kernel_block("linear", [36, 512, 32], [linear_input, weight, bias, F.linear(linear_input, weight, bias)])
    query, key, value = (torch.randn(1, 2, 30, 64, generator=generator) for _ in range(3))
    attention = F.scaled_dot_product_attention(query, key, value, scale=64 ** -0.5)
    lines += kernel_block("attention", [30, 2, 64], [heads_side_by_side(tensor) for tensor in (query, key, value, attention)])
    return lines


def write_kernel_cases(path):
    """Inputs and outputs of the torch kernels that the C++ kernels reproduce bit for bit.

    The cases run on one thread. The linear case has 36 rows, a multiple of the 6-row tile of the MKL
    sgemm, and the attention case has 30 tokens, since other sizes and thread counts make MKL change the
    summation order of some outputs.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    lines = ["# Kernel cases: float32 bit patterns of the inputs and the torch outputs, one array per line"]
    lines += kernel_lines(torch.Generator().manual_seed(KERNEL_SEED))
    torch.set_num_threads(threads)
    with open(path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")


def main():
    """Tag the corpus and write the reference files that the command line names."""
    arguments = parse_arguments()
    if arguments.threads:
        torch.set_num_threads(arguments.threads)
    if arguments.kernel_cases:
        write_kernel_cases(arguments.kernel_cases)
    if not arguments.corpus_out:
        return
    tagger = DistilBertTagger(MODEL_DIR, local=True, device="cpu")
    records = tag_corpus(tagger, read_corpus(arguments.limit))
    header = [(f"SCALAR scalar-3-0 33f0327, model {os.path.basename(os.path.normpath(MODEL_DIR))}, "
               f"corpus {os.path.basename(CORPUS)}"),
              (f"torch {torch.__version__}, {torch.get_num_threads()} threads, CPU capability "
               f"{torch.backends.cpu.get_cpu_capability()}")]
    write_records(arguments.corpus_out, records, header)
    statuses = [record["status"] for record in records]
    print(f"rows {len(records)}, tagged {statuses.count('ok')}, too long {statuses.count('too-long')}")
    if arguments.golden_dir:
        golden = golden_subset(records)
        write_records(f"{arguments.golden_dir}/golden_rows.tsv", golden, header)
        write_crf_cases(f"{arguments.golden_dir}/crf_cases.txt", tagger.model)
        print(f"golden rows {len(golden)}")


if __name__ == "__main__":
    main()
