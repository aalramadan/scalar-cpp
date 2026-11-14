# SCALAR port

This repository holds a C++17 port of the inference of the SCALAR part-of-speech tagger. With the port, a C++ program, such as nameChecker, tags identifier names without a tagger server and without Python. The port follows the server path of SCALAR, `listen()` and `dictionary_lookup()` of `src/tag_identifier.py`, and loads the published model files unchanged. It uses only the C++ standard library, the thread library of the system, and `include/json.hpp` of nlohmann/json.

```bash
cmake -B build -G Ninja      # downloads the model files once, see the README
cmake --build build
ctest --test-dir build -R scalar
```

## Upstream versions

| Part | Upstream | Version |
|---|---|---|
| Tagger | [SCANL/scanl_tagger](https://github.com/SCANL/scanl_tagger), branch `scalar-3-0` | commit `33f0327` |
| Model | [sourceslicer/scalar_lm_best](https://huggingface.co/sourceslicer/scalar_lm_best) | revision `faef837c1d5f0f4904a53993aaeee747fec73e89` |
| Splitter | Spiral, cnewman fork | commit `dff5373` |
| Stemmer | NLTK, `EnglishStemmer` | 3.10.3 |
| Dictionary flag | NLTK words corpus, files `en` and `en-basic` | nltk_data commit `b2085c7` |
| Tokenizer | tokenizers and transformers (`DistilBertTokenizerFast`) | 0.23.3 and 5.19.0 |
| Kernels | torch (ATen) and oneDNN, CPU float32 | 2.14.1 and 3.12.0 |

`CMakeLists.txt` pins the model revision, the NLTK commit, and the SHA-256 of each downloaded file.

## How it works

A name and its context, such as `getValue` and `FUNCTION`, pass these steps in order. Every step is static. None calls a model service.

1. `src/ronin.cpp`, `src/heuristicSplitter.cpp`, `src/englishStemmer.cpp`. Splits the name into words, as `spiral.ronin.split` does, with the tables of `data/ronin/`.
2. `src/featureTokens.cpp`. Builds the feature tokens of the context and the position token before each word (`distilbert_preprocessing.py`). `src/sequenceMatcher.cpp` and `src/pythonSum.cpp` port the parts of `difflib` and `sum()` that the features need.
3. `src/wordpieceTokenizer.cpp`, `src/modelInput.cpp`. Turns the tokens into input ids and records the first subtoken of each word (`DistilBertTagger.tag_identifier`). A word without a subtoken after the truncation to 512 ids gives the error `Identifier is too long for the model input.`
4. `src/safetensors.cpp`, `src/linearLayer.cpp`, `src/encoderKernels.cpp`, `src/distilbertCrf.cpp`. Runs the six layers of DistilBERT and the classifier on the CPU and reads the scores of each word (`distilbert_crf.py`).
5. `src/crfDecoder.cpp`. Picks the tag of each word with the Viterbi algorithm of torchcrf, with the transitions N to N, N to NPL, and NPL to NPL forbidden as in scalar-3-0.
6. `src/scalarTagger.cpp`. Sets the dictionary flag of each word, DW for a word of the NLTK words or of the custom word lists in any case, DD for digits, and UC otherwise. It tags many names at once on all cores and returns the JSON answer of the server.

`src/asciiText.cpp`, `src/jsonReader.cpp`, and `src/kernelDispatch.hpp` hold the shared helpers. `tools/scalarTagCorpus.cpp` tags a whole corpus for the comparison with the Python SCALAR.

## Built-in files

A program can keep every file that the tagger loads in its binary, as nameChecker does, therefore a copy of the binary runs without the model folder and without this repository. To put the files into the binary, the CMake function `scalar_embed_files` runs `cmake/embedFiles.cmake`, which writes a C++ file at each build after a change of one of the files. Its `.incbin` directives copy the bytes of each file into the read-only data of the binary. The Microsoft Visual C++ compiler (MSVC) lacks this directive, therefore its build lists the files in a resource script instead.

At run time, `src/dataFile.cpp` opens a path that starts with `builtin:` in the binary and any other path on disk. Each built-in file has a name of the form `builtin:<folder>/<file>`, such as `builtin:model/config.json`. The folder is `model/` for the model files and the NLTK words, `ronin/` for the tables of `data/ronin/`, while a file that the program adds keeps the name it gives, such as `resources/common_abbreviations.txt` of nameChecker. In contrast, the tests and `tools/scalarTagCorpus.cpp` load the files on disk through `folderFiles()`.

At its start, the tagger copies the weights of `model.safetensors` from the binary into its own arrays. After this copy, the 265 MB of the file stay mapped, therefore the resident memory of a run includes these pages. The system shares the pages between runs of the same binary and can drop them under memory pressure.

## Deviations from the Python SCALAR

- Terms with numbers. Spiral matches the terms of `common_terms_with_numbers` in the order of a Python set, which follows `PYTHONHASHSEED`. A server draws a new seed at each start. For 23 pairs of terms that start at the same position, the split of a name therefore depends on the start of the server, such as `TLS_ECDH_anon_WITH_3DES_EDE_CBC_SHA` with `3DES` or `3D` and `ES`. The port always takes the longer term, the one order that keeps every term reachable. In the test corpus, 87 of 268,480 strings change with the seed.
- Text outside ASCII. The splitter treats each byte above 0x7F as a character that is no letter, digit, or space, and the encoder refuses such a byte. nameChecker, for example, sends only ASCII names.
- Dictionary flag. SCALAR adds each line of `words/en.txt` as written and looks up the lower-case form of a word, therefore an entry with a capital letter never matches. The port lowercases each line, as the setup note of the old tagger server told its users to change the server. As a result, 23 entries of `resources/common_abbreviations.txt` of nameChecker, such as `EOF` and `IBM`, give DW in the port and UC in an unchanged server.
- Long names. Ronin splits a run of one letter case by a recursion over its suffixes, whose depth and time grow with the length of the run. Python already stops some runs of 2,000 letters, such as `byby...by`, with a `RecursionError`, and a run of 2,000 letters takes about 3 s in the port. The port therefore answers a name of more than 2,000 characters with the error of a name too long for the model, also where Python tags the name, such as `abab...ab` of 2,002 letters. A token of more than 1,000 characters is split from its shortest suffix up. This order needs no deep recursion on any thread stack and gives the splits of the recursion.
- Rounding. The kernels follow the summation order of torch on the CPU in its main loops, but not in every edge case of its matrix library. The scores of the model differ from those of torch by at most 3.1e-5. Every test gives the tags of Python. The equality rests on these tests, since the smallest margin between two tags that the review found is 1.5 times this difference.
- CPU. The port runs on the CPU only. A single build runs on any CPU of its architecture, since the tagger picks its kernels, the functions that do the math of the network, at run time. On x86-64, the compiler builds two copies of these kernels. One copy uses the AVX2 and FMA instructions of newer CPUs, while the other copy runs on every x86-64 CPU. At its start, the tagger checks the CPU and takes the fast copy wherever the CPU has these instructions. On other CPUs, such as Arm64, GCC and Clang turn the kernels into vector code where the CPU has FMA instructions. An exact emulation of FMA serves the remaining CPUs. On 300 names of the corpus with 16 threads, the AVX2 kernels took 2.15 s, the vector kernels of a portable build for x86-64-v3 took 2.19 s, and the emulation took 16.5 s, each with the tags of the Python SCALAR.
- Server. The server read the name from its URL, therefore it cut a name such as `operator/=` at the slash. Its route `/probe/<cache_id>` came before the route of names, therefore the name `probe` without a cache ID got the probe message of the server in place of tags. The port receives the whole name. In place of the database of the server, nameChecker keeps the answers of a cache ID in a file of the user cache folder.

## Checks

On the 44,366 rows of `tools/identifiers_tag.tsv`, a corpus of names and contexts from public sources, the port gives the answer of the Python SCALAR for every row, the split, the tag, and the dictionary flag of each of 141,796 words. Five rows fail in the port and in Python alike as too long for the model. The parts were checked against Python on larger inputs.

| Part | Inputs compared | Differences |
|---|---|---|
| Splitter | 268,480 corpus strings and 120,000 generated strings | 0 |
| Stemmer | 756,978 words | 0 |
| Encoder | 1,141,127 cases | 0 |
| Model | 44,361 corpus rows and 9,817 generated rows | 0 tags |

ctest runs these parts offline against the golden files of `tests/data/`, which the Python scripts of `tests/` wrote, and runs the whole tagger against 421 answers of the Python SCALAR.

## Upgrading SCALAR

The effort depends on what changes upstream.

| Change upstream | Steps |
|---|---|
| A retrained model of the same kind | Set `SCALAR_MODEL_REVISION` and the SHA-256 of each file in `CMakeLists.txt`, configure again, and run the comparison below. The port reads the sizes, the labels, the selected features, and the vocabulary from the model files. |
| The Python code of SCALAR | Compare the upstream files named in How it works between the two commits, port each change to the matching C++ file, write the golden files again, and run ctest and the comparison below. |
| Spiral or its word tables | Run `tools/export_ronin_data.py` with the new Spiral after updating the commit it checks, then write the splitter golden files again. |

The comparison tags the corpus with the Python SCALAR and with the port and compares the answers word by word. It needs a checkout of the new SCALAR, its model folder, Python with the requirements of that checkout, and the words corpus of NLTK (`python3 -m nltk.downloader words`).

```bash
PYTHONHASHSEED=0 python3 tools/tag_with_python_scalar.py --checkout <scanl_tagger> --model <model folder> \
    --en <word list>... --corpus tools/identifiers_tag.tsv --out python.jsonl
build/scalarTagCorpus tools/identifiers_tag.tsv port.jsonl "$(nproc)" <word list>...
python3 tools/compare_tag_answers.py --first python.jsonl --second port.jsonl --out comparison.json
```

The golden files of ctest come from these commands, run from the repository folder after the comparison. `SCALAR_SPLIT_CORPUS` names a file of one identifier per line. The shipped splitter files came from 268,480 identifier strings of the public sources of the corpus, which the repository does not hold, and any such list serves a new check.

```bash
SCALAR_SPLIT_CORPUS=<one identifier per line> python3 tests/make_golden.py
PYTHONHASHSEED=0 python3 tests/write_golden.py --scalar <scanl_tagger> --model <model folder> --corpus tools/identifiers_tag.tsv
SCALAR_CHECKOUT=<scanl_tagger> SCALAR_MODEL=<model folder> SCALAR_CORPUS=tools/identifiers_tag.tsv PYTHONHASHSEED=0 \
    python3 tests/write_golden_files.py --corpus-out model_reference.tsv --golden-dir tests/data/model --kernel-cases tests/data/model/kernel_cases.txt
python3 tests/write_tagger_golden.py --answers python.jsonl --out tests/data/tagger/golden_answers.jsonl
PYTHONHASHSEED=0 python3 tests/write_long_tokens.py --out tests/data/tagger/long_tokens.tsv
```
