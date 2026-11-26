# SCALAR port

This repository holds a C++17 port of the inference of the SCALAR part-of-speech tagger. With the port, a C++ program, such as nameChecker, tags identifier names without a tagger server and without Python. The port follows the server path of SCALAR, `listen()` and `dictionary_lookup()` of `scalar_tagger/tag_identifier.py`, and loads the published model files unchanged. It uses only the C++ standard library, the thread library of the system, and `include/json.hpp` of nlohmann/json.

```bash
cmake -B build -G Ninja      # downloads the model files once, see the README
cmake --build build
ctest --test-dir build -R scalar
```

## Upstream versions

| Part | Upstream | Version |
|---|---|---|
| Tagger | [SCANL/scanl_tagger](https://github.com/SCANL/scanl_tagger), branch `scalar_30_experimental` | tag `v3.0.0-rc.3`, commit `728f1e1` |
| Model | [sourceslicer/scalar_lm_v3](https://huggingface.co/sourceslicer/scalar_lm_v3) | revision `5e5e8b1fe1d012a1df523718c013c66bdd12c2ad` |
| Splitter | Spiral, cnewman fork | commit `dff5373` |
| Stemmer | NLTK, `EnglishStemmer` | 3.10.3 |
| Dictionary flag | NLTK words corpus, files `en` and `en-basic`, the source of `scalar_tagger/data/english_words.txt.gz` | nltk_data commit `b2085c7` |
| Tokenizer | tokenizers and transformers (`DistilBertTokenizerFast`) | 0.23.3 and 5.19.0 |
| Kernels | torch (ATen) and oneDNN, CPU float32 | 2.14.1 and 3.12.0 |

`cmake/scalarRelease.cmake` names the release and pins the model revision, the NLTK commit, and the SHA-256 of each downloaded file. `tools/update_release.py` writes this file, as Upgrading SCALAR describes.

## How it works

A name and its context, such as `getValue` and `FUNCTION`, pass these steps in order. Every step is static. None calls a model service.

1. `src/ronin.cpp`, `src/heuristicSplitter.cpp`, `src/englishStemmer.cpp`. Splits the name into words, as `spiral.ronin.split` does, with the tables of `data/ronin/`.
2. `src/featureTokens.cpp`. Builds the feature tokens that the model selects in its `config.json`, the context and the declared type for the release model, and the position token before each word (`distilbert_preprocessing.py`). `src/sequenceMatcher.cpp` and `src/pythonSum.cpp` port the parts of `difflib` and `sum()` that the features need.
3. `src/wordpieceTokenizer.cpp`, `src/modelInput.cpp`. Reads the vocabulary from `vocab.txt` or, for a model without `vocab.txt`, from `tokenizer.json`. Turns the tokens into input ids and records the first subtoken of each word (`DistilBertTagger.tag_identifier`). A word without a subtoken after the truncation to `model_max_length` of `tokenizer_config.json`, 512 ids for the release model, gives the error `Identifier is too long for the model input.`
4. `src/safetensors.cpp`, `src/linearLayer.cpp`, `src/encoderKernels.cpp`, `src/distilbertCrf.cpp`. Runs the six layers of DistilBERT and the classifier on the CPU and reads the scores of each word (`distilbert_crf.py`).
5. `src/crfDecoder.cpp`. Picks the tag of each word with the Viterbi algorithm of torchcrf, with the 15 tag pairs of `FORBIDDEN_TRANSITIONS` forbidden, such as N to N and any tag but PRE to PRE.
6. `src/scalarTagger.cpp`. Passes the context, the type, the language, and the system of a name to the encoder. Sets the dictionary flag of each word, DW for a word of the NLTK words or of the custom word lists in any case, DD for digits, and UC otherwise. It tags many names at once on all cores and returns the JSON answer of the server.

`src/scalarRelease.hpp` holds the constants of the Python SCALAR that the port reads, such as the forbidden tag pairs, the feature lists, and the word sets of the type feature. `src/nameInContext.hpp` holds the inputs of a name, which the tagger passes to the feature code unchanged. `src/asciiText.cpp`, `src/jsonReader.cpp`, and `src/kernelDispatch.hpp` hold the shared helpers. `tools/scalarTagCorpus.cpp` tags a whole corpus for the comparison with the Python SCALAR.

## Built-in files

A program can keep every file that the tagger loads in its binary, as nameChecker does, therefore a copy of the binary runs without the model folder and without this repository. To put the files into the binary, the CMake function `scalar_embed_files` runs `cmake/embedFiles.cmake`, which writes a C++ file at each build after a change of one of the files. Its `.incbin` directives copy the bytes of each file into the read-only data of the binary. The Microsoft Visual C++ compiler (MSVC) lacks this directive, therefore its build lists the files in a resource script instead.

At run time, `src/dataFile.cpp` opens a path that starts with `builtin:` in the binary and any other path on disk. Each built-in file has a name of the form `builtin:<folder>/<file>`, such as `builtin:model/config.json`. The folder is `model/` for the model files and the NLTK words, `ronin/` for the tables of `data/ronin/`, while a file that the program adds keeps the name it gives, such as `resources/common_abbreviations.txt` of nameChecker. In contrast, the tests and `tools/scalarTagCorpus.cpp` load the files on disk through `folderFiles()`.

At its start, the tagger copies the weights of `model.safetensors` from the binary into its own arrays. After this copy, the 265 MB of the file stay mapped, therefore the resident memory of a run includes these pages. The system shares the pages between runs of the same binary and can drop them under memory pressure.

## Deviations from the Python SCALAR

- Terms with numbers. Spiral matches the terms of `common_terms_with_numbers` in the order of a Python set, which follows `PYTHONHASHSEED`. A server draws a new seed at each start. For 23 pairs of terms that start at the same position, the split of a name therefore depends on the start of the server, such as `TLS_ECDH_anon_WITH_3DES_EDE_CBC_SHA` with `3DES` or `3D` and `ES`. The port always takes the longer term, the one order that keeps every term reachable. In the test corpus, 87 of 268,480 strings change with the seed.
- Text outside ASCII. The splitter treats each byte above 0x7F as a character that is no letter, digit, or space, and the encoder refuses such a byte. nameChecker, for example, sends only ASCII names.
- Features. The port computes every feature of `distilbert_preprocessing.py` except `lead_word`, which needs the WordNet corpus of NLTK. A model that selects `lead_word` fails to load with a message that names the feature. The release model selects `context` and `type`.
- Confidence and batch requests. The port gives the tags of the server path without the confidence values `p` and `alt` and without the JSON contract of `tag_batch`, such as the offsets of each word and the error codes per name.
- Long names. Ronin splits a run of one letter case by a recursion over its suffixes, whose depth and time grow with the length of the run. Python already stops some runs of 2,000 letters, such as `byby...by`, with a `RecursionError`, and a run of 2,000 letters takes about 3 s in the port. The port therefore answers a name of more than 2,000 characters with the error of a name too long for the model, also where Python tags the name, such as `abab...ab` of 2,002 letters. A token of more than 1,000 characters is split from its shortest suffix up. This order needs no deep recursion on any thread stack and gives the splits of the recursion.
- Rounding. The kernels follow the summation order of torch on the CPU in its main loops, but not in every edge case of its matrix library. The scores of the model differ from those of torch by at most 3.1e-5. Every test gives the tags of Python. The equality rests on these tests, since the smallest margin between two tags that the review found is 1.5 times this difference.
- CPU. The port runs on the CPU only. A single build runs on any CPU of its architecture, since the tagger picks its kernels, the functions that do the math of the network, at run time. On x86-64, the compiler builds two copies of these kernels. One copy uses the AVX2 and FMA instructions of newer CPUs, while the other copy runs on every x86-64 CPU. At its start, the tagger checks the CPU and takes the fast copy wherever the CPU has these instructions. On other CPUs, such as Arm64, GCC and Clang turn the kernels into vector code where the CPU has FMA instructions. An exact emulation of FMA serves the remaining CPUs. On 300 names of the corpus with 16 threads, the AVX2 kernels took 2.15 s, the vector kernels of a portable build for x86-64-v3 took 2.19 s, and the emulation took 16.5 s, each with the tags of the Python SCALAR.
- Server. The server read the name from its URL, therefore it cut a name such as `operator/=` at the slash. Its route `/probe/<cache_id>` came before the route of names, therefore the name `probe` without a cache ID got the probe message of the server in place of tags. The port receives the whole name. In place of the database of the server, nameChecker keeps the answers of a cache ID in a file of the user cache folder.

## Checks

On the 44,366 rows of `tools/identifiers_tag.tsv`, a corpus of names and contexts from public sources, the port gives the answer of the Python SCALAR for every row, the split, the tag, and the dictionary flag of each of 141,796 words. Five rows fail in the port and in Python alike as too long for the model. Since this corpus has no types, a second corpus checks the type input. On 27,823 names of Notepad++ with their declared types and the contexts of the SCALAR README, the port gives the answer of Python for every row, 72,041 words in all. The table gives the check of each part against Python on larger inputs.

| Part | Inputs compared | Differences |
|---|---|---|
| Splitter | 268,480 corpus strings and 120,000 generated strings, in release 1.0.0 with the same Spiral | 0 |
| Stemmer | 756,978 words, in release 1.0.0 | 0 |
| Encoder | 44,366 corpus rows in three feature sets, 4,261 rows of the SCALAR training data, and 705 edge cases | 0 |
| Model | 44,361 corpus rows and 27,823 Notepad++ names with types | 0 tags |

ctest runs these parts offline against the golden files of `tests/data/`, which the scripts of `tests/` wrote, and runs the whole tagger against 421 answers of the Python SCALAR.

## Upgrading SCALAR

Two files hold everything that names a SCALAR release. `cmake/scalarRelease.cmake` names the release and pins its model with the SHA-256 of each file. `src/scalarRelease.hpp` holds the constants of the Python code that the port reads, such as `FORBIDDEN_TRANSITIONS` and `TYPE_BOOL_TOKENS`. `tools/update_release.py` writes the two files from a checkout of the release, therefore an upgrade never edits either file directly. The header of each source file names the Python file it ports. A move to a newer release takes these steps.

1. Install the new release and check out the same tag, for its training data.

   ```bash
   pip install "scalar-tagger @ git+https://github.com/SCANL/scanl_tagger.git@<tag>"
   git clone --branch <tag> https://github.com/SCANL/scanl_tagger.git
   ```

2. Pin the release with the script. It reads the tag, the commit, the model of `RELEASE_MODEL` and `RELEASE_REVISION`, and the constants from the checkout. It then asks Hugging Face for the SHA-256 of each model file. It also checks that the English words that the release bundles equal the NLTK words of the port. Last, it lists the files that still name the previous release, such as `README.md`, for an update of their text. Configuring again downloads the new model.

   ```bash
   python3 tools/update_release.py scanl_tagger
   cmake -B build && cmake --build build
   ```

   The script stops with the name of a constant that the release no longer assigns. A new constant that the port needs goes into `HEADER_CONSTANTS` of the script.

3. Diff the inference code between the old and the new tag and port each change to the C++ file that How it works names. A new feature goes into `featureFunctions` of `src/featureTokens.cpp`. A new input of a name, such as the type in release 3.0.0, goes into `NameInContext` of `src/nameInContext.hpp` and into the corpus columns of `tools/scalarTagCorpus.cpp`, `tools/tag_with_python_scalar.py`, and `tools/compare_tag_answers.py`. The tokenizer stops with the name of the file and the setting when the model brings a tokenizer setting that the port does not follow.

   ```bash
   git diff <old tag> <new tag> -- scalar_tagger/tag_identifier.py scalar_tagger/tagging_backend.py \
       scalar_tagger/lm_based_tagger/distilbert_tagger.py scalar_tagger/lm_based_tagger/distilbert_crf.py \
       scalar_tagger/lm_based_tagger/distilbert_preprocessing.py
   ```

4. Write the golden files of ctest again and run the tests.

   ```bash
   tests/write_release_golden_files.sh <scanl_tagger checkout> build
   ctest --test-dir build
   ```

5. Compare the port with the Python SCALAR on the corpus of the repository and on a corpus with types, which needs the columns name, context, and type. Every row should give the same answer.

   ```bash
   tools/compare_with_python.sh build tools/identifiers_tag.tsv comparison
   ```

A new Spiral or new Ronin tables need two more steps. `tools/export_ronin_data.py` writes the tables of `data/ronin` from the new Spiral after the commit it checks is updated. The splitter golden files then come from the commands below. `SCALAR_SPLIT_CORPUS` names a file of one identifier per line. The shipped splitter files came from 268,480 identifier strings of the public sources of the corpus, which the repository does not hold, and any such list serves a new check.

```bash
SCALAR_SPLIT_CORPUS=<one identifier per line> python3 tests/make_golden.py
PYTHONHASHSEED=0 python3 tests/write_long_tokens.py --out tests/data/tagger/long_tokens.tsv
```
