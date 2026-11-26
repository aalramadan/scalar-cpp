# @file scalarRelease.cmake
# The SCALAR release that scalar-cpp ports. A move to a newer release starts here, as PORT.md describes.

# The Python SCALAR that the port follows and that tools/compare_with_python.sh installs
set(SCALAR_UPSTREAM_TAG    "v3.0.0-rc.3")
set(SCALAR_UPSTREAM_COMMIT "728f1e1df1ce9ef449f941c5b1c471bea9587af8")

# Its release model, RELEASE_MODEL and RELEASE_REVISION of scalar_tagger/cli.py, with the SHA-256 of
#   each file. A file that the model lacks, such as vocab.txt, is left out of the list.
set(SCALAR_MODEL_REPO     "sourceslicer/scalar_lm_v3")
set(SCALAR_MODEL_REVISION "5e5e8b1fe1d012a1df523718c013c66bdd12c2ad")
set(SCALAR_MODEL_FILES
    config.json           72f07333b4b2e50f25ce3f2410a0424605ab4803ed31395013e5313c477905ca
    tokenizer.json        435667fab0c06c165b1283ecb422497c37124f2d6a35b2ac73dc876332fc9518
    tokenizer_config.json e731cd915bceb73664cccc7245eca2d99bae2ba1ed539f07fce6fba9fe450968
    model.safetensors     714800a14254839eb32d17578c7a777bfa1d0370a05fd0e52da4c15b5c2dd791
)

# The NLTK words corpus, the source of scalar_tagger/data/english_words.txt.gz that the dictionary
#   flag reads, pinned to its nltk_data commit
set(NLTK_WORDS_URL    "https://raw.githubusercontent.com/nltk/nltk_data/b2085c79e91e22fc23c3f059cde261a73c5fc7a0/packages/corpora/words.zip")
set(NLTK_WORDS_SHA256 "54ed02917d6771dcc3e8141218960d020947f7f2ccfd9ac9b320979349746015")
