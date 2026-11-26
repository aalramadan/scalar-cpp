# 🏷️ scalar-cpp

scalar-cpp is a C++17 port of the [SCALAR](https://github.com/SCANL/scanl_tagger/tree/scalar_30_experimental) 3.0.0-rc.3 part-of-speech tagger for identifier names. It splits a name such as `getValue` into its words and tags each word, with the answers of the Python SCALAR and without Python or a tagger server. [nameChecker](https://github.com/KSU-SDML/nameChecker) uses it to check identifier names.

## 🧰 Requirements
- CMake 3.28 or newer and a C++17 compiler, such as GCC or Clang
- Network access for the first `cmake -B build`, which downloads the SCALAR model once

## 🔨 Build and test
```bash
git clone https://github.com/aalramadan/scalar-cpp.git
cd scalar-cpp
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build
```

The first `cmake -B build` downloads the model [sourceslicer/scalar_lm_v3](https://huggingface.co/sourceslicer/scalar_lm_v3) (about 266 MB) and the NLTK word list to `~/.cache/scalar-cpp/model-5e5e8b1`. On Windows, the folder is `%LOCALAPPDATA%\scalar-cpp\model-5e5e8b1`. With `XDG_CACHE_HOME` set, the folder lies under `$XDG_CACHE_HOME/scalar-cpp`. Later builds reuse the downloaded files.

## 📦 Use in a CMake project
FetchContent downloads scalar-cpp when your project configures. `scalar_embed_files` puts the model files into your binary, therefore the binary runs from any folder without the model folder.

```cmake
include(FetchContent)
FetchContent_Declare(scalar
    URL https://github.com/aalramadan/scalar-cpp/archive/refs/tags/v2.0.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(scalar)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE scalar::scalar)
scalar_embed_files(app)
```

```cpp
#include "builtInFiles.hpp"
#include "scalarTagger.hpp"

#include <iostream>

int main() {
    registerBuiltInFiles();
    const scalar::ScalarTagger tagger(scalar::builtInFiles({}));
    for (const scalar::TaggedWord& word : tagger.tag({"getValue", "FUNCTION", "int"}).words) {
        std::cout << word.word << ' ' << word.tag << '\n';
    }
}
```

`scalar_embed_files(app <name> <file> ...)` also puts your own files into the binary, such as word lists that `scalar::builtInFiles({"builtin:<name>"})` then reads. Without `scalar_embed_files`, `scalar::folderFiles(<model folder>, <scalar-cpp>/data/ronin, {<word list>...})` loads the files from disk.

## 📋 API
`src/scalarTagger.hpp` declares the tagger.

| Call | Result |
|---|---|
| `scalar::ScalarTagger tagger(files)` | Loads the model once. The methods are safe to call from several threads. |
| `tagger.tag({name, context, type, language, system})` | The words of the name, each with its tag, such as `N` or `V`, and its dictionary flag, `DW`, `DD`, or `UC` |
| `tagger.tagAll(names, threads)` | The answers of a list of names, tagged on several threads |
| `tagger.tagEach(names, threads, onAnswer)` | Each answer of a list, given to `onAnswer` as soon as it is ready |
| `scalar::wordsToJson(words)` | The JSON answer of the SCALAR server for these words |

The context is the category of the name, one of `FUNCTION`, `PARAMETER`, `ATTRIBUTE`, `DECLARATION`, and `CLASS`. The type is the declared type as the parser gives it, the return type of a function, and `class` for a class. The model reads the context and the type, while the language and the system may stay empty. The section "What to send in context, type, and system" of the SCALAR README gives the context and the type of each kind of name, such as `CLASS` for a namespace, and the names to leave out, such as constructors and macros. A word list gives `DW` to each of its words, as the `en.txt` files of the SCALAR server do.

## 🛠️ Command line
`build/scalarTagCorpus` tags each row of a TSV file whose header names the columns `name` and `context`. The columns `type`, `language`, and `system` may follow. It writes one JSON line per name.

```bash
build/scalarTagCorpus names.tsv answers.jsonl [threads [word list]...]
```

## 📄 License
scalar-cpp is licensed under GPL-3.0, as SCALAR and Spiral are. `THIRD_PARTY_NOTICES.txt` lists the projects it ports. `PORT.md` describes the port, its checks against the Python SCALAR, and the upgrade to a newer SCALAR.
