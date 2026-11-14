// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file scalarTagger.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// The SCALAR tagger as the SCALAR server answers its clients: Ronin splits a name into words, the
// encoder builds the model input for the context of the name, DistilBERT and the CRF give each word
// its part-of-speech tag, and the NLTK words with the custom word lists give its dictionary flag.

#ifndef SCALAR_TAGGER_HPP
#define SCALAR_TAGGER_HPP

#include "distilbertCrf.hpp"
#include "ronin.hpp"
#include "wordpieceTokenizer.hpp"

#include <cstddef>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

namespace scalar {

// One word of an answer with its tag, such as N or V, and its dictionary flag: DW for a word of
//   the word lists, DD for digits, and UC for any other word
struct TaggedWord {
    std::string word;
    std::string tag;
    std::string dictionary;
};

// A name with the context of its category, such as FUNCTION or PARAMETER
struct NameInContext {
    std::string name;
    std::string context;
};

// The answer for one name. A name that Ronin splits into no words is tagged with no words, as the
//   server answers. error holds the reason where the server answers with an error.
struct TagResult {
    bool                    tagged = false;
    std::vector<TaggedWord> words;
    std::string             error;
};

// The files of the tagger
struct TaggerFiles {
    std::string              modelDirectory;   // config.json, vocab.txt, the tokenizer files, model.safetensors
    std::string              roninDirectory;   // the tables that tools/export_ronin_data.py writes
    std::vector<std::string> nltkWordFiles;    // read as one text, as nltk.corpus.words.words() reads its files
    std::vector<std::string> customWordFiles;  // read as one text with strip() and lower() per line, as en.txt
};

// Loaded once and immutable afterwards, therefore the methods are safe to call from several threads
class ScalarTagger {
public:
    // Throws std::runtime_error when a file is missing or malformed
    explicit ScalarTagger(const TaggerFiles& files);

    TagResult tag(const NameInContext& name) const;

    // The answers in the order of the names, tagged on up to the given number of threads
    std::vector<TagResult> tagAll(const std::vector<NameInContext>& names, unsigned threads) const;

    // Tags the names on up to the given number of threads, the names of lower index first, and gives
    //   each answer with the index of its name to onAnswer as soon as its batch is done. onAnswer runs
    //   on the threads of the tagger, at the same time for different names.
    void tagEach(const std::vector<NameInContext>& names, unsigned threads,
                 const std::function<void(std::size_t index, TagResult answer)>& onAnswer) const;

    // DW, DD, or UC, as dictionary_lookup of the server sets the flag
    std::string dictionaryFlag(const std::string& word) const;

private:
    // Tags the names from begin to end in one call of the model
    void tagBatch(const std::vector<NameInContext>& names, std::size_t begin, std::size_t end,
                  std::vector<TagResult>& results) const;

    Ronin                           splitter;
    WordPieceTokenizer              tokenizer;
    std::vector<std::string>        features;
    DistilBertCrf                   model;
    std::unordered_set<std::string> dictionaryWords;
};

// The files in the model folder that CMake downloads, which also holds the NLTK words, in the folder of
//   the Ronin tables, and in the given custom word lists, such as the en.txt files
TaggerFiles folderFiles(const std::string& modelDirectory, const std::string& roninDirectory,
                        const std::vector<std::string>& customWordFiles);

// The model files, the NLTK words, and the Ronin tables that the build put into the binary, with the
//   given custom word lists
TaggerFiles builtInFiles(const std::vector<std::string>& customWordFiles);

// The JSON body of the server answer, such as {"words":[{"get":{"dictionary":"DW","tag":"V"}}]}
std::string wordsToJson(const std::vector<TaggedWord>& words);

} // namespace scalar

#endif
