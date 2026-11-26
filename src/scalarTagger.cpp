// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file scalarTagger.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// Tags names as listen() and dictionary_lookup() of scalar_tagger/tag_identifier.py of SCALAR do:
// spiral.ronin.split, DistilBertTagger.tag_identifier with the type, language, and system name of the
// query, and the dictionary flag of each word.

#include "scalarTagger.hpp"
#include "asciiText.hpp"
#include "dataFile.hpp"
#include "featureTokens.hpp"
#include "modelInput.hpp"

#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>

namespace scalar {

namespace {

// The model tags about four names per call fastest on each thread
constexpr std::size_t namesPerBatch = 4;

// Ronin takes up to about 3 s for a run of 2,000 letters of one case, and Python stops some such runs with
//  a RecursionError, therefore a longer name gets the error of a name too long for the model.
constexpr std::size_t longestName = 2000;

std::string readText(const std::string& path) {
    const std::optional<DataFile> file = openDataFile(path);
    if (!file) throw std::runtime_error("cannot read " + path);
    return std::string(file->bytes());
}

// The line ends of str.splitlines() in ASCII text
bool endsLine(char c) {
    return c == '\n' || c == '\r' || c == '\v' || c == '\f' || (c >= '\x1c' && c <= '\x1e');
}

// line_tokenize of NLTK with blank lines discarded: str.splitlines(), without the lines that
//  rstrip() leaves empty
std::vector<std::string> nltkLines(const std::string& text) {
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t end = start;
        while (end < text.size() && !endsLine(text[end])) ++end;
        const std::string line = text.substr(start, end - start);
        if (!pythonStrip(line).empty()) lines.push_back(line);
        start = end + ((end + 1 < text.size() && text[end] == '\r' && text[end + 1] == '\n') ? 2 : 1);
    }
    return lines;
}

// The lines of a file that Python opens as text, which ends a line at \n, \r\n, or \r
std::vector<std::string> universalLines(const std::string& text) {
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t end = start;
        while (end < text.size() && text[end] != '\n' && text[end] != '\r') ++end;
        lines.push_back(text.substr(start, end - start));
        start = end + ((end + 1 < text.size() && text[end] == '\r' && text[end + 1] == '\n') ? 2 : 1);
    }
    return lines;
}

std::string concatenatedText(const std::vector<std::string>& paths) {
    std::string text;
    for (const std::string& path : paths) text += readText(path);
    return text;
}

} // namespace

ScalarTagger::ScalarTagger(const TaggerFiles& files)
    : splitter(files.roninDirectory),
      tokenizer(files.modelDirectory),
      features(modelFeatures(readSavedFeatures(files.modelDirectory + "/config.json"))),
      model(files.modelDirectory) {
    // app.english_words: the NLTK words in lower case, then each line of en.txt with strip() and lower()
    for (const std::string& word : nltkLines(concatenatedText(files.nltkWordFiles))) {
        dictionaryWords.insert(toLowerAscii(word));
    }
    for (const std::string& line : universalLines(concatenatedText(files.customWordFiles))) {
        dictionaryWords.insert(toLowerAscii(pythonStrip(line)));
    }
}

std::string ScalarTagger::dictionaryFlag(const std::string& word) const {
    if (dictionaryWords.count(toLowerAscii(word)) > 0) return "DW";
    if (!word.empty() && std::all_of(word.begin(), word.end(), isAsciiDigit)) return "DD";
    return "UC";
}

TagResult ScalarTagger::tag(const NameInContext& name) const {
    return tagAll({name}, 1).front();
}

std::vector<TagResult> ScalarTagger::tagAll(const std::vector<NameInContext>& names, unsigned threads) const {
    std::vector<TagResult> results(names.size());
    tagEach(names, threads, [&results](std::size_t index, TagResult answer) { results[index] = std::move(answer); });
    return results;
}

void ScalarTagger::tagEach(const std::vector<NameInContext>& names, unsigned threads,
                           const std::function<void(std::size_t, TagResult)>& onAnswer) const {
    std::vector<TagResult> results(names.size());
    const std::size_t batches = (names.size() + namesPerBatch - 1) / namesPerBatch;
    std::atomic<std::size_t> nextBatch{0};
    const auto work = [&]() {
        for (std::size_t batch = nextBatch++; batch < batches; batch = nextBatch++) {
            const std::size_t begin = batch * namesPerBatch;
            const std::size_t end = std::min(names.size(), begin + namesPerBatch);
            tagBatch(names, begin, end, results);
            for (std::size_t i = begin; i < end; ++i) onAnswer(i, std::move(results[i]));
        }
    };

    const std::size_t workers = std::min<std::size_t>(std::max(1u, threads), batches);
    std::vector<std::thread> pool;
    for (std::size_t i = 1; i < workers; ++i) {
        // Fewer threads tag the same names when the system refuses another one
        try {
            pool.emplace_back(work);
        }
        catch (const std::system_error&) {
            break;
        }
    }
    work();
    for (std::thread& thread : pool) thread.join();
}

void ScalarTagger::tagBatch(const std::vector<NameInContext>& names, std::size_t begin, std::size_t end,
                            std::vector<TagResult>& results) const {
    try {
        std::vector<ModelInput> inputs;
        std::vector<std::size_t> owners;
        std::vector<std::vector<std::string>> words(end - begin);
        for (std::size_t i = begin; i < end; ++i) {
            try {
                if (names[i].name.size() > longestName) throw IdentifierTooLongError();
                words[i - begin] = splitter.split(names[i].name);
                if (words[i - begin].empty()) {
                    results[i].tagged = true;
                    continue;
                }
                IdentifierRow row;
                row.context    = names[i].context;
                row.type       = names[i].type;
                row.language   = names[i].language;
                row.systemName = names[i].system;
                inputs.push_back(encodeIdentifier(tokenizer, row, words[i - begin], features));
                owners.push_back(i);
            }
            catch (const std::exception& e) {
                results[i].error = e.what();
            }
        }
        if (inputs.empty()) return;

        const std::vector<std::vector<int>> labels = model.decodeBatch(inputs);
        for (std::size_t k = 0; k < owners.size(); ++k) {
            TagResult& result = results[owners[k]];
            const std::vector<std::string>& nameWords = words[owners[k] - begin];
            for (std::size_t w = 0; w < nameWords.size(); ++w) {
                result.words.push_back({nameWords[w], model.labelNames()[static_cast<std::size_t>(labels[k][w])],
                                        dictionaryFlag(nameWords[w])});
            }
            result.tagged = true;
        }
    }
    catch (const std::exception& e) {
        for (std::size_t i = begin; i < end; ++i) {
            if (!results[i].tagged && results[i].error.empty()) results[i].error = e.what();
        }
    }
}

TaggerFiles folderFiles(const std::string& modelDirectory, const std::string& roninDirectory,
                        const std::vector<std::string>& customWordFiles) {
    TaggerFiles files;
    files.modelDirectory  = modelDirectory;
    files.roninDirectory  = roninDirectory;
    files.nltkWordFiles   = {modelDirectory + "/nltk/words/en", modelDirectory + "/nltk/words/en-basic"};
    files.customWordFiles = customWordFiles;
    return files;
}

TaggerFiles builtInFiles(const std::vector<std::string>& customWordFiles) {
    return folderFiles(std::string(builtInPrefix) + "model", std::string(builtInPrefix) + "ronin", customWordFiles);
}

std::string wordsToJson(const std::vector<TaggedWord>& words) {
    nlohmann::json list = nlohmann::json::array();
    for (const TaggedWord& word : words) {
        nlohmann::json entry = nlohmann::json::object();
        entry[word.word] = {{"dictionary", word.dictionary}, {"tag", word.tag}};
        list.push_back(entry);
    }
    nlohmann::json body = nlohmann::json::object();
    body["words"] = list;
    return body.dump();
}

} // namespace scalar
