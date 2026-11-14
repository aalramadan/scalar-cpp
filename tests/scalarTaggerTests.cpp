// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file scalarTaggerTests.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// Tests of the whole SCALAR tagger against the answers of the Python SCALAR
// in data/tagger. Usage: scalarTaggerTests <test> <folder of data/tagger>

#include "dataFile.hpp"
#include "ronin.hpp"
#include "scalarTagger.hpp"

#include "json.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <pthread.h>
#endif

namespace {

struct GoldenAnswer {
    scalar::NameInContext           name;
    bool                            tagged = false;
    std::vector<scalar::TaggedWord> words;
};

const scalar::ScalarTagger& sharedTagger() {
    static const scalar::ScalarTagger tagger(scalar::folderFiles(SCALAR_DATA_DIR, SCALAR_RONIN_DIR, {SCALAR_TEST_WORDS}));
    return tagger;
}

std::vector<GoldenAnswer> readGoldenAnswers(const std::string& folder) {
    std::ifstream file(folder + "/golden_answers.jsonl");
    if (!file) throw std::runtime_error("cannot read " + folder + "/golden_answers.jsonl");
    std::vector<GoldenAnswer> answers;
    std::string line;
    while (std::getline(file, line)) {
        const nlohmann::json record = nlohmann::json::parse(line);
        GoldenAnswer answer;
        answer.name = {record.at("name").get<std::string>(), record.at("context").get<std::string>()};
        answer.tagged = record.contains("words");
        if (answer.tagged) {
            for (const nlohmann::json& entry : record.at("words")) {
                for (const auto& [word, value] : entry.items()) {
                    answer.words.push_back({word, value.at("tag").get<std::string>(), value.at("dictionary").get<std::string>()});
                }
            }
        }
        answers.push_back(answer);
    }
    return answers;
}

bool sameWords(const std::vector<scalar::TaggedWord>& left, const std::vector<scalar::TaggedWord>& right) {
    return std::equal(left.begin(), left.end(), right.begin(), right.end(),
                      [](const scalar::TaggedWord& a, const scalar::TaggedWord& b) {
                          return a.word == b.word && a.tag == b.tag && a.dictionary == b.dictionary;
                      });
}

// The answers of Python took their dictionary flags from the word lists of nameChecker, therefore the
//  answers are compared on the split and the tags alone
bool sameSplitAndTags(const std::vector<scalar::TaggedWord>& left, const std::vector<scalar::TaggedWord>& right) {
    return std::equal(left.begin(), left.end(), right.begin(), right.end(),
                      [](const scalar::TaggedWord& a, const scalar::TaggedWord& b) { return a.word == b.word && a.tag == b.tag; });
}

bool sameResult(const scalar::TagResult& left, const scalar::TagResult& right) {
    return left.tagged == right.tagged && left.error == right.error && sameWords(left.words, right.words);
}

std::string describe(const scalar::TagResult& result) {
    if (!result.tagged) return "error '" + result.error + "'";
    std::string text = "[";
    for (const scalar::TaggedWord& word : result.words) text += " " + word.word + "/" + word.tag + "/" + word.dictionary;
    return text + " ]";
}

std::vector<scalar::NameInContext> namesOf(const std::vector<GoldenAnswer>& answers) {
    std::vector<scalar::NameInContext> names;
    for (const GoldenAnswer& answer : answers) names.push_back(answer.name);
    return names;
}

bool taggerGivesThePythonAnswersOnGoldenNames(const std::string& folder) {
    const std::vector<GoldenAnswer> golden = readGoldenAnswers(folder);
    const std::vector<scalar::TagResult> results = sharedTagger().tagAll(namesOf(golden), std::thread::hardware_concurrency());
    std::size_t differing = 0;
    for (std::size_t i = 0; i < golden.size(); ++i) {
        const bool equal = results[i].tagged == golden[i].tagged && (!golden[i].tagged || sameSplitAndTags(results[i].words, golden[i].words));
        if (!equal && ++differing <= 10) {
            scalar::TagResult expected;
            expected.tagged = golden[i].tagged;
            expected.words = golden[i].words;
            expected.error = golden[i].tagged ? "" : "any";
            std::cout << "  " << golden[i].name.name << " (" << golden[i].name.context << "): Python " << describe(expected)
                      << ", C++ " << describe(results[i]) << "\n";
        }
    }
    std::cout << "  " << golden.size() - differing << " of " << golden.size() << " answers equal those of Python\n";
    return differing == 0;
}

bool tagAllGivesTheAnswersOfTagOnEveryThreadCount(const std::string& folder) {
    std::vector<scalar::NameInContext> names = namesOf(readGoldenAnswers(folder));
    names.resize(std::min<std::size_t>(names.size(), 60));
    const std::vector<scalar::TagResult> oneThread = sharedTagger().tagAll(names, 1);
    const std::vector<scalar::TagResult> eightThreads = sharedTagger().tagAll(names, 8);
    bool passed = true;
    for (std::size_t i = 0; i < names.size(); ++i) {
        const scalar::TagResult single = sharedTagger().tag(names[i]);
        if (!sameResult(single, oneThread[i]) || !sameResult(single, eightThreads[i])) {
            std::cout << "  " << names[i].name << ": tag " << describe(single) << ", one thread " << describe(oneThread[i])
                      << ", eight threads " << describe(eightThreads[i]) << "\n";
            passed = false;
        }
    }
    return passed && sharedTagger().tagAll({}, 4).empty();
}

bool dictionaryFlagFollowsTheWordListsOfTheServer(const std::string&) {
    // value is an NLTK word, Addr, API, and zymograms come only from tests/data/tagger/custom_words.txt,
    //  and the lookup lowers the word first
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"value", "DW"}, {"VALUE", "DW"}, {"Value", "DW"}, {"addr", "DW"}, {"API", "DW"}, {"zymograms", "DW"},
        {"42", "DD"}, {"0042", "DD"}, {"4a", "UC"}, {"xyzzq", "UC"}, {"Xyzzq", "UC"}};
    bool passed = true;
    for (const auto& [word, flag] : cases) {
        const std::string found = sharedTagger().dictionaryFlag(word);
        if (found != flag) {
            std::cout << "  " << word << ": expected " << flag << ", got " << found << "\n";
            passed = false;
        }
    }
    return passed;
}

bool namesWithoutWordsAreTaggedWithNoWords(const std::string&) {
    bool passed = true;
    for (const std::string name : {"__", "~", "$", "_"}) {
        const scalar::TagResult result = sharedTagger().tag({name, "DECLARATION"});
        if (!result.tagged || !result.words.empty()) {
            std::cout << "  " << name << ": " << describe(result) << "\n";
            passed = false;
        }
    }
    return passed;
}

bool namesTooLongForTheModelGetItsError(const std::string&) {
    // Ronin splits each WordN into Word and N, and 400 words need more than 512 tokens
    std::string manyWords;
    for (int i = 0; i < 200; ++i) manyWords += "Word" + std::to_string(i);
    const std::vector<scalar::NameInContext> names = {{manyWords, "FUNCTION"}, {std::string(2001, 'a'), "FUNCTION"}};
    bool passed = true;
    for (const scalar::TagResult& result : sharedTagger().tagAll(names, 2)) {
        if (result.tagged || result.error != "Identifier is too long for the model input.") {
            std::cout << "  " << describe(result) << "\n";
            passed = false;
        }
    }
    return passed;
}

// Splits each long token of data/tagger/long_tokens.tsv on a thread with a stack of 256 KiB, half
//  the stack of a secondary thread of macOS, and compares the tokens with those of Python
bool longRunsOfOneCaseSplitOnASmallStack(const std::string& folder) {
    struct Work {
        const scalar::Ronin*                  ronin;
        std::vector<std::string>              tokens;
        std::vector<std::vector<std::string>> expected;
        bool                                  passed = true;
    };
    const scalar::Ronin ronin(SCALAR_RONIN_DIR);
    Work work{&ronin, {}, {}};
    std::ifstream file(folder + "/long_tokens.tsv");
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream fields(line);
        std::string field;
        std::getline(fields, field, '\t');
        work.tokens.push_back(field);
        work.expected.emplace_back();
        while (std::getline(fields, field, '\t')) work.expected.back().push_back(field);
    }
    if (work.tokens.empty()) throw std::runtime_error("no long tokens in " + folder);

    const auto run = [](void* argument) -> void* {
        Work& w = *static_cast<Work*>(argument);
        for (std::size_t i = 0; i < w.tokens.size(); ++i) {
            if (w.ronin->split(w.tokens[i]) != w.expected[i]) {
                std::cout << "  the split of a token of " << w.tokens[i].size() << " letters differs from Python\n";
                w.passed = false;
            }
        }
        return nullptr;
    };
#if defined(__unix__) || defined(__APPLE__)
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setstacksize(&attributes, 256 * 1024);
    pthread_t thread;
    if (pthread_create(&thread, &attributes, run, &work) != 0) throw std::runtime_error("cannot start a thread");
    pthread_join(thread, nullptr);
    pthread_attr_destroy(&attributes);
#else
    run(&work);
#endif
    return work.passed;
}

bool wordsToJsonGivesTheBodyOfTheServer(const std::string&) {
    const std::string body = scalar::wordsToJson({{"get", "V", "DW"}, {"value", "N", "DW"}});
    const std::string expected = R"({"words":[{"get":{"dictionary":"DW","tag":"V"}},{"value":{"dictionary":"DW","tag":"N"}}]})";
    if (body != expected || scalar::wordsToJson({}) != R"({"words":[]})") {
        std::cout << "  got " << body << "\n";
        return false;
    }
    return true;
}

bool builtInFilesNameOnlyFilesInTheBinary(const std::string&) {
    const scalar::TaggerFiles files = scalar::builtInFiles({});
    std::vector<std::string> paths = files.nltkWordFiles;
    paths.push_back(files.modelDirectory);
    paths.push_back(files.roninDirectory);
    bool passed = true;
    for (const std::string& path : paths) {
        if (!scalar::isBuiltInPath(path)) {
            std::cout << "  " << path << " names a file on disk\n";
            passed = false;
        }
    }
    return passed;
}

} // namespace

int main(int argc, char* argv[]) {
    const std::map<std::string, std::function<bool(const std::string&)>> tests = {
        {"taggerGivesThePythonAnswersOnGoldenNames", taggerGivesThePythonAnswersOnGoldenNames},
        {"tagAllGivesTheAnswersOfTagOnEveryThreadCount", tagAllGivesTheAnswersOfTagOnEveryThreadCount},
        {"dictionaryFlagFollowsTheWordListsOfTheServer", dictionaryFlagFollowsTheWordListsOfTheServer},
        {"namesWithoutWordsAreTaggedWithNoWords", namesWithoutWordsAreTaggedWithNoWords},
        {"namesTooLongForTheModelGetItsError", namesTooLongForTheModelGetItsError},
        {"longRunsOfOneCaseSplitOnASmallStack", longRunsOfOneCaseSplitOnASmallStack},
        {"wordsToJsonGivesTheBodyOfTheServer", wordsToJsonGivesTheBodyOfTheServer},
        {"builtInFilesNameOnlyFilesInTheBinary", builtInFilesNameOnlyFilesInTheBinary},
    };
    if (argc != 3 || tests.count(argv[1]) == 0) {
        std::cerr << "Usage: scalarTaggerTests <test> <folder of data/tagger>" << std::endl;
        return 2;
    }
    try {
        const bool passed = tests.at(argv[1])(argv[2]);
        std::cout << (passed ? "PASS " : "FAIL ") << argv[1] << "\n";
        return passed ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cout << "FAIL " << argv[1] << ": " << error.what() << "\n";
        return 1;
    }
}
