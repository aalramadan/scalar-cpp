// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file encoderTests.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Tests of the SCALAR model-input encoder against the golden files that tests/write_golden.py writes from the upstream Python

#include "caseFormat.hpp"

#include "featureTokens.hpp"
#include "jsonReader.hpp"
#include "modelInput.hpp"
#include "pythonSum.hpp"
#include "sequenceMatcher.hpp"
#include "wordpieceTokenizer.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using scalarTests::Outcome;
using scalarTests::ReferenceCase;

struct TestContext {
    std::filesystem::path             dataFolder;
    std::filesystem::path             modelFolder;
    const scalar::WordPieceTokenizer* tokenizer = nullptr;
};

// The failures of the tests; the first ones of each test print in full
struct FailureLog {
    std::size_t count       = 0;
    std::size_t countInTest = 0;

    void startTest() { countInTest = 0; }

    void fail(const std::string& message) {
        ++count;
        if (++countInTest <= 20) std::cout << "  FAIL " << message << "\n";
    }

    void expect(bool condition, const std::string& message) {
        if (!condition) fail(message);
    }
};

FailureLog failures;

std::vector<std::string> readLines(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    std::vector<std::string> lines;
    for (std::string line; std::getline(input, line);) lines.push_back(line);
    return lines;
}

std::string readText(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

void writeText(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << text;
    if (!output) throw std::runtime_error("cannot write " + path.string());
}

std::vector<std::string> unescapedWords(const std::vector<std::string_view>& fields, std::size_t first) {
    std::vector<std::string> words;
    for (std::size_t i = first; i < fields.size(); ++i) words.push_back(scalarTests::unescapeField(fields[i]));
    return words;
}

// A folder of the system temporary folder that the test removes at its end
class TemporaryFolder {
public:
    explicit TemporaryFolder(const std::string& stem) {
        std::random_device random;
        path = std::filesystem::temp_directory_path() / (stem + "-" + std::to_string(random()) + std::to_string(random()));
        std::filesystem::create_directories(path);
    }
    ~TemporaryFolder() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    TemporaryFolder(const TemporaryFolder&)            = delete;
    TemporaryFolder& operator=(const TemporaryFolder&) = delete;

    std::filesystem::path path;
};

void tokenizerLoadsTheVocabularyOfTheModelFolder(const TestContext& context) {
    const scalar::WordPieceTokenizer& tokenizer = *context.tokenizer;
    failures.expect(tokenizer.vocabularySize() == 30522, "the vocabulary has 30522 entries");
    failures.expect(tokenizer.tokenId("[UNK]") == 100 && tokenizer.tokenId("[CLS]") == 101 && tokenizer.tokenId("[SEP]") == 102,
                    "the ids of [UNK], [CLS], and [SEP]");
    const scalar::Encoding encoding = tokenizer.encode({"[CLS]", "getX"});
    failures.expect(encoding.ids == std::vector<std::int32_t>{101, 101, *tokenizer.tokenId("get"), *tokenizer.tokenId("##x"), 102},
                    "the ids of [CLS] getX");
    failures.expect(encoding.wordIds == std::vector<std::optional<std::size_t>>{std::nullopt, 0, 1, 1, std::nullopt},
                    "the word index of each id of [CLS] getX");
}

// The lines of a vocab.txt that agrees with model.vocab of tokenizer.json
std::string vocabularyFileOf(const std::filesystem::path& tokenizerJson) {
    const scalar::JsonValue  json  = scalar::readJsonFile(tokenizerJson.string());
    const scalar::JsonValue& vocab = *json.find("model")->find("vocab");
    std::vector<std::string> lines(vocab.items.size());
    for (std::size_t i = 0; i < vocab.keys.size(); ++i) lines[static_cast<std::size_t>(vocab.items[i].number)] = vocab.keys[i];
    std::string text;
    for (const std::string& line : lines) text += line + "\n";
    return text;
}

void tokenizerRejectsSettingsThatThePortDoesNotFollow(const TestContext& context) {
    const TemporaryFolder folder("scalar-encoder-settings");
    for (const char* file : {"tokenizer.json", "tokenizer_config.json"}) {
        std::filesystem::copy_file(context.modelFolder / file, folder.path / file);
    }

    const auto loads = [&folder] {
        try {
            const scalar::WordPieceTokenizer tokenizer(folder.path.string());
            return true;
        }
        catch (const std::runtime_error&) {
            return false;
        }
    };
    failures.expect(loads(), "an unchanged copy of the model folder loads");

    // The files that older models add, which must agree with tokenizer.json and tokenizer_config.json
    writeText(folder.path / "vocab.txt", vocabularyFileOf(folder.path / "tokenizer.json"));
    writeText(folder.path / "special_tokens_map.json",
              "{\"cls_token\": \"[CLS]\", \"mask_token\": \"[MASK]\", \"pad_token\": \"[PAD]\", "
              "\"sep_token\": \"[SEP]\", \"unk_token\": \"[UNK]\"}");
    failures.expect(loads(), "a vocab.txt and a special_tokens_map.json that agree with tokenizer.json load");

    struct Change {
        const char* file = nullptr;
        std::string from;
        std::string to;
    };
    const Change changes[] = {
        {"tokenizer.json", "\"lowercase\": true", "\"lowercase\": false"},
        {"tokenizer.json", "\"strip_accents\": null", "\"strip_accents\": true"},
        {"tokenizer.json", "\"clean_text\": true", "\"clean_text\": false"},
        {"tokenizer.json", "\"handle_chinese_chars\": true", "\"handle_chinese_chars\": false"},
        {"tokenizer.json", "\"type\": \"BertPreTokenizer\"", "\"type\": \"Whitespace\""},
        {"tokenizer.json", "\"continuing_subword_prefix\": \"##\"", "\"continuing_subword_prefix\": \"@@\""},
        {"tokenizer.json", "\"max_input_chars_per_word\": 100", "\"max_input_chars_per_word\": 200"},
        {"tokenizer.json", "\"id\": 100,", "\"id\": 99,"},
        {"tokenizer.json", "\"added_tokens\": [",
         "\"added_tokens\": [{\"id\": 1, \"content\": \"[unused0]\", \"single_word\": false, \"lstrip\": false, "
         "\"rstrip\": false, \"normalized\": false, \"special\": true},"},
        {"tokenizer.json", "\"direction\": \"Right\"", "\"direction\": \"Left\""},
        {"vocab.txt", "\n[unused0]\n", "\n[unused00]\n"},
        {"tokenizer_config.json", "\"do_lower_case\": true", "\"do_lower_case\": false"},
        {"tokenizer_config.json", "\"strip_accents\": null", "\"strip_accents\": true"},
        {"tokenizer_config.json", "\"tokenize_chinese_chars\": true", "\"tokenize_chinese_chars\": false"},
        {"tokenizer_config.json", "\"model_max_length\": 512", "\"model_max_length\": 256"},
        {"tokenizer_config.json", "\"do_lower_case\": true",
         "\"added_tokens_decoder\": {\"1\": {\"content\": \"[unused0]\", \"lstrip\": false, \"normalized\": false, "
         "\"rstrip\": false, \"single_word\": false, \"special\": true}}, \"do_lower_case\": true"},
        {"tokenizer_config.json", "\"do_lower_case\": true", "\"extra_special_tokens\": {\"x\": \"[unused0]\"}, \"do_lower_case\": true"},
        {"tokenizer_config.json", "\"unk_token\": \"[UNK]\"", "\"unk_token\": \"[unused0]\""},
        {"special_tokens_map.json", "\"unk_token\": \"[UNK]\"", "\"unk_token\": \"[MASK]\""},
    };
    for (const Change& change : changes) {
        const std::filesystem::path path     = folder.path / change.file;
        const std::string           original = readText(path);
        const std::size_t           found    = original.find(change.from);
        if (found == std::string::npos || original.find(change.from, found + 1) != std::string::npos) {
            failures.fail(std::string(change.file) + " holds " + change.from + " other than once");
            continue;
        }
        writeText(path, std::string(original).replace(found, change.from.size(), change.to));
        failures.expect(!loads(), std::string(change.file) + " with " + change.to + " is rejected");
        writeText(path, original);
    }

    writeText(folder.path / "added_tokens.json", "{\"[unused0]\": 1}");
    failures.expect(!loads(), "an added_tokens.json is rejected");
    std::filesystem::remove(folder.path / "added_tokens.json");
    std::filesystem::remove(folder.path / "tokenizer.json");
    failures.expect(!loads(), "a folder without tokenizer.json is rejected");
}

void featureListsNormalizeAsInPython(const TestContext& context) {
    for (const std::string& line : readLines(context.dataFolder / "feature_lists.tsv")) {
        const std::vector<std::string_view> fields = scalarTests::splitFields(line);
        if (fields[0] == "model" || fields[0] == "legacy") continue;
        std::optional<std::vector<std::string>> given;
        std::size_t                             next = 1;
        if (fields[0] == "list") {
            std::size_t count = 0;
            std::from_chars(fields[1].data(), fields[1].data() + fields[1].size(), count);
            given = unescapedWords(std::vector<std::string_view>(fields.begin() + 2, fields.begin() + 2 + static_cast<std::ptrdiff_t>(count)), 0);
            next  = 2 + count;
        }
        const std::string expectedKind = std::string(fields[next]);
        try {
            const std::vector<std::string> normalized = scalar::normalizeSelectedFeatures(given);
            failures.expect(expectedKind == "ok" && normalized == unescapedWords(fields, next + 1), "the normalized list of " + line);
        }
        catch (const std::invalid_argument& error) {
            failures.expect(expectedKind == "error" && error.what() == scalarTests::unescapeField(fields[next + 1]),
                            "the error of " + line + ": " + error.what());
        }
    }
}

void modelFeaturesComeFromTheConfigOrTheLegacyList(const TestContext& context) {
    const std::optional<std::vector<std::string>> saved = scalar::readSavedFeatures((context.modelFolder / "config.json").string());
    for (const std::string& line : readLines(context.dataFolder / "feature_lists.tsv")) {
        const std::vector<std::string_view> fields = scalarTests::splitFields(line);
        if (fields[0] == "model") {
            failures.expect(scalar::modelFeatures(saved) == unescapedWords(fields, 1), "the features of DistilBertTagger.__init__");
        }
        if (fields[0] == "legacy") {
            failures.expect(scalar::modelFeatures(std::nullopt) == unescapedWords(fields, 1), "the features of a config without a list");
        }
    }

    const TemporaryFolder folder("scalar-encoder-config");
    const std::filesystem::path config = folder.path / "config.json";
    writeText(config, "{\"dim\": 768, \"selected_features\": [\"cvr\", \" context\"]}");
    const auto listed = scalar::readSavedFeatures(config.string());
    failures.expect(listed == std::vector<std::string>{"cvr", " context"}, "a saved list reads as written");
    failures.expect(scalar::modelFeatures(listed) == std::vector<std::string>{"context", "cvr"}, "a saved list is normalized");
    writeText(config, "{\"selected_features\": []}");
    failures.expect(scalar::modelFeatures(scalar::readSavedFeatures(config.string())).empty(), "an empty saved list selects no feature");
    writeText(config, "{\"selected_features\": [\"context\", \"lead_word\"]}");
    bool refused = false;
    try {
        scalar::modelFeatures(scalar::readSavedFeatures(config.string()));
    }
    catch (const std::runtime_error&) {
        refused = true;
    }
    failures.expect(refused, "a model that selects lead_word, which the port does not compute, is refused");
    writeText(config, "{\"selected_features\": null}");
    failures.expect(!scalar::readSavedFeatures(config.string()), "a null list counts as no list");
    writeText(config, "{\"selected_features\": \"context\"}");
    try {
        scalar::readSavedFeatures(config.string());
        failures.fail("a string in place of the list is rejected");
    }
    catch (const std::runtime_error&) {
    }
}

// The golden value of a double, nan and inf included. std::strtod reads it on every C++ library,
//  while std::from_chars of double is missing from some, and the tests keep the C locale.
double parseDouble(std::string_view text) {
    return std::strtod(std::string(text).c_str(), nullptr);
}

void sequenceMatcherRatioEqualsDifflib(const TestContext& context) {
    std::size_t compared = 0;
    for (const std::string& line : readLines(context.dataFolder / "ratio_cases.tsv")) {
        const std::vector<std::string_view> fields = scalarTests::splitFields(line);
        const double expected = parseDouble(fields[2]);
        const double actual = scalar::sequenceMatcherRatio(scalarTests::unescapeField(fields[0]), scalarTests::unescapeField(fields[1]));
        failures.expect(actual == expected, "ratio of " + line + " is " + std::to_string(actual));
        ++compared;
    }
    std::cout << "  " << compared << " pairs compared\n";
}

void pythonSumEqualsTheBuiltInSum(const TestContext& context) {
    std::size_t compared = 0;
    for (const std::string& line : readLines(context.dataFolder / "float_sums.tsv")) {
        const std::vector<std::string_view> fields = scalarTests::splitFields(line);
        std::vector<double>                 values;
        for (std::size_t start = 0; start < fields[0].size();) {
            std::size_t end = fields[0].find(' ', start);
            if (end == std::string_view::npos) end = fields[0].size();
            values.push_back(parseDouble(fields[0].substr(start, end - start)));
            start = end + 1;
        }
        const double expected = parseDouble(fields[1]);
        const double actual = scalar::pythonSum(values);
        const bool   same   = std::isnan(expected) ? std::isnan(actual) : actual == expected && std::signbit(actual) == std::signbit(expected);
        failures.expect(same, "sum of " + line + " is " + std::to_string(actual));
        ++compared;
    }
    std::cout << "  " << compared << " lists compared\n";
}

void stressCasesEncodeAsInPython(const TestContext& context) {
    std::size_t compared = 0;
    for (const std::string& line : readLines(context.dataFolder / "stress_cases.tsv")) {
        const ReferenceCase expected = scalarTests::parseCase(line);
        const Outcome       actual   = scalarTests::encodeCase(*context.tokenizer, expected.row, expected.words, expected.featureSet);
        failures.expect(scalarTests::sameOutcome(expected, actual), scalarTests::describeMismatch(expected, actual));
        ++compared;
    }
    std::cout << "  " << compared << " cases compared\n";
}

void corpusRowsEncodeAsInPythonInThreeFeatureSets(const TestContext& context) {
    const std::vector<std::string> lines   = readLines(context.dataFolder / "corpus_words.tsv");
    const std::string              digests = readText(context.dataFolder / "corpus_digests.bin");
    const std::vector<std::string> tagger  = readLines(context.dataFolder / "tagger_rows.tsv");
    failures.expect(digests.size() == lines.size() * 12, "corpus_digests.bin holds three digests per row");
    if (digests.size() != lines.size() * 12) return;

    std::vector<scalar::IdentifierRow> taggerRows;
    for (const std::string& line : tagger) {
        const std::vector<std::string_view> fields = scalarTests::splitFields(line);
        taggerRows.push_back({"", scalarTests::unescapeField(fields[3]), scalarTests::unescapeField(fields[1]),
                              scalarTests::unescapeField(fields[2])});
    }

    const char* sets[] = {"legacy", "default with an empty type, language, and system", "default with the values of tagger_data.tsv"};
    std::array<std::size_t, 3>               mismatches{};
    std::array<std::chrono::nanoseconds, 3> elapsed{};
    for (std::size_t index = 0; index < lines.size(); ++index) {
        const std::vector<std::string_view> fields = scalarTests::splitFields(lines[index]);
        const std::vector<std::string>      words  = unescapedWords(fields, 1);
        scalar::IdentifierRow               row;
        row.context = scalarTests::contextOfLetter(fields[0].at(0));
        scalar::IdentifierRow taggerRow = taggerRows[index % taggerRows.size()];
        taggerRow.context               = row.context;
        const std::pair<const scalar::IdentifierRow*, const char*> runs[] = {{&row, "legacy"}, {&row, "default"}, {&taggerRow, "default"}};
        for (std::size_t set = 0; set < 3; ++set) {
            const auto    start   = std::chrono::steady_clock::now();
            const Outcome outcome = scalarTests::encodeCase(*context.tokenizer, *runs[set].first, words, runs[set].second);
            elapsed[set] += std::chrono::steady_clock::now() - start;

            std::uint32_t expected = 0;
            for (int byte = 3; byte >= 0; --byte) {
                expected = (expected << 8) | static_cast<unsigned char>(digests[index * 12 + set * 4 + static_cast<std::size_t>(byte)]);
            }
            if (scalarTests::digestOf(outcome) != expected) {
                ++mismatches[set];
                failures.fail("row " + std::to_string(index) + " " + sets[set] + ": " +
                              scalarTests::describeOutcome(runs[set].second, *runs[set].first, words, outcome));
            }
        }
    }
    for (std::size_t set = 0; set < 3; ++set) {
        const double microseconds = std::chrono::duration<double, std::micro>(elapsed[set]).count() / static_cast<double>(lines.size());
        std::cout << "  " << sets[set] << ": " << lines.size() << " rows, " << mismatches[set] << " mismatches, " << microseconds
                  << " microseconds per encoding\n";
    }
}

void taggerRowsEncodeAsInPythonWithTheirOwnTypeLanguageAndSystem(const TestContext& context) {
    std::size_t compared   = 0;
    std::size_t mismatches = 0;
    for (const std::string& line : readLines(context.dataFolder / "tagger_rows.tsv")) {
        const std::vector<std::string_view> fields = scalarTests::splitFields(line);
        const scalar::IdentifierRow row{scalarTests::unescapeField(fields[0]), scalarTests::unescapeField(fields[3]),
                                        scalarTests::unescapeField(fields[1]), scalarTests::unescapeField(fields[2])};
        const std::vector<std::string> words   = unescapedWords(fields, 5);
        const Outcome                  outcome = scalarTests::encodeCase(*context.tokenizer, row, words, "default");
        std::uint32_t                  expected = 0;
        std::from_chars(fields[4].data(), fields[4].data() + fields[4].size(), expected, 16);
        if (scalarTests::digestOf(outcome) != expected) {
            ++mismatches;
            failures.fail(scalarTests::describeOutcome("default", row, words, outcome));
        }
        ++compared;
    }
    std::cout << "  " << compared << " rows, " << mismatches << " mismatches\n";
}

void wordsWithoutASubtokenThrowIdentifierTooLongError(const TestContext& context) {
    const scalar::IdentifierRow row{"FUNCTION", "", "", ""};
    const auto throwsTooLong = [&context, &row](const std::vector<std::string>& words) {
        try {
            scalar::encodeIdentifier(*context.tokenizer, row, words, scalarTests::featureList("legacy"));
            return false;
        }
        catch (const scalar::IdentifierTooLongError& error) {
            return std::string(error.what()) == "Identifier is too long for the model input.";
        }
    };
    failures.expect(throwsTooLong(std::vector<std::string>(300, "a")), "300 words pass the 512-id limit");
    failures.expect(throwsTooLong({""}), "an empty word has no subtoken");
    failures.expect(throwsTooLong({"get", " \t"}), "a word of whitespace has no subtoken");
    failures.expect(!throwsTooLong({}), "an identifier without words encodes");
}

void bytesAboveAsciiAreRejected(const TestContext& context) {
    const auto rejects = [&context](const scalar::IdentifierRow& row, const std::vector<std::string>& words) {
        try {
            scalar::encodeIdentifier(*context.tokenizer, row, words, scalarTests::featureList("default"));
            return false;
        }
        catch (const std::invalid_argument&) {
            return true;
        }
    };
    failures.expect(rejects({"FUNCTION", "", "", ""}, {"caf\xc3\xa9"}), "a word in UTF-8");
    failures.expect(rejects({"FUNCTION", "", "\xff", ""}, {"x"}), "a type byte above 0x7F");
    failures.expect(rejects({"FUNCTION\x80", "", "", ""}, {"x"}), "a context byte above 0x7F");
    try {
        context.tokenizer->encode({"\xe4"});
        failures.fail("the tokenizer rejects a byte above 0x7F");
    }
    catch (const std::invalid_argument&) {
    }
}

struct Test {
    const char* name;
    void (*run)(const TestContext&);
};

const Test tests[] = {
    {"tokenizerLoadsTheVocabularyOfTheModelFolder", tokenizerLoadsTheVocabularyOfTheModelFolder},
    {"tokenizerRejectsSettingsThatThePortDoesNotFollow", tokenizerRejectsSettingsThatThePortDoesNotFollow},
    {"featureListsNormalizeAsInPython", featureListsNormalizeAsInPython},
    {"modelFeaturesComeFromTheConfigOrTheLegacyList", modelFeaturesComeFromTheConfigOrTheLegacyList},
    {"sequenceMatcherRatioEqualsDifflib", sequenceMatcherRatioEqualsDifflib},
    {"pythonSumEqualsTheBuiltInSum", pythonSumEqualsTheBuiltInSum},
    {"stressCasesEncodeAsInPython", stressCasesEncodeAsInPython},
    {"corpusRowsEncodeAsInPythonInThreeFeatureSets", corpusRowsEncodeAsInPythonInThreeFeatureSets},
    {"taggerRowsEncodeAsInPythonWithTheirOwnTypeLanguageAndSystem", taggerRowsEncodeAsInPythonWithTheirOwnTypeLanguageAndSystem},
    {"wordsWithoutASubtokenThrowIdentifierTooLongError", wordsWithoutASubtokenThrowIdentifierTooLongError},
    {"bytesAboveAsciiAreRejected", bytesAboveAsciiAreRejected},
};

} // namespace

// Runs every test, or the one test named by the third argument
int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: encoderTests <golden data folder> <model folder> [test name]\n";
        return 2;
    }
    const std::string only = argc == 4 ? argv[3] : "";
    try {
        const scalar::WordPieceTokenizer tokenizer(argv[2]);
        const TestContext                context{argv[1], argv[2], &tokenizer};
        std::size_t                      failed = 0;
        std::size_t                      ran    = 0;
        for (const Test& test : tests) {
            if (!only.empty() && only != test.name) continue;
            std::cout << test.name << "\n";
            const std::size_t before = failures.count;
            failures.startTest();
            try {
                test.run(context);
            }
            catch (const std::exception& error) {
                failures.fail(std::string("exception: ") + error.what());
            }
            ++ran;
            if (failures.count != before) {
                ++failed;
                std::cout << "  FAILED with " << failures.count - before << " failures\n";
            }
        }
        if (ran == 0) {
            std::cerr << "no test named " << only << "\n";
            return 2;
        }
        std::cout << ran - failed << " of " << ran << " tests passed\n";
        return failed == 0 ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
