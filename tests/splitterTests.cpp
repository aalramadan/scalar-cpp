// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file splitterTests.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// Tests of the splitter and its stemmer against golden files that tests/make_golden.py writes
// from Spiral dff5373 and NLTK 3.10.3.

#include "englishStemmer.hpp"
#include "escapedLines.hpp"
#include "heuristicSplitter.hpp"
#include "ronin.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Folders {
    std::string data;
    std::string golden;
};

using Rows = std::vector<std::vector<std::string>>;
using Tokens = std::vector<std::string>;

// Prints the first failures of a test and counts all of them
class Failures {
public:
    void add(const std::string& message) {
        if (count < 10) {
            std::cout << "    " << message << '\n';
        }
        ++count;
    }

    std::size_t total() const {
        return count;
    }

private:
    std::size_t count = 0;
};

Rows readRows(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot read the golden file " + path);
    }
    Rows rows;
    std::string line;
    while (std::getline(input, line)) {
        rows.push_back(scalar::unescapeFields(line));
    }
    return rows;
}

std::string shown(const Tokens& tokens) {
    std::string text = "[";
    for (const std::string& token : tokens) {
        text += (text.size() > 1 ? ", '" : "'") + scalar::escapeField(token) + "'";
    }
    return text + "]";
}

std::uint64_t bitsOf(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

std::size_t stemmerMatchesNltkOnGoldenWords(const Folders& folders) {
    Failures failures;
    for (const auto& row : readRows(folders.golden + "/stem_golden.tsv")) {
        const std::string stem = scalar::stemEnglish(row.at(0));
        if (stem != row.at(1)) {
            failures.add("'" + scalar::escapeField(row[0]) + "': NLTK '" + scalar::escapeField(row[1]) + "', C++ '" +
                         scalar::escapeField(stem) + "'");
        }
    }
    return failures.total();
}

std::size_t splitMatchesPythonOnGoldenIdentifiers(const Folders& folders) {
    const scalar::Ronin ronin(folders.data);
    Failures failures;
    for (const auto& row : readRows(folders.golden + "/split_golden.tsv")) {
        const Tokens expected(row.begin() + 1, row.end());
        const Tokens tokens = ronin.split(row.at(0));
        if (tokens != expected) {
            failures.add("'" + scalar::escapeField(row[0]) + "': Python " + shown(expected) + ", C++ " + shown(tokens));
        }
    }
    return failures.total();
}

std::size_t splitGivesTheGoldenTokensOnEightThreadsAtOnce(const Folders& folders) {
    const scalar::Ronin ronin(folders.data);
    const Rows rows = readRows(folders.golden + "/split_golden.tsv");
    std::atomic<std::size_t> mismatches{0};
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < 8; ++t) {
        threads.emplace_back([&ronin, &rows, &mismatches, t] {
            for (std::size_t i = 0; i < rows.size(); ++i) {
                const std::vector<std::string>& row = rows[(i + t * rows.size() / 8) % rows.size()];
                if (ronin.split(row[0]) != Tokens(row.begin() + 1, row.end())) {
                    ++mismatches;
                }
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    Failures failures;
    if (mismatches > 0) {
        failures.add(std::to_string(mismatches.load()) + " splits differ from the golden tokens");
    }
    return failures.total();
}

std::size_t longestTermWithNumbersWinsWhereTermsCompete(const Folders& folders) {
    const scalar::Ronin ronin(folders.data);
    const std::vector<std::pair<std::string, Tokens>> cases{
        {"code128a", {"code128a"}},          {"CODE128B", {"CODE128B"}},         {"use3des", {"use", "3des"}},
        {"ipv6cpState", {"ipv6cp", "State"}}, {"mpeg25Decoder", {"mpeg25", "Decoder"}}, {"html40Parser", {"html40", "Parser"}},
        {"g729aCodec", {"g729a", "Codec"}},  {"x11r6Window", {"x11r6", "Window"}},  {"x214y", {"x214", "y"}},
        {"vp6sFrame", {"vp6s", "Frame"}},    {"ec2nKey", {"ec2n", "Key"}},         {"d2d2tLink", {"d2d2t", "Link"}},
        {"3gpp2Header", {"3gpp2", "Header"}}, {"sha1024", {"sha1024"}}};
    Failures failures;
    for (const auto& [identifier, expected] : cases) {
        const Tokens tokens = ronin.split(identifier);
        if (tokens != expected) {
            failures.add("'" + identifier + "': expected " + shown(expected) + ", C++ " + shown(tokens));
        }
    }
    return failures.total();
}

std::size_t termPatternsKeepTheirRegexMeaning(const Folders& folders) {
    const scalar::HeuristicSplitter splitter({"iso[0-9]{2,}", "rfc[0-9]{3,}", "utf8"}, {"16"});
    const std::vector<std::pair<std::string, std::size_t>> matches{
        {"iso9001x", 7}, {"ISO9", 0}, {"iso", 0}, {"RFC2616", 7}, {"rfc26", 0}, {"Utf8", 4}, {"iso[0-9]{2,}", 0}};
    Failures failures;
    for (const auto& [text, length] : matches) {
        if (splitter.longestTermAt(text, 0) != length) {
            failures.add("'" + text + "': expected a match of " + std::to_string(length) + " bytes, found " +
                         std::to_string(splitter.longestTermAt(text, 0)));
        }
    }
    // Set membership reads the literal entry, as Python's 'in common_terms_with_numbers' does
    const scalar::Ronin ronin(folders.data);
    const Tokens expected{"iso[0-9]{2,}", "16"};
    if (ronin.split("iso[0-9]{2,}16") != expected) {
        failures.add("'iso[0-9]{2,}16': expected " + shown(expected) + ", C++ " + shown(ronin.split("iso[0-9]{2,}16")));
    }
    return failures.total();
}

std::size_t identifiersWithoutTokensGiveNoTokens(const Folders& folders) {
    const scalar::Ronin ronin(folders.data);
    Failures failures;
    for (const std::string identifier : {"", "_", "$~_.:/@", "   ", "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f"}) {
        if (!ronin.split(identifier).empty()) {
            failures.add("'" + scalar::escapeField(identifier) + "': expected [], C++ " + shown(ronin.split(identifier)));
        }
    }
    return failures.total();
}

std::size_t unsupportedTermSyntaxIsRejected(const Folders&) {
    Failures failures;
    for (const std::string term : {"a(b", "x|y", "a.b", "[0-9]*", "[0-9]{2", "[]", "a{0}", "[9-0]"}) {
        try {
            scalar::HeuristicSplitter({term}, {});
            failures.add("'" + term + "' was accepted");
        } catch (const std::invalid_argument&) {
        }
    }
    return failures.total();
}

std::size_t loadingAFolderWithoutTablesThrows(const Folders& folders) {
    Failures failures;
    try {
        const scalar::Ronin ronin(folders.golden + "/no-such-folder");
        failures.add("the constructor returned");
    } catch (const std::runtime_error&) {
    }
    return failures.total();
}

std::size_t platformPowMatchesPythonMathPowForEveryRescaledCount(const Folders& folders) {
    Failures failures;
    const auto check = [&failures](const std::string& base, double exponent, const std::string& bits) {
        const double value = std::pow(std::stod(base), exponent);
        if (bitsOf(value) != std::stoull(bits, nullptr, 16)) {
            failures.add("pow(" + base + ", " + std::to_string(exponent) + "): Python bits " + bits + ", C++ " + std::to_string(value));
        }
    };
    for (const auto& row : readRows(folders.golden + "/rescale_golden.tsv")) {
        check(row.at(0), scalar::RoninParameters::dictWordExponent, row.at(1));
        check(row.at(0), scalar::RoninParameters::normalExponent, row.at(2));
    }
    for (const auto& row : readRows(folders.golden + "/alt_golden.tsv")) {
        check(row.at(0), scalar::RoninParameters::altExponent, row.at(1));
    }
    return failures.total();
}

const std::vector<std::pair<std::string, std::function<std::size_t(const Folders&)>>> tests{
    {"stemmerMatchesNltkOnGoldenWords", stemmerMatchesNltkOnGoldenWords},
    {"splitMatchesPythonOnGoldenIdentifiers", splitMatchesPythonOnGoldenIdentifiers},
    {"splitGivesTheGoldenTokensOnEightThreadsAtOnce", splitGivesTheGoldenTokensOnEightThreadsAtOnce},
    {"longestTermWithNumbersWinsWhereTermsCompete", longestTermWithNumbersWinsWhereTermsCompete},
    {"termPatternsKeepTheirRegexMeaning", termPatternsKeepTheirRegexMeaning},
    {"identifiersWithoutTokensGiveNoTokens", identifiersWithoutTokensGiveNoTokens},
    {"unsupportedTermSyntaxIsRejected", unsupportedTermSyntaxIsRejected},
    {"loadingAFolderWithoutTablesThrows", loadingAFolderWithoutTablesThrows},
    {"platformPowMatchesPythonMathPowForEveryRescaledCount", platformPowMatchesPythonMathPowForEveryRescaledCount},
};

}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: splitterTests <test name or all> <data folder> <golden folder>\n";
        return 2;
    }
    const std::string selected = argv[1];
    const Folders folders{argv[2], argv[3]};
    bool found = false;
    bool passed = true;
    for (const auto& [name, test] : tests) {
        if (selected != "all" && selected != name) {
            continue;
        }
        found = true;
        std::size_t failures = 0;
        try {
            failures = test(folders);
        } catch (const std::exception& error) {
            std::cout << "    exception: " << error.what() << '\n';
            failures = 1;
        }
        std::cout << (failures == 0 ? "PASS " : "FAIL ") << name;
        if (failures > 0) {
            std::cout << " (" << failures << " failures)";
        }
        std::cout << std::endl;
        passed = passed && failures == 0;
    }
    if (!found) {
        std::cerr << "no test named " << selected << '\n';
        return 2;
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
