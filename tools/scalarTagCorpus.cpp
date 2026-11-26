// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file scalarTagCorpus.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// Tags each row of a corpus with the SCALAR tagger and writes one JSON line per row in the form
// of tools/tag_with_python_scalar.py, which tools/compare_tag_answers.py compares word by word.
// Usage: scalarTagCorpus <corpus.tsv> <answers.jsonl> [threads [word list]...], with a corpus header naming name
// and context, and optionally type, language, and system. The word lists set the dictionary flag DW, as the
// en.txt files of the SCALAR server do.

#include "scalarTagger.hpp"

#include "json.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

// The fields of a line without its line end, which is \n or \r\n. An empty last field stays, as csv.DictReader
//   keeps it, for a row with no type in the last column.
std::vector<std::string> splitTabs(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> fields;
    std::size_t start = 0;
    for (std::size_t tab = line.find('\t'); tab != std::string::npos; tab = line.find('\t', start)) {
        fields.push_back(line.substr(start, tab - start));
        start = tab + 1;
    }
    fields.push_back(line.substr(start));
    return fields;
}

// A column besides name and context that a corpus may have, with the input of the tagger that it fills
struct OptionalColumn {
    const char*                        name;
    std::string scalar::NameInContext::*member;
};

// In the order of tools/tag_with_python_scalar.py
constexpr OptionalColumn optionalColumns[] = {
    {"type", &scalar::NameInContext::type}, {"language", &scalar::NameInContext::language}, {"system", &scalar::NameInContext::system}};

struct Corpus {
    std::vector<scalar::NameInContext> rows;
    std::vector<OptionalColumn>        optionalColumnsPresent;
};

Corpus readCorpus(const std::string& path) {
    std::ifstream corpus(path);
    if (!corpus) throw std::runtime_error("cannot read " + path);
    std::string line;
    std::getline(corpus, line);
    const std::vector<std::string> header = splitTabs(line);
    const auto findColumn = [&](const std::string& name) {
        return static_cast<std::size_t>(std::find(header.begin(), header.end(), name) - header.begin());
    };
    const std::size_t nameColumn    = findColumn("name");
    const std::size_t contextColumn = findColumn("context");
    if (nameColumn == header.size() || contextColumn == header.size()) throw std::runtime_error(path + " lacks the column name or context");

    Corpus result;
    std::vector<std::size_t> optionalIndices;
    for (const OptionalColumn& column : optionalColumns) {
        if (findColumn(column.name) == header.size()) continue;
        result.optionalColumnsPresent.push_back(column);
        optionalIndices.push_back(findColumn(column.name));
    }
    while (std::getline(corpus, line)) {
        // csv.DictReader of tools/tag_with_python_scalar.py skips empty lines as well
        if (line.empty() || line == "\r") continue;
        const std::vector<std::string> fields = splitTabs(line);
        scalar::NameInContext row;
        const auto field = [&](std::size_t index) -> const std::string& {
            if (index >= fields.size()) throw std::runtime_error("short row in " + path + ": " + line);
            return fields[index];
        };
        row.name    = field(nameColumn);
        row.context = field(contextColumn);
        for (std::size_t k = 0; k < optionalIndices.size(); ++k) {
            row.*result.optionalColumnsPresent[k].member = field(optionalIndices[k]);
        }
        result.rows.push_back(std::move(row));
    }
    return result;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: scalarTagCorpus <corpus.tsv> <answers.jsonl> [threads [word list]...]" << std::endl;
        return EXIT_FAILURE;
    }
    try {
        Corpus corpus = readCorpus(argv[1]);
        const std::vector<scalar::NameInContext>& names = corpus.rows;
        const unsigned threads = argc >= 4 ? static_cast<unsigned>(std::stoul(argv[3])) : std::thread::hardware_concurrency();
        const std::vector<std::string> wordLists(argv + std::min(argc, 4), argv + argc);
        const scalar::ScalarTagger tagger(scalar::folderFiles(SCALAR_DATA_DIR, SCALAR_RONIN_DIR, wordLists));
        const std::vector<scalar::TagResult> results = tagger.tagAll(names, threads);

        std::ofstream answers(argv[2]);
        for (std::size_t i = 0; i < names.size(); ++i) {
            nlohmann::json answer = {{"name", names[i].name}, {"context", names[i].context}};
            for (const OptionalColumn& column : corpus.optionalColumnsPresent) answer[column.name] = names[i].*column.member;
            if (results[i].tagged) {
                answer["words"] = nlohmann::json::array();
                for (const scalar::TaggedWord& word : results[i].words) {
                    nlohmann::json entry = nlohmann::json::object();
                    entry[word.word] = {{"dictionary", word.dictionary}, {"tag", word.tag}};
                    answer["words"].push_back(entry);
                }
            }
            else {
                answer["error"] = results[i].error;
            }
            answers << answer.dump() << '\n';
        }
        if (!answers) throw std::runtime_error(std::string("cannot write ") + argv[2]);
    }
    catch (const std::exception& e) {
        std::cerr << "scalarTagCorpus: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
