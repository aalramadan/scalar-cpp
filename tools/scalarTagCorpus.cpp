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
// and context. The word lists set the dictionary flag DW, as the en.txt files of the SCALAR server do.

#include "scalarTagger.hpp"

#include "json.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// The fields of a line without its line end, which is \n or \r\n
std::vector<std::string> splitTabs(std::string line) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) fields.push_back(field);
    return fields;
}

std::vector<scalar::NameInContext> readCorpus(const std::string& path) {
    std::ifstream corpus(path);
    if (!corpus) throw std::runtime_error("cannot read " + path);
    std::string line;
    std::getline(corpus, line);
    const std::vector<std::string> header = splitTabs(line);
    const auto column = [&](const char* name) {
        const auto found = std::find(header.begin(), header.end(), name);
        if (found == header.end()) throw std::runtime_error(path + " has no column " + name);
        return static_cast<std::size_t>(found - header.begin());
    };
    const std::size_t nameColumn = column("name");
    const std::size_t contextColumn = column("context");

    std::vector<scalar::NameInContext> names;
    while (std::getline(corpus, line)) {
        // csv.DictReader of tools/tag_with_python_scalar.py skips empty lines as well
        if (line.empty() || line == "\r") continue;
        const std::vector<std::string> fields = splitTabs(line);
        if (fields.size() <= std::max(nameColumn, contextColumn)) throw std::runtime_error("short row in " + path + ": " + line);
        names.push_back({fields[nameColumn], fields[contextColumn]});
    }
    return names;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: scalarTagCorpus <corpus.tsv> <answers.jsonl> [threads [word list]...]" << std::endl;
        return EXIT_FAILURE;
    }
    try {
        const std::vector<scalar::NameInContext> names = readCorpus(argv[1]);
        const unsigned threads = argc >= 4 ? static_cast<unsigned>(std::stoul(argv[3])) : std::thread::hardware_concurrency();
        const std::vector<std::string> wordLists(argv + std::min(argc, 4), argv + argc);
        const scalar::ScalarTagger tagger(scalar::folderFiles(SCALAR_DATA_DIR, SCALAR_RONIN_DIR, wordLists));
        const std::vector<scalar::TagResult> results = tagger.tagAll(names, threads);

        std::ofstream answers(argv[2]);
        for (std::size_t i = 0; i < names.size(); ++i) {
            nlohmann::json answer = {{"name", names[i].name}, {"context", names[i].context}};
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
