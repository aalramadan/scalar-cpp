// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file goldenData.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Reads the reference files that tests/write_golden_files.py writes from the Python SCALAR model.

#ifndef SCALAR_GOLDEN_DATA_HPP
#define SCALAR_GOLDEN_DATA_HPP

#include "crfDecoder.hpp"
#include "distilbertCrf.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace scalar {

// One row of a reference file: an identifier of identifiers_tag.tsv and what the Python model gives it
struct ReferenceRow {
    std::size_t        row = 0;          // index in identifiers_tag.tsv, the header excluded
    std::string        name;
    std::string        context;
    bool               tooLong = false;  // tag_identifier raised the too-long error, therefore no input
    ModelInput         input;
    std::vector<int>   labels;
    std::vector<float> emissions;        // words × labels in row-major order
    double             margin = 0.0;     // best minus second-best Viterbi score, NaN without words
};

// Rows of a reference file. Throws std::runtime_error when the file cannot be read or parsed.
std::vector<ReferenceRow> readReferenceRows(const std::string& path);

// One sequence of crf_cases.txt with the path that torchcrf decodes
struct CrfCase {
    std::size_t        length = 0;
    std::vector<float> emissions;
    std::vector<int>   path;
};

// A CRF of crf_cases.txt with its cases
struct CrfCaseSet {
    CrfDecoder           decoder;
    std::vector<CrfCase> cases;
};

// The CRFs and cases of crf_cases.txt. Throws std::runtime_error when the file cannot be parsed.
std::vector<CrfCaseSet> readCrfCases(const std::string& path);

// Values of one kernel case of kernel_cases.txt: a name and its float32 arrays, in file order
struct KernelCase {
    std::string                     name;
    std::vector<std::size_t>        sizes;
    std::vector<std::vector<float>> arrays;
};

// The cases of kernel_cases.txt. Throws std::runtime_error when the file cannot be parsed.
std::vector<KernelCase> readKernelCases(const std::string& path);

}

#endif
