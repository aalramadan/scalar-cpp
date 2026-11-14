// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file crfDecoder.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Finds the best label sequence of a linear-chain CRF with the Viterbi algorithm of torchcrf.
// Port of CRF._viterbi_decode of torchcrf/__init__.py, pytorch-crf 0.7.2, MIT license.

#ifndef SCALAR_CRF_DECODER_HPP
#define SCALAR_CRF_DECODER_HPP

#include <cstddef>
#include <vector>

namespace scalar {

// The scores of a CRF and its Viterbi decoding of one sequence without padding. The arithmetic
// follows torchcrf in float32, therefore equal emissions give equal labels, ties included.
class CrfDecoder {
public:
    // An empty CRF, which the model replaces with the one of its checkpoint
    CrfDecoder() = default;

    // Scores of a CRF with tagCount tags; transitionScores[from * tagCount + to] scores the tag
    // bigram (from, to). Throws std::invalid_argument when a vector has the wrong size.
    CrfDecoder(std::size_t tagCount, std::vector<float> startScores, std::vector<float> endScores,
               std::vector<float> transitionScores);

    // Number of tags, the width of each emission row
    std::size_t tagCount() const { return tags; }

    // Best tag of each of the length steps of emissions, which holds length × tagCount() scores in
    // row-major order. A tie goes to the lower tag, as torch.max does. An empty sequence gives no tags.
    std::vector<int> decode(const float* emissions, std::size_t length) const;

private:
    std::size_t        tags = 0;
    std::vector<float> start;
    std::vector<float> end;
    std::vector<float> transitions;
};

}

#endif
