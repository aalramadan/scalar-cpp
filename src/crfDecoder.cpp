// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file crfDecoder.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Decodes the best label sequence of a CRF with the float32 Viterbi recursion of torchcrf.
// Port of CRF._viterbi_decode of torchcrf/__init__.py, pytorch-crf 0.7.2, MIT license.

#include "crfDecoder.hpp"

#include <stdexcept>
#include <utility>

namespace scalar {

CrfDecoder::CrfDecoder(std::size_t tagCount, std::vector<float> startScores, std::vector<float> endScores,
                       std::vector<float> transitionScores)
    : tags(tagCount), start(std::move(startScores)), end(std::move(endScores)), transitions(std::move(transitionScores)) {
    if (tags == 0 || start.size() != tags || end.size() != tags || transitions.size() != tags * tags) {
        throw std::invalid_argument("a CRF needs tagCount start scores, end scores, and tagCount² transition scores");
    }
}

std::vector<int> CrfDecoder::decode(const float* emissions, std::size_t length) const {
    if (length == 0) {
        return {};
    }
    std::vector<float> score(tags);
    std::vector<float> next(tags);
    std::vector<int>   history((length - 1) * tags);
    for (std::size_t to = 0; to < tags; ++to) {
        score[to] = start[to] + emissions[to];
    }
    for (std::size_t step = 1; step < length; ++step) {
        const float* emission = emissions + step * tags;
        int*         origin   = history.data() + (step - 1) * tags;
        for (std::size_t to = 0; to < tags; ++to) {
            // torchcrf adds the transition to the score first and the emission second
            float best     = (score[0] + transitions[to]) + emission[to];
            int   bestFrom = 0;
            for (std::size_t from = 1; from < tags; ++from) {
                const float candidate = (score[from] + transitions[from * tags + to]) + emission[to];
                if (candidate > best) {
                    best     = candidate;
                    bestFrom = static_cast<int>(from);
                }
            }
            next[to]   = best;
            origin[to] = bestFrom;
        }
        score.swap(next);
    }
    int last = 0;
    float bestScore = score[0] + end[0];
    for (std::size_t tag = 1; tag < tags; ++tag) {
        const float candidate = score[tag] + end[tag];
        if (candidate > bestScore) {
            bestScore = candidate;
            last      = static_cast<int>(tag);
        }
    }
    std::vector<int> path(length);
    path[length - 1] = last;
    for (std::size_t step = length - 1; step > 0; --step) {
        path[step - 1] = history[(step - 1) * tags + static_cast<std::size_t>(path[step])];
    }
    return path;
}

}
