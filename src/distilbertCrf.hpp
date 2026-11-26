// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file distilbertCrf.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Tags the words of an identifier with the DistilBERT encoder and the constrained CRF of SCALAR.
// Port of DistilBertCRFForTokenClassification of scalar_tagger/lm_based_tagger/distilbert_crf.py, scanl_tagger, GPL-3.0.

#ifndef SCALAR_DISTILBERT_CRF_HPP
#define SCALAR_DISTILBERT_CRF_HPP

#include "crfDecoder.hpp"
#include "linearLayer.hpp"
#include "modelInput.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace scalar {

// DistilBertCRFForTokenClassification in eval mode, loaded from config.json and model.safetensors of a
// model folder. The CRF scores the transitions of release::forbiddenTransitions with -1e4. The const
// methods may run on several threads at once, each call on one thread.
class DistilBertCrf {
public:
    // Throws std::runtime_error when the folder lacks a file or a file breaks the format or the config.
    explicit DistilBertCrf(const std::string& modelDirectory);

    // Label id of each word. Throws std::invalid_argument unless 3 <= inputIds.size() <= maxInputLength(),
    // each id is below vocabularySize(), and the word positions increase strictly within [1, size - 2].
    std::vector<int> decode(const std::vector<std::int32_t>& inputIds, const std::vector<std::size_t>& wordPositions) const;

    // decode for several identifiers in one pass: the linear layers take the rows of all identifiers
    // at once, and each identifier attends only to its own tokens. Equals decode bit for bit.
    std::vector<std::vector<int>> decodeBatch(const std::vector<ModelInput>& inputs) const;

    // Emissions of each identifier at its word positions: words × labelNames().size() scores in row-major order.
    std::vector<std::vector<float>> wordEmissions(const std::vector<ModelInput>& inputs) const;

    // Label names in the order of their ids, from id2label of config.json
    const std::vector<std::string>& labelNames() const { return labels; }

    // The CRF with the forbidden transitions in place
    const CrfDecoder& crf() const { return decoder; }

    // Most token ids per input, max_position_embeddings of the config
    std::size_t maxInputLength() const { return maxPositions; }

    // Token ids below this bound are valid, the rows of the word embeddings in the checkpoint
    std::size_t vocabularySize() const { return vocabulary; }

private:
    struct EncoderLayer {
        LinearLayer        queryKeyValue;   // q_lin, k_lin, and v_lin side by side
        LinearLayer        attentionOutput; // out_lin
        LinearLayer        expand;          // ffn.lin1
        LinearLayer        contract;        // ffn.lin2
        std::vector<float> attentionNormWeight;
        std::vector<float> attentionNormBias;
        std::vector<float> outputNormWeight;
        std::vector<float> outputNormBias;
    };

    // Throws std::invalid_argument for an input outside the contract of decode
    void checkInput(const ModelInput& input) const;

    // Emissions at the word positions of all inputs of the batch, which must have words, one after another
    void encodeBatch(const std::vector<const ModelInput*>& batch, std::vector<float>& emissions) const;

    std::size_t               dim          = 0;
    std::size_t               heads        = 0;
    std::size_t               hiddenDim    = 0;
    std::size_t               maxPositions = 0;
    std::size_t               vocabulary   = 0;
    float                     attentionScale = 0.0f;
    std::vector<float>        wordEmbeddings;
    std::vector<float>        positionEmbeddings;
    std::vector<float>        embeddingNormWeight;
    std::vector<float>        embeddingNormBias;
    std::vector<EncoderLayer> layers;
    LinearLayer               classifier;
    CrfDecoder                decoder;
    std::vector<std::string>  labels;
};

}

#endif
