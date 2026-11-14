// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file distilbertCrf.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Loads the DistilBERT and CRF model of SCALAR and computes its emissions and labels for identifiers.
// Port of DistilBertCRFForTokenClassification of src/lm_based_tagger/distilbert_crf.py, scanl_tagger 33f0327, GPL-3.0.
// Port of DistilBertModel of transformers/models/distilbert/modeling_distilbert.py, transformers 5.19.0, Apache-2.0.

#include "distilbertCrf.hpp"

#include "dataFile.hpp"
#include "encoderKernels.hpp"
#include "safetensors.hpp"

#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace scalar {

namespace {

constexpr float LAYER_NORM_EPSILON = 1e-12f; // every nn.LayerNorm of DistilBERT uses eps=1e-12
constexpr float FORBIDDEN_SCORE    = -1e4f;  // the score _constrained_crf gives a forbidden transition

// FORBIDDEN_TRANSITIONS of distilbert_crf.py: a noun directly followed by another noun should have been NM
const std::pair<const char*, const char*> FORBIDDEN_TRANSITIONS[] = {{"N", "N"}, {"N", "NPL"}, {"NPL", "NPL"}};

struct ModelConfig {
    std::size_t              dim          = 0;
    std::size_t              layerCount   = 0;
    std::size_t              heads        = 0;
    std::size_t              hiddenDim    = 0;
    std::size_t              maxPositions = 0;
    std::vector<std::string> labels;
};

[[noreturn]] void failConfig(const std::string& path, const std::string& reason) {
    throw std::runtime_error("invalid model config '" + path + "': " + reason);
}

// A positive size of the config, or the default of DistilBertConfig when the key is absent
std::size_t configSize(const nlohmann::json& config, const std::string& path, const char* key, std::size_t fallback) {
    if (!config.contains(key)) {
        return fallback;
    }
    const nlohmann::json& value = config[key];
    if (!value.is_number_unsigned() || value.get<std::uint64_t>() == 0) {
        failConfig(path, std::string(key) + " is no positive integer");
    }
    return value.get<std::size_t>();
}

std::vector<std::string> readLabels(const nlohmann::json& config, const std::string& path) {
    if (!config.contains("id2label") || !config["id2label"].is_object() || config["id2label"].empty()) {
        failConfig(path, "id2label is missing or empty");
    }
    std::map<std::size_t, std::string> byId;
    for (const auto& item : config["id2label"].items()) {
        const std::string& key = item.key();
        if (key.empty() || key.size() > 9 || key.find_first_not_of("0123456789") != std::string::npos ||
            !item.value().is_string()) {
            failConfig(path, "id2label maps '" + key + "' to no label");
        }
        byId[std::stoul(key)] = item.value().get<std::string>();
    }
    std::vector<std::string> labels;
    for (const auto& item : byId) {
        if (item.first != labels.size()) {
            failConfig(path, "the ids of id2label are not 0 to " + std::to_string(byId.size() - 1));
        }
        labels.push_back(item.second);
    }
    if (std::set<std::string>(labels.begin(), labels.end()).size() != labels.size()) {
        failConfig(path, "id2label repeats a label");
    }
    return labels;
}

ModelConfig readConfig(const std::string& path) {
    const std::optional<DataFile> file = openDataFile(path);
    if (!file) {
        throw std::runtime_error("cannot read the model config '" + path + "'");
    }
    try {
        const nlohmann::json config = nlohmann::json::parse(file->bytes());
        if (!config.is_object()) {
            failConfig(path, "it is no JSON object");
        }
        ModelConfig result;
        result.dim          = configSize(config, path, "dim", 768);
        result.layerCount   = configSize(config, path, "n_layers", 6);
        result.heads        = configSize(config, path, "n_heads", 12);
        result.hiddenDim    = configSize(config, path, "hidden_dim", 3072);
        result.maxPositions = configSize(config, path, "max_position_embeddings", 512);
        if (result.dim % result.heads != 0) {
            failConfig(path, "n_heads does not divide dim");
        }
        if (config.value("activation", std::string("gelu")) != "gelu") {
            failConfig(path, "the activation is not gelu");
        }
        result.labels = readLabels(config, path);
        return result;
    } catch (const nlohmann::json::exception& error) {
        failConfig(path, error.what());
    }
}

std::string layerPrefix(std::size_t layer) {
    return "bert.transformer.layer." + std::to_string(layer) + ".";
}

// The parameter names of the model, which the checkpoint must hold exactly, as load_state_dict does
std::set<std::string> parameterNames(std::size_t layerCount) {
    std::set<std::string> names = {
        "bert.embeddings.word_embeddings.weight", "bert.embeddings.position_embeddings.weight",
        "bert.embeddings.LayerNorm.weight",       "bert.embeddings.LayerNorm.bias",
        "classifier.weight",                      "classifier.bias",
        "crf.start_transitions",                  "crf.end_transitions",
        "crf.transitions",
    };
    for (std::size_t layer = 0; layer < layerCount; ++layer) {
        for (const char* part : {"attention.q_lin", "attention.k_lin", "attention.v_lin", "attention.out_lin", "ffn.lin1",
                                 "ffn.lin2", "sa_layer_norm", "output_layer_norm"}) {
            names.insert(layerPrefix(layer) + part + ".weight");
            names.insert(layerPrefix(layer) + part + ".bias");
        }
    }
    return names;
}

void checkParameterNames(const SafetensorsFile& weights, std::size_t layerCount, const std::string& path) {
    const std::set<std::string> expected = parameterNames(layerCount);
    const std::vector<std::string> names = weights.tensorNames();
    const std::set<std::string> present(names.begin(), names.end());
    for (const std::string& name : expected) {
        if (present.count(name) == 0) {
            throw std::runtime_error("the checkpoint '" + path + "' lacks the tensor '" + name + "'");
        }
    }
    for (const std::string& name : present) {
        if (expected.count(name) == 0) {
            throw std::runtime_error("the checkpoint '" + path + "' holds the unexpected tensor '" + name + "'");
        }
    }
}

void append(std::vector<float>& target, const std::vector<float>& values) {
    target.insert(target.end(), values.begin(), values.end());
}

}

DistilBertCrf::DistilBertCrf(const std::string& modelDirectory) {
    const std::filesystem::path folder(modelDirectory);
    const ModelConfig config = readConfig((folder / "config.json").string());
    const std::string weightsPath = (folder / "model.safetensors").string();
    const SafetensorsFile weights(weightsPath);
    checkParameterNames(weights, config.layerCount, weightsPath);

    dim          = config.dim;
    heads        = config.heads;
    hiddenDim    = config.hiddenDim;
    maxPositions = config.maxPositions;
    labels       = config.labels;
    // attention_head_size**-0.5 in double, rounded to the float scale of the flash attention
    attentionScale = static_cast<float>(std::pow(static_cast<double>(dim / heads), -0.5));

    // from_pretrained resizes the word embeddings to the vocabulary of the checkpoint
    const TensorEntry& wordEntry = weights.entry("bert.embeddings.word_embeddings.weight");
    if (wordEntry.shape.size() != 2 || wordEntry.shape[0] == 0 || wordEntry.shape[1] != dim) {
        throw std::runtime_error("the word embeddings of '" + weightsPath + "' do not have " + std::to_string(dim) + " columns");
    }
    vocabulary          = wordEntry.shape[0];
    wordEmbeddings      = weights.readF32("bert.embeddings.word_embeddings.weight", {vocabulary, dim});
    positionEmbeddings  = weights.readF32("bert.embeddings.position_embeddings.weight", {maxPositions, dim});
    embeddingNormWeight = weights.readF32("bert.embeddings.LayerNorm.weight", {dim});
    embeddingNormBias   = weights.readF32("bert.embeddings.LayerNorm.bias", {dim});

    for (std::size_t index = 0; index < config.layerCount; ++index) {
        const std::string prefix = layerPrefix(index);
        EncoderLayer layer;
        std::vector<float> queryKeyValueWeight;
        std::vector<float> queryKeyValueBias;
        for (const char* projection : {"attention.q_lin", "attention.k_lin", "attention.v_lin"}) {
            append(queryKeyValueWeight, weights.readF32(prefix + projection + ".weight", {dim, dim}));
            append(queryKeyValueBias, weights.readF32(prefix + projection + ".bias", {dim}));
        }
        layer.queryKeyValue   = LinearLayer(queryKeyValueWeight, queryKeyValueBias, dim);
        layer.attentionOutput = LinearLayer(weights.readF32(prefix + "attention.out_lin.weight", {dim, dim}),
                                            weights.readF32(prefix + "attention.out_lin.bias", {dim}), dim);
        layer.expand   = LinearLayer(weights.readF32(prefix + "ffn.lin1.weight", {hiddenDim, dim}),
                                     weights.readF32(prefix + "ffn.lin1.bias", {hiddenDim}), dim);
        layer.contract = LinearLayer(weights.readF32(prefix + "ffn.lin2.weight", {dim, hiddenDim}),
                                     weights.readF32(prefix + "ffn.lin2.bias", {dim}), hiddenDim);
        layer.attentionNormWeight = weights.readF32(prefix + "sa_layer_norm.weight", {dim});
        layer.attentionNormBias   = weights.readF32(prefix + "sa_layer_norm.bias", {dim});
        layer.outputNormWeight    = weights.readF32(prefix + "output_layer_norm.weight", {dim});
        layer.outputNormBias      = weights.readF32(prefix + "output_layer_norm.bias", {dim});
        layers.push_back(std::move(layer));
    }

    const std::size_t labelCount = labels.size();
    classifier = LinearLayer(weights.readF32("classifier.weight", {labelCount, dim}),
                             weights.readF32("classifier.bias", {labelCount}), dim);
    std::vector<float> transitions = weights.readF32("crf.transitions", {labelCount, labelCount});
    std::map<std::string, std::size_t> labelIds;
    for (std::size_t id = 0; id < labelCount; ++id) {
        labelIds[labels[id]] = id;
    }
    for (const auto& forbidden : FORBIDDEN_TRANSITIONS) {
        const auto from = labelIds.find(forbidden.first);
        const auto to   = labelIds.find(forbidden.second);
        if (from != labelIds.end() && to != labelIds.end()) {
            transitions[from->second * labelCount + to->second] = FORBIDDEN_SCORE;
        }
    }
    decoder = CrfDecoder(labelCount, weights.readF32("crf.start_transitions", {labelCount}),
                         weights.readF32("crf.end_transitions", {labelCount}), std::move(transitions));
}

void DistilBertCrf::checkInput(const ModelInput& input) const {
    const std::size_t length = input.inputIds.size();
    if (length < 3 || length > maxPositions) {
        throw std::invalid_argument("an input needs 3 to " + std::to_string(maxPositions) + " token ids, not " +
                                    std::to_string(length));
    }
    for (const std::int32_t id : input.inputIds) {
        if (id < 0 || static_cast<std::size_t>(id) >= vocabulary) {
            throw std::invalid_argument("the token id " + std::to_string(id) + " lies outside the vocabulary");
        }
    }
    std::size_t previous = 0;
    for (const std::size_t position : input.wordPositions) {
        if (position <= previous || position > length - 2) {
            throw std::invalid_argument("the word positions must increase strictly within [1, " +
                                        std::to_string(length - 2) + "]");
        }
        previous = position;
    }
}

void DistilBertCrf::encodeBatch(const std::vector<const ModelInput*>& batch, std::vector<float>& emissions) const {
    std::vector<std::size_t> rowStart(batch.size() + 1, 0);
    std::vector<std::size_t> wordStart(batch.size() + 1, 0);
    std::size_t longest = 0;
    for (std::size_t b = 0; b < batch.size(); ++b) {
        rowStart[b + 1]  = rowStart[b] + batch[b]->inputIds.size();
        wordStart[b + 1] = wordStart[b] + batch[b]->wordPositions.size();
        longest = std::max(longest, batch[b]->inputIds.size());
    }
    const std::size_t rows  = rowStart.back();
    const std::size_t words = wordStart.back();
    const std::size_t headSize = dim / heads;

    std::vector<float> hidden(rows * dim);
    for (std::size_t b = 0; b < batch.size(); ++b) {
        for (std::size_t t = 0; t < batch[b]->inputIds.size(); ++t) {
            const float* word     = wordEmbeddings.data() + static_cast<std::size_t>(batch[b]->inputIds[t]) * dim;
            const float* position = positionEmbeddings.data() + t * dim;
            float*       out      = hidden.data() + (rowStart[b] + t) * dim;
            for (std::size_t c = 0; c < dim; ++c) {
                out[c] = word[c] + position[c];
            }
        }
    }
    layerNormRows(hidden.data(), rows, dim, embeddingNormWeight.data(), embeddingNormBias.data(), LAYER_NORM_EPSILON);

    std::vector<std::size_t> everyRow(longest);
    std::iota(everyRow.begin(), everyRow.end(), std::size_t{0});
    std::vector<float> queryKeyValue(rows * 3 * dim);
    std::vector<float> context(rows * dim);
    std::vector<float> attended(rows * dim);
    std::vector<float> expanded(rows * hiddenDim);
    std::vector<float> wordRows(words * dim);
    for (std::size_t index = 0; index < layers.size(); ++index) {
        const EncoderLayer& layer = layers[index];
        // The last layer needs its outputs only at the word positions, which the classifier reads
        const bool last = index + 1 == layers.size();
        layer.queryKeyValue.apply(hidden.data(), rows, queryKeyValue.data());
        for (std::size_t b = 0; b < batch.size(); ++b) {
            const ModelInput& input  = *batch[b];
            const float*      source = queryKeyValue.data() + rowStart[b] * 3 * dim;
            if (last) {
                attendIdentifier(source, input.inputIds.size(), heads, headSize, attentionScale, input.wordPositions.data(),
                                 input.wordPositions.size(), context.data() + wordStart[b] * dim);
                for (std::size_t w = 0; w < input.wordPositions.size(); ++w) {
                    std::copy_n(hidden.data() + (rowStart[b] + input.wordPositions[w]) * dim, dim,
                                wordRows.data() + (wordStart[b] + w) * dim);
                }
            } else {
                attendIdentifier(source, input.inputIds.size(), heads, headSize, attentionScale, everyRow.data(),
                                 input.inputIds.size(), context.data() + rowStart[b] * dim);
            }
        }
        const std::size_t active   = last ? words : rows;
        const float*      residual = last ? wordRows.data() : hidden.data();
        layer.attentionOutput.apply(context.data(), active, attended.data());
        for (std::size_t i = 0; i < active * dim; ++i) {
            attended[i] = attended[i] + residual[i];
        }
        layerNormRows(attended.data(), active, dim, layer.attentionNormWeight.data(), layer.attentionNormBias.data(),
                      LAYER_NORM_EPSILON);
        layer.expand.apply(attended.data(), active, expanded.data());
        geluInPlace(expanded.data(), active * hiddenDim);
        layer.contract.apply(expanded.data(), active, hidden.data());
        for (std::size_t i = 0; i < active * dim; ++i) {
            hidden[i] = hidden[i] + attended[i];
        }
        layerNormRows(hidden.data(), active, dim, layer.outputNormWeight.data(), layer.outputNormBias.data(),
                      LAYER_NORM_EPSILON);
    }
    emissions.assign(words * labels.size(), 0.0f);
    classifier.apply(hidden.data(), words, emissions.data());
}

std::vector<std::vector<float>> DistilBertCrf::wordEmissions(const std::vector<ModelInput>& inputs) const {
    std::vector<const ModelInput*> batch;
    for (const ModelInput& input : inputs) {
        checkInput(input);
        if (!input.wordPositions.empty()) {
            batch.push_back(&input);
        }
    }
    std::vector<float> emissions;
    if (!batch.empty()) {
        encodeBatch(batch, emissions);
    }
    std::vector<std::vector<float>> result(inputs.size());
    std::size_t offset = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::size_t count = inputs[i].wordPositions.size() * labels.size();
        result[i].assign(emissions.begin() + static_cast<std::ptrdiff_t>(offset),
                         emissions.begin() + static_cast<std::ptrdiff_t>(offset + count));
        offset += count;
    }
    return result;
}

std::vector<std::vector<int>> DistilBertCrf::decodeBatch(const std::vector<ModelInput>& inputs) const {
    const std::vector<std::vector<float>> emissions = wordEmissions(inputs);
    std::vector<std::vector<int>> result;
    result.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        result.push_back(decoder.decode(emissions[i].data(), inputs[i].wordPositions.size()));
    }
    return result;
}

std::vector<int> DistilBertCrf::decode(const std::vector<std::int32_t>& inputIds,
                                       const std::vector<std::size_t>& wordPositions) const {
    return decodeBatch({ModelInput{inputIds, wordPositions}}).front();
}

}
