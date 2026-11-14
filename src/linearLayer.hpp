// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file linearLayer.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Applies a fully connected layer with a cache-blocked matrix product in float32.

#ifndef SCALAR_LINEAR_LAYER_HPP
#define SCALAR_LINEAR_LAYER_HPP

#include <cstddef>
#include <vector>

namespace scalar {

// The layer y = x W^T + b of torch.nn.Linear. Each output starts from its bias and adds the FMA chain of
// each run of 256 inputs, in the order of the single-thread MKL sgemm behind torch on AVX2, therefore
// its value never depends on the number of rows or on the position of its row.
class LinearLayer {
public:
    // An empty layer, which the model replaces with one of its checkpoint
    LinearLayer() = default;

    // weights holds outputs × inputs values in row-major order, as torch stores the weight of
    // nn.Linear, and bias holds one value per output. Throws std::invalid_argument on other sizes.
    LinearLayer(const std::vector<float>& weights, const std::vector<float>& bias, std::size_t inputs);

    // Values per input row and per output row
    std::size_t inputCount() const { return inputs; }
    std::size_t outputCount() const { return outputs; }

    // Writes rows × outputCount() values to output for rows × inputCount() values of input.
    void apply(const float* input, std::size_t rows, float* output) const;

private:
    std::size_t        inputs  = 0;
    std::size_t        outputs = 0;
    std::vector<float> panels;      // per chain of 256 inputs, the weights of each group of 16 outputs, input by input
    std::vector<float> outputBias;
};

}

#endif
