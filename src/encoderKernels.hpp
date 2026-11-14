// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file encoderKernels.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Computes the layer norm, the GELU, and the self-attention of the DistilBERT encoder in float32.
// Port of moments_utils.h, vec256_float.h (exp_u20), and FlashAttentionKernel.cpp of ATen, torch 2.14.1, BSD-3-Clause.
// Port of the AVX2 gelu_erf kernel of jit_uni_eltwise_injector, oneDNN 3.12.0 commit 80afa71, Apache-2.0.

#ifndef SCALAR_ENCODER_KERNELS_HPP
#define SCALAR_ENCODER_KERNELS_HPP

#include <cstddef>

namespace scalar {

// Normalizes rowCount rows of width values in place, as torch.nn.LayerNorm computes them on a CPU
// with AVX2: the mean and the variance come from the Welford cascade of RowwiseMoments over 8 lanes.
// gamma and beta hold width values.
void layerNormRows(float* rows, std::size_t rowCount, std::size_t width, const float* gamma, const float* beta,
                   float epsilon);

// Replaces each of count values by its GELU with erf, as the AVX2 gelu_erf kernel of oneDNN 3.12
// behind torch.nn.functional.gelu computes it, with the Abramowitz-Stegun erf of that kernel.
void geluInPlace(float* values, std::size_t count);

// Self-attention of one identifier without a mask, as the CPU flash attention of torch computes it in
// float32. Each of the length rows of qkv holds the queries, keys, and values of all heads (3 × heads ×
// headSize values), and output gets one row of heads × headSize values per row named in queries.
void attendIdentifier(const float* qkv, std::size_t length, std::size_t heads, std::size_t headSize, float scale,
                      const std::size_t* queries, std::size_t queryCount, float* output);

}

#endif
