// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file encoderKernels.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Reproduces the float32 CPU kernels of torch for the layer norm, the GELU, and the attention.
// Port of moments_utils.h, vec256_float.h (exp_u20), and FlashAttentionKernel.cpp of ATen, torch 2.14.1, BSD-3-Clause.
// Port of the AVX2 gelu_erf kernel of jit_uni_eltwise_injector, oneDNN 3.12.0 commit 80afa71, Apache-2.0.

#include "encoderKernels.hpp"

#include "kernelDispatch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace scalar {

namespace {

constexpr std::size_t LANES       = 8;   // floats per AVX2 vector, the lanes of the torch reductions
constexpr std::size_t CHUNK       = 16;  // vectors per Welford chunk of RowwiseMoments
constexpr std::size_t CHAIN_DEPTH = 256; // terms per FMA chain of the MKL sgemm inside the attention
constexpr std::size_t MAX_DEPTH   = 64;  // levels of the Welford cascade, enough for any row length

SCALAR_INLINE float floatFromBits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

SCALAR_INLINE std::uint32_t bitsOfFloat(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

// 2^exponent for exponent in [-127, 127]. The bit pattern of 2^-127 is +0: the exponent that the clamp
// at ln(FLT_MIN) gives, therefore the zero that torch and oneDNN blend in below that bound comes out here too
SCALAR_INLINE float powerOfTwo(float exponent) {
    return floatFromBits(static_cast<std::uint32_t>(static_cast<std::int32_t>(exponent) + 127) << 23);
}

// AddMomentsVec of RowwiseMoments: merges the moments of countAdd values per lane into the stack entry
SCALAR_INLINE void addMomentsLanes(std::int64_t countAdd, const float* meanAdd, const float* m2Add, std::int64_t& count,
                                   float* mean, float* m2) {
    const std::int64_t total = count + countAdd;
    const float c = total == 0 ? 0.0f : static_cast<float>(countAdd) / static_cast<float>(total);
    const float countValue = static_cast<float>(count);
    for (std::size_t lane = 0; lane < LANES; ++lane) {
        const float delta      = meanAdd[lane] - mean[lane];
        const float m2Sum      = m2[lane] + m2Add[lane];
        const float cDelta     = c * delta;
        const float countDelta = delta * countValue;
        mean[lane] = mean[lane] + cDelta;
        m2[lane]   = std::fma(countDelta, cDelta, m2Sum);
    }
    count = total;
}

// AddMoments of RowwiseMoments, with the two multiply-adds that GCC fuses in the AVX2 build of torch
SCALAR_INLINE void addMoments(std::int64_t countAdd, float meanAdd, float m2Add, std::int64_t& count, float& mean, float& m2) {
    const std::int64_t total = count + countAdd;
    const float c     = total == 0 ? 0.0f : static_cast<float>(countAdd) / static_cast<float>(total);
    const float delta = meanAdd - mean;
    mean = std::fma(c, delta, mean);
    m2   = m2 + std::fma((delta * delta) * c, static_cast<float>(count), m2Add);
    count = total;
}

SCALAR_INLINE int ceilLog2(std::int64_t value) {
    int depth = 0;
    while ((std::int64_t{1} << depth) < value) {
        ++depth;
    }
    return depth;
}

// Mean and population variance of one row, by RowwiseMomentsImpl for float on AVX2
SCALAR_INLINE void rowMoments(const float* row, std::size_t width, float& mean, float& variance) {
    const auto vectors = static_cast<std::int64_t>(width / LANES);
    const std::int64_t chunks = (vectors + static_cast<std::int64_t>(CHUNK) - 1) / static_cast<std::int64_t>(CHUNK);
    const int depth = ceilLog2(chunks);
    std::int64_t stackCount[MAX_DEPTH] = {};
    float stackMean[MAX_DEPTH * LANES] = {};
    float stackM2[MAX_DEPTH * LANES] = {};

    for (std::int64_t chunk = 0; chunk < chunks; ++chunk) {
        const std::int64_t count = std::min(static_cast<std::int64_t>(CHUNK), vectors - chunk * static_cast<std::int64_t>(CHUNK));
        const float* chunkStart = row + static_cast<std::size_t>(chunk) * CHUNK * LANES;
        float chunkMean[LANES] = {};
        float chunkM2[LANES] = {};
        for (std::int64_t j = 0; j < count; ++j) {
            const float inverse = 1.0f / static_cast<float>(j + 1);
            for (std::size_t lane = 0; lane < LANES; ++lane) {
                const float x     = chunkStart[static_cast<std::size_t>(j) * LANES + lane];
                const float delta = x - chunkMean[lane];
                chunkMean[lane] = std::fma(inverse, delta, chunkMean[lane]);
                chunkM2[lane]   = std::fma(delta, x - chunkMean[lane], chunkM2[lane]);
            }
        }
        addMomentsLanes(count, chunkMean, chunkM2, stackCount[0], stackMean, stackM2);
        std::int64_t mask = chunk + 1;
        for (int level = 1; level < depth && (mask & 1) == 0; ++level) {
            const auto below = static_cast<std::size_t>(level - 1);
            const auto above = static_cast<std::size_t>(level);
            addMomentsLanes(stackCount[below], stackMean + below * LANES, stackM2 + below * LANES, stackCount[above],
                            stackMean + above * LANES, stackM2 + above * LANES);
            stackCount[below] = 0;
            std::fill_n(stackMean + below * LANES, LANES, 0.0f);
            std::fill_n(stackM2 + below * LANES, LANES, 0.0f);
            mask >>= 1;
        }
    }
    for (int level = 1; level < depth; ++level) {
        const auto index = static_cast<std::size_t>(level);
        addMomentsLanes(stackCount[index], stackMean + index * LANES, stackM2 + index * LANES, stackCount[0], stackMean,
                        stackM2);
    }

    std::int64_t count = 0;
    float m1 = 0.0f;
    float m2 = 0.0f;
    for (std::size_t i = static_cast<std::size_t>(vectors) * LANES; i < width; ++i) {
        const float x     = row[i];
        const float delta = x - m1;
        ++count;
        m1 += delta / static_cast<float>(count);
        // torch rounds the product and the sum apart here
        m2 = m2 + delta * (x - m1);
    }
    for (std::size_t lane = 0; lane < LANES; ++lane) {
        addMoments(vectors, stackMean[lane], stackM2[lane], count, m1, m2);
    }
    mean     = m1;
    variance = m2 / static_cast<float>(width);
}

// The clamp of _mm256_min_ps and then _mm256_max_ps, which give the bound for a NaN, therefore a NaN
// never reaches the conversion to an integer in powerOfTwo
SCALAR_INLINE float clampLikeAvx(float x, float lowest, float highest) {
    const float bounded = x < highest ? x : highest;
    return bounded > lowest ? bounded : lowest;
}

// exp_u20 of the AVX2 Vectorized<float> of torch, with its polynomial of degree 5
SCALAR_INLINE float expU20(float x) {
    const float lowest  = floatFromBits(0xc2aeac50u);
    const float highest = floatFromBits(0x42b17218u);
    const float source  = clampLikeAvx(x, lowest, highest);
    const float whole   = std::floor(std::fma(source, floatFromBits(0x3fb8aa3bu), 0.5f));
    const float rest    = std::fma(-whole, floatFromBits(0x3f317218u), source);
    float result = std::fma(rest, 0.00828929059f, 0.0418978221f);
    result = std::fma(rest, result, 0.166676521f);
    result = std::fma(rest, result, 0.499991506f);
    result = std::fma(rest, result, 0.999999701f);
    result = std::fma(rest, result, 1.0f);
    return (result * powerOfTwo(whole - 1.0f)) * 2.0f;
}

// exp of the AVX2 eltwise injector of oneDNN, which rounds the product and the sum of x log2(e) + 0.5 apart
SCALAR_INLINE float expOneDnn(float x) {
    const float lowest  = floatFromBits(0xc2aeac50u);
    const float highest = floatFromBits(0x42b17218u);
    const float source  = clampLikeAvx(x, lowest, highest);
    const float whole   = std::floor(source * floatFromBits(0x3fb8aa3bu) + 0.5f);
    const float rest    = std::fma(-whole, floatFromBits(0x3f317218u), source);
    float result = floatFromBits(0x3c07cfceu);
    result = std::fma(result, rest, floatFromBits(0x3d2b9d0du));
    result = std::fma(result, rest, floatFromBits(0x3e2aad40u));
    result = std::fma(result, rest, floatFromBits(0x3efffee3u));
    result = std::fma(result, rest, floatFromBits(0x3f7ffffbu));
    result = std::fma(result, rest, 1.0f);
    return (result * powerOfTwo(whole - 1.0f)) * 2.0f;
}

SCALAR_INLINE float geluOneDnn(float x) {
    const float s = x * floatFromBits(0x3f3504f3u);
    const float t = 1.0f / std::fma(floatFromBits(0x3ea7ba05u), std::fabs(s), 1.0f);
    const float weighted = -expOneDnn(-(s * s)) * t;
    float polynomial = floatFromBits(0x3f87dc22u);
    polynomial = std::fma(polynomial, t, floatFromBits(0xbfba00e3u));
    polynomial = std::fma(polynomial, t, floatFromBits(0x3fb5f0e3u));
    polynomial = std::fma(polynomial, t, floatFromBits(0xbe91a98eu));
    polynomial = std::fma(polynomial, t, floatFromBits(0x3e827906u));
    // oneDNN flips the sign bit of erf(|s|) with the sign bit of x
    const float erf  = floatFromBits(bitsOfFloat(std::fma(weighted, polynomial, 1.0f)) ^ (bitsOfFloat(x) & 0x80000000u));
    const float half = x * 0.5f;
    return std::fma(erf, half, half);
}

}

SCALAR_HOT_KERNEL
void layerNormRows(float* rows, std::size_t rowCount, std::size_t width, const float* gamma, const float* beta,
                   float epsilon) {
    for (std::size_t r = 0; r < rowCount; ++r) {
        float* row = rows + r * width;
        float mean = 0.0f;
        float variance = 0.0f;
        rowMoments(row, width, mean, variance);
        const float inverseDeviation = 1.0f / std::sqrt(variance + epsilon);
        for (std::size_t i = 0; i < width; ++i) {
            row[i] = std::fma((row[i] - mean) * inverseDeviation, gamma[i], beta[i]);
        }
    }
}

SCALAR_HOT_KERNEL
void geluInPlace(float* values, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = geluOneDnn(values[i]);
    }
}

SCALAR_HOT_KERNEL
void attendIdentifier(const float* qkv, std::size_t length, std::size_t heads, std::size_t headSize, float scale,
                      const std::size_t* queries, std::size_t queryCount, float* output) {
    const std::size_t width  = heads * headSize;
    const std::size_t stride = 3 * width;
    const std::size_t groups = length / LANES;
    std::vector<float> keys(headSize * length);   // keys of one head, transposed: keys[d * length + j]
    std::vector<float> weights(length);
    std::vector<float> scoreChain(length);
    std::vector<float> sums(headSize);
    std::vector<float> chain(headSize);
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t j = 0; j < length; ++j) {
            const float* key = qkv + j * stride + width + head * headSize;
            for (std::size_t d = 0; d < headSize; ++d) {
                keys[d * length + j] = key[d];
            }
        }
        for (std::size_t q = 0; q < queryCount; ++q) {
            const float* query = qkv + queries[q] * stride + head * headSize;
            // Both products of the attention follow the chains of the MKL sgemm, as in the linear layers
            std::fill(weights.begin(), weights.end(), 0.0f);
            for (std::size_t first = 0; first < headSize; first += CHAIN_DEPTH) {
                const std::size_t last = std::min(headSize, first + CHAIN_DEPTH);
                std::fill(scoreChain.begin(), scoreChain.end(), 0.0f);
                for (std::size_t d = first; d < last; ++d) {
                    const float  value  = query[d];
                    const float* column = keys.data() + d * length;
                    for (std::size_t j = 0; j < length; ++j) {
                        scoreChain[j] = std::fma(value, column[j], scoreChain[j]);
                    }
                }
                for (std::size_t j = 0; j < length; ++j) {
                    weights[j] += scoreChain[j];
                }
            }
            float highest = -std::numeric_limits<float>::infinity();
            for (std::size_t j = 0; j < length; ++j) {
                weights[j] *= scale;
                highest = std::max(highest, weights[j]);
            }
            // The vector lanes of exp_u20 first, then the horizontal sum of torch for AVX2
            float lanes[LANES] = {};
            for (std::size_t g = 0; g < groups; ++g) {
                for (std::size_t lane = 0; lane < LANES; ++lane) {
                    const float weight = expU20(weights[g * LANES + lane] - highest);
                    weights[g * LANES + lane] = weight;
                    lanes[lane] += weight;
                }
            }
            float total = ((lanes[0] + lanes[4]) + (lanes[2] + lanes[6])) + ((lanes[1] + lanes[5]) + (lanes[3] + lanes[7]));
            // The scalar tail of torch rounds the double exp to float, which expf does not always match
            for (std::size_t j = groups * LANES; j < length; ++j) {
                const float weight = static_cast<float>(std::exp(static_cast<double>(weights[j] - highest)));
                weights[j] = weight;
                total += weight;
            }
            std::fill(sums.begin(), sums.end(), 0.0f);
            for (std::size_t first = 0; first < length; first += CHAIN_DEPTH) {
                const std::size_t last = std::min(length, first + CHAIN_DEPTH);
                std::fill(chain.begin(), chain.end(), 0.0f);
                for (std::size_t j = first; j < last; ++j) {
                    const float  weight = weights[j];
                    const float* value  = qkv + j * stride + 2 * width + head * headSize;
                    for (std::size_t d = 0; d < headSize; ++d) {
                        chain[d] = std::fma(weight, value[d], chain[d]);
                    }
                }
                for (std::size_t d = 0; d < headSize; ++d) {
                    sums[d] += chain[d];
                }
            }
            const float reciprocal = 1.0f / total;
            float* out = output + q * width + head * headSize;
            for (std::size_t d = 0; d < headSize; ++d) {
                out[d] = sums[d] * reciprocal;
            }
        }
    }
}

}
