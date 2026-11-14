// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file linearLayer.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Multiplies rows of activations by packed weight panels in register tiles of 16 outputs and 4 or 6 rows.

#include "linearLayer.hpp"

#include "kernelDispatch.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__clang__)
#define SCALAR_UNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define SCALAR_UNROLL _Pragma("GCC unroll 8")
#endif

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(SCALAR_PORTABLE_KERNELS)
#define SCALAR_AVX2_TILES
#include <immintrin.h>
#endif

// The vectors of GCC and Clang hold the chains of a tile in registers, as the AVX2 tiles do
#if (defined(__GNUC__) || defined(__clang__)) && defined(SCALAR_NATIVE_FMA)
#define SCALAR_LANE_TILES
#endif

namespace scalar {

namespace {

constexpr std::size_t PANEL_WIDTH = 16;  // outputs per weight panel, two vectors of 8 floats
constexpr std::size_t CHAIN_DEPTH = 256; // inputs per FMA chain before MKL adds the chain to the output
constexpr std::size_t ROW_BLOCK   = 192; // rows whose inputs of one chain, 192 KB, stay in the L2 cache

// The tiles of the CPUs without AVX2: plain floats with the exact multiply-add of kernelDispatch.hpp, or
// vectors of GCC and Clang where the target fuses in hardware
#ifndef SCALAR_LANE_TILES
// Adds to Rows output rows of one panel the FMA chain over the inputs [first, last); panel holds the
// weights of these inputs only
struct ScalarTiles {
    static constexpr std::size_t ROWS = 6; // rows per tile, as in the AVX2 sgemm kernel of MKL

    template <std::size_t Rows>
    static SCALAR_INLINE void addChain(const float* input, std::size_t inputs, const float* panel, std::size_t first,
                                       std::size_t last, std::size_t columns, float* output, std::size_t outputs) {
        float chain[Rows][PANEL_WIDTH] = {};
        for (std::size_t k = first; k < last; ++k) {
            const float* weight = panel + (k - first) * PANEL_WIDTH;
            for (std::size_t row = 0; row < Rows; ++row) {
                const float value = input[row * inputs + k];
                for (std::size_t column = 0; column < PANEL_WIDTH; ++column) {
                    chain[row][column] = fusedMultiplyAdd(value, weight[column], chain[row][column]);
                }
            }
        }
        for (std::size_t row = 0; row < Rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                output[row * outputs + column] += chain[row][column];
            }
        }
    }
};
using PlainTiles = ScalarTiles;
#else
// Eight floats in a vector of GCC and Clang: one AVX register, or two NEON registers on Arm64
typedef float FloatLanes __attribute__((vector_size(8 * sizeof(float))));

// The same sums as ScalarTiles in the vectors of GCC and Clang. The compilers turn the std::fma of the
// eight lanes into one vector FMA, such as fmla on Arm64, and keep the chains in registers.
struct LaneTiles {
    // Four rows keep the chains, the weights, and the broadcast inputs within the 16 vector registers of AVX2
    static constexpr std::size_t ROWS = 4;

    static SCALAR_INLINE void load(FloatLanes& lanes, const float* source) {
        std::memcpy(&lanes, source, sizeof lanes);
    }

    static SCALAR_INLINE void store(float* target, const FloatLanes& lanes) {
        std::memcpy(target, &lanes, sizeof lanes);
    }

    static SCALAR_INLINE void multiplyAdd(FloatLanes& chain, float value, const FloatLanes& weights) {
        for (int lane = 0; lane < 8; ++lane) {
            chain[lane] = std::fma(value, weights[lane], chain[lane]);
        }
    }

    template <std::size_t Rows>
    static SCALAR_INLINE void addChain(const float* input, std::size_t inputs, const float* panel, std::size_t first,
                                       std::size_t last, std::size_t columns, float* output, std::size_t outputs) {
        FloatLanes chainLow[Rows];
        FloatLanes chainHigh[Rows];
        SCALAR_UNROLL
        for (std::size_t row = 0; row < Rows; ++row) {
            chainLow[row]  = FloatLanes{};
            chainHigh[row] = FloatLanes{};
        }
        for (std::size_t k = first; k < last; ++k) {
            FloatLanes weightLow;
            FloatLanes weightHigh;
            load(weightLow, panel + (k - first) * PANEL_WIDTH);
            load(weightHigh, panel + (k - first) * PANEL_WIDTH + 8);
            SCALAR_UNROLL
            for (std::size_t row = 0; row < Rows; ++row) {
                const float value = input[row * inputs + k];
                multiplyAdd(chainLow[row], value, weightLow);
                multiplyAdd(chainHigh[row], value, weightHigh);
            }
        }
        SCALAR_UNROLL
        for (std::size_t row = 0; row < Rows; ++row) {
            float* target = output + row * outputs;
            if (columns == PANEL_WIDTH) {
                FloatLanes low;
                FloatLanes high;
                load(low, target);
                load(high, target + 8);
                store(target, low + chainLow[row]);
                store(target + 8, high + chainHigh[row]);
            } else {
                float sums[PANEL_WIDTH];
                store(sums, chainLow[row]);
                store(sums + 8, chainHigh[row]);
                for (std::size_t column = 0; column < columns; ++column) {
                    target[column] += sums[column];
                }
            }
        }
    }
};
using PlainTiles = LaneTiles;
#endif

#ifdef SCALAR_AVX2_TILES
// The same sums as ScalarTiles, with the chains held in AVX2 registers
struct Avx2Tiles {
    static constexpr std::size_t ROWS = 6; // rows per register tile, as in the AVX2 sgemm kernel of MKL

    template <std::size_t Rows>
    static inline __attribute__((target("avx2,fma"))) void
    addChain(const float* input, std::size_t inputs, const float* panel, std::size_t first, std::size_t last,
             std::size_t columns, float* output, std::size_t outputs) {
        // Unrolled row loops let the compiler keep the chains in registers instead of storing them at each step
        __m256 chainLow[Rows];
        __m256 chainHigh[Rows];
        SCALAR_UNROLL
        for (std::size_t row = 0; row < Rows; ++row) {
            chainLow[row]  = _mm256_setzero_ps();
            chainHigh[row] = _mm256_setzero_ps();
        }
        for (std::size_t k = first; k < last; ++k) {
            const __m256 weightLow  = _mm256_loadu_ps(panel + (k - first) * PANEL_WIDTH);
            const __m256 weightHigh = _mm256_loadu_ps(panel + (k - first) * PANEL_WIDTH + 8);
            SCALAR_UNROLL
            for (std::size_t row = 0; row < Rows; ++row) {
                const __m256 value = _mm256_broadcast_ss(input + row * inputs + k);
                chainLow[row]  = _mm256_fmadd_ps(value, weightLow, chainLow[row]);
                chainHigh[row] = _mm256_fmadd_ps(value, weightHigh, chainHigh[row]);
            }
        }
        SCALAR_UNROLL
        for (std::size_t row = 0; row < Rows; ++row) {
            float* target = output + row * outputs;
            if (columns == PANEL_WIDTH) {
                _mm256_storeu_ps(target, _mm256_add_ps(_mm256_loadu_ps(target), chainLow[row]));
                _mm256_storeu_ps(target + 8, _mm256_add_ps(_mm256_loadu_ps(target + 8), chainHigh[row]));
            } else {
                float sums[PANEL_WIDTH];
                _mm256_storeu_ps(sums, chainLow[row]);
                _mm256_storeu_ps(sums + 8, chainHigh[row]);
                for (std::size_t column = 0; column < columns; ++column) {
                    target[column] += sums[column];
                }
            }
        }
    }
};
#endif

// Adds the chains of the rowCount rows that remain below the full tiles of a block, from 1 to Rows rows
template <typename Tiles, std::size_t Rows>
SCALAR_INLINE void addLastRows(std::size_t rowCount, const float* input, std::size_t inputs, const float* panel,
                               std::size_t first, std::size_t last, std::size_t columns, float* output, std::size_t outputs) {
    if constexpr (Rows > 0) {
        if (rowCount == Rows) {
            Tiles::template addChain<Rows>(input, inputs, panel, first, last, columns, output, outputs);
        } else {
            addLastRows<Tiles, Rows - 1>(rowCount, input, inputs, panel, first, last, columns, output, outputs);
        }
    }
}

// output = bias, then output += chain for each run of CHAIN_DEPTH inputs, the order of the MKL sgemm.
// The runs form the outer loop, therefore the weights pass through the caches once per block of rows.
template <typename Tiles>
SCALAR_INLINE void multiplyRowsWith(const float* input, std::size_t rows, std::size_t inputs, const float* panels,
                                    const float* bias, std::size_t outputs, float* output) {
    for (std::size_t row = 0; row < rows; ++row) {
        std::memcpy(output + row * outputs, bias, outputs * sizeof(float));
    }
    const std::size_t panelCount = (outputs + PANEL_WIDTH - 1) / PANEL_WIDTH;
    for (std::size_t first = 0; first < inputs; first += CHAIN_DEPTH) {
        const std::size_t last = std::min(inputs, first + CHAIN_DEPTH);
        for (std::size_t blockStart = 0; blockStart < rows; blockStart += ROW_BLOCK) {
            const std::size_t blockEnd = std::min(rows, blockStart + ROW_BLOCK);
            for (std::size_t p = 0; p < panelCount; ++p) {
                const float*      panel    = panels + (first * panelCount + p * (last - first)) * PANEL_WIDTH;
                const std::size_t columns  = std::min(PANEL_WIDTH, outputs - p * PANEL_WIDTH);
                float*            panelOut = output + p * PANEL_WIDTH;
                std::size_t row = blockStart;
                for (; row + Tiles::ROWS <= blockEnd; row += Tiles::ROWS) {
                    Tiles::template addChain<Tiles::ROWS>(input + row * inputs, inputs, panel, first, last, columns,
                                                          panelOut + row * outputs, outputs);
                }
                addLastRows<Tiles, Tiles::ROWS - 1>(blockEnd - row, input + row * inputs, inputs, panel, first, last,
                                                     columns, panelOut + row * outputs, outputs);
            }
        }
    }
}

#ifdef SCALAR_AVX2_TILES
__attribute__((target("avx2,fma"))) void multiplyRowsAvx2(const float* input, std::size_t rows, std::size_t inputs,
                                                          const float* panels, const float* bias, std::size_t outputs,
                                                          float* output) {
    multiplyRowsWith<Avx2Tiles>(input, rows, inputs, panels, bias, outputs, output);
}
#endif

void multiplyRows(const float* input, std::size_t rows, std::size_t inputs, const float* panels, const float* bias,
                  std::size_t outputs, float* output) {
#ifdef SCALAR_AVX2_TILES
    static const bool hasAvx2 = [] {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    if (hasAvx2) {
        multiplyRowsAvx2(input, rows, inputs, panels, bias, outputs, output);
        return;
    }
#endif
    multiplyRowsWith<PlainTiles>(input, rows, inputs, panels, bias, outputs, output);
}

}

LinearLayer::LinearLayer(const std::vector<float>& weights, const std::vector<float>& bias, std::size_t inputCount)
    : inputs(inputCount), outputs(bias.size()), outputBias(bias) {
    if (inputs == 0 || outputs == 0 || weights.size() != inputs * outputs) {
        throw std::invalid_argument("a linear layer needs outputs × inputs weights and one bias per output");
    }
    const std::size_t panelCount = (outputs + PANEL_WIDTH - 1) / PANEL_WIDTH;
    panels.assign(panelCount * inputs * PANEL_WIDTH, 0.0f);
    // Chain by chain, the panels lie one after another, therefore each chain reads its weights in one sweep
    for (std::size_t out = 0; out < outputs; ++out) {
        const std::size_t p      = out / PANEL_WIDTH;
        const std::size_t column = out % PANEL_WIDTH;
        for (std::size_t k = 0; k < inputs; ++k) {
            const std::size_t first = k - k % CHAIN_DEPTH;
            const std::size_t depth = std::min(CHAIN_DEPTH, inputs - first);
            panels[(first * panelCount + p * depth + (k - first)) * PANEL_WIDTH + column] = weights[out * inputs + k];
        }
    }
}

void LinearLayer::apply(const float* input, std::size_t rows, float* output) const {
    multiplyRows(input, rows, inputs, panels.data(), outputBias.data(), outputs, output);
}

}
