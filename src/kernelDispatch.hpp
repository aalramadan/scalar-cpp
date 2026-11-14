// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file kernelDispatch.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Compiles each hot kernel for AVX2 with FMA and for the default target, inlines its helpers, and
// computes the fused multiply-add of float bit for bit on targets without an FMA instruction.

#ifndef SCALAR_KERNEL_DISPATCH_HPP
#define SCALAR_KERNEL_DISPATCH_HPP

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

// The kernels use std::fma where torch does. On x86-64 ELF targets the loader picks the x86-64-v3 clone on
// CPUs with AVX2 and FMA, and SCALAR_PORTABLE_KERNELS builds the default clone alone, to test other CPUs.
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__ELF__) && !defined(SCALAR_PORTABLE_KERNELS)
#define SCALAR_HOT_KERNEL __attribute__((target_clones("arch=x86-64-v3", "default")))
#else
#define SCALAR_HOT_KERNEL
#endif

// Helpers of a hot kernel must be inlined into each clone to be compiled for its target. A rare path
// stays out of line, therefore the compiler lays out the common path without jumps.
#if defined(__GNUC__) || defined(__clang__)
#define SCALAR_INLINE inline __attribute__((always_inline))
#define SCALAR_COLD __attribute__((noinline, cold)) inline
#elif defined(_MSC_VER)
#define SCALAR_INLINE inline
#define SCALAR_COLD __declspec(noinline) inline
#else
#define SCALAR_INLINE inline
#define SCALAR_COLD inline
#endif

// The target of the compiler fuses a multiply-add of float in one instruction, which std::fma becomes
// and which the compilers vectorize, as on Arm64 and on x86-64 compiled for FMA, such as x86-64-v3
#if defined(__FP_FAST_FMAF) || defined(__FMA__) || defined(__ARM_FEATURE_FMA) || defined(_M_ARM64) || \
    (defined(_MSC_VER) && defined(__AVX2__))
#define SCALAR_NATIVE_FMA
#endif

// Elsewhere std::fma calls the C library for each value, and the emulation below takes a quarter of
// that time. The emulation needs double arithmetic without excess precision and without reassociation.
#if !defined(SCALAR_NATIVE_FMA) && !defined(__FAST_MATH__) && !defined(_M_FP_FAST) && \
    ((defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD == 0) || defined(_M_X64))
#define SCALAR_EMULATED_FMA
#endif

namespace scalar {

// The fused multiply-add of a double sum that may lie on a rounding boundary of float. The sum rounded
// to odd in double rounds to the float nearest the exact sum, since double carries more than two bits
// beyond float (Boldo and Melquiond, IEEE Transactions on Computers, 2008).
SCALAR_COLD float fusedMultiplyAddOnBoundary(double product, double addend, double sum) {
    // The exact error of the rounded sum, by the TwoSum algorithm of Knuth
    const double addendPart = sum - product;
    const double error      = (product - (sum - addendPart)) + (addend - addendPart);
    std::uint64_t bits;
    std::uint64_t errorBits;
    std::memcpy(&bits, &sum, sizeof bits);
    std::memcpy(&errorBits, &error, sizeof errorBits);
    // An inexact sum is truncated toward zero and gets an odd last bit. The test fails for the NaN
    // error of an infinite sum, which therefore stays infinite.
    if (std::fabs(error) > 0.0) {
        bits = (bits - ((bits ^ errorBits) >> 63)) | 1u;
    }
    double odd;
    std::memcpy(&odd, &bits, sizeof odd);
    return static_cast<float>(odd);
}

// std::fma(a, b, c) for float without an FMA instruction, bit for bit. The product of two floats is exact
// in double, therefore only the sum rounds, to the nearest float except on a rounding boundary of float,
// which ends in at least 28 zero bits as a double, as zeros and infinities do.
SCALAR_INLINE float emulatedFusedMultiplyAdd(float a, float b, float c) {
    const double product = static_cast<double>(a) * static_cast<double>(b);
    const double addend  = static_cast<double>(c);
    const double sum     = product + addend;
    std::uint64_t bits;
    std::memcpy(&bits, &sum, sizeof bits);
    if ((bits & 0x0FFFFFFFu) != 0) {
        return static_cast<float>(sum);
    }
    return fusedMultiplyAddOnBoundary(product, addend, sum);
}

// std::fma(a, b, c) for float, by the emulation where SCALAR_EMULATED_FMA is defined
SCALAR_INLINE float fusedMultiplyAdd(float a, float b, float c) {
#ifdef SCALAR_EMULATED_FMA
    return emulatedFusedMultiplyAdd(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

}

#endif
