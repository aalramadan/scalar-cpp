// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file sequenceMatcher.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The similarity ratio of difflib.SequenceMatcher, which the sys_sim feature of SCALAR uses
// Port of SequenceMatcher of Lib/difflib.py, CPython v3.14.10 (PSF-2.0)

#include "sequenceMatcher.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace scalar {

namespace {

struct Match {
    std::size_t a    = 0;
    std::size_t b    = 0;
    std::size_t size = 0;
};

// SequenceMatcher with isjunk=None. Without a junk function the set of junk elements is empty,
//   therefore the junk-extension steps of find_longest_match never apply and are left out.
class SequenceMatcher {
public:
    SequenceMatcher(std::string_view first, std::string_view second) : a(first), b(second) {
        for (std::size_t j = 0; j < b.size(); ++j) b2j[static_cast<unsigned char>(b[j])].push_back(j);

        // autojunk: an element of more than 1% of a long b is "popular" and leaves b2j
        if (b.size() >= 200) {
            const std::size_t ntest = b.size() / 100 + 1;
            for (std::vector<std::size_t>& indices : b2j) {
                if (indices.size() > ntest) indices.clear();
            }
        }
    }

    // Sum of the sizes of get_matching_blocks()
    std::size_t matchingCount() const {
        std::vector<std::array<std::size_t, 4>> queue{{0, a.size(), 0, b.size()}};
        std::size_t matches = 0;
        while (!queue.empty()) {
            const auto [alo, ahi, blo, bhi] = queue.back();
            queue.pop_back();
            const Match match = findLongestMatch(alo, ahi, blo, bhi);
            if (match.size == 0) continue;
            matches += match.size;
            if (alo < match.a && blo < match.b) queue.push_back({alo, match.a, blo, match.b});
            if (match.a + match.size < ahi && match.b + match.size < bhi) {
                queue.push_back({match.a + match.size, ahi, match.b + match.size, bhi});
            }
        }
        return matches;
    }

private:
    Match findLongestMatch(std::size_t alo, std::size_t ahi, std::size_t blo, std::size_t bhi) const {
        Match best{alo, blo, 0};

        // Entry j + 1 holds the length of the longest match that ends with a[i - 1] and b[j],
        //   the j2len dictionary of the Python; entry 0 stands for the missing j = -1
        std::vector<std::size_t> j2len(b.size() + 1, 0);
        std::vector<std::size_t> newj2len(b.size() + 1, 0);
        std::vector<std::size_t> touched;
        std::vector<std::size_t> newTouched;
        for (std::size_t i = alo; i < ahi; ++i) {
            for (const std::size_t j : b2j[static_cast<unsigned char>(a[i])]) {
                if (j < blo) continue;
                if (j >= bhi) break;
                const std::size_t k = j2len[j] + 1;
                newj2len[j + 1] = k;
                newTouched.push_back(j + 1);
                if (k > best.size) best = {i - k + 1, j - k + 1, k};
            }
            for (const std::size_t index : touched) j2len[index] = 0;
            std::swap(j2len, newj2len);
            std::swap(touched, newTouched);
            newTouched.clear();
        }

        // Extend the best match by equal elements on each end, popular elements included
        while (best.a > alo && best.b > blo && a[best.a - 1] == b[best.b - 1]) {
            --best.a;
            --best.b;
            ++best.size;
        }
        while (best.a + best.size < ahi && best.b + best.size < bhi && a[best.a + best.size] == b[best.b + best.size]) {
            ++best.size;
        }
        return best;
    }

    std::string_view                         a;
    std::string_view                         b;
    std::array<std::vector<std::size_t>, 256> b2j;
};

} // namespace

double sequenceMatcherRatio(std::string_view a, std::string_view b) {
    const std::size_t length = a.size() + b.size();
    if (length == 0) return 1.0;
    const std::size_t matches = SequenceMatcher(a, b).matchingCount();
    return 2.0 * static_cast<double>(matches) / static_cast<double>(length);
}

} // namespace scalar
