// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file englishStemmer.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// English Snowball stemmer that the Ronin splitter of SCALAR calls.
// Port of nltk/stem/snowball.py (EnglishStemmer) of NLTK release 3.10.3, Apache-2.0.

#ifndef SCALAR_ENGLISH_STEMMER_HPP
#define SCALAR_ENGLISH_STEMMER_HPP

#include <string>

namespace scalar {

// Stem of SnowballStemmer('english').stem(word) of NLTK 3.10.3, without stopwords.
// Equal to NLTK for 7-bit ASCII; a byte of 0x80 or more is kept as an uncased consonant.
std::string stemEnglish(const std::string& word);

}

#endif
