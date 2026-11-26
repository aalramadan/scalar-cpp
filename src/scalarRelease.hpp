// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file scalarRelease.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// The constants of SCALAR v3.0.0-rc.3 that the port reads, under the camelCase form of their Python names.
// tools/update_release.py writes this file from a checkout of the release; edit the script, never the file.

#ifndef SCALAR_RELEASE_HPP
#define SCALAR_RELEASE_HPP

#include <string_view>
#include <utility>

namespace scalar::release {

// DistilBertCRFForTokenClassification.FORBIDDEN_TRANSITIONS of scalar_tagger/lm_based_tagger/distilbert_crf.py
inline constexpr std::pair<std::string_view, std::string_view> forbiddenTransitions[] = {
    {"N", "N"}, {"N", "NPL"}, {"NPL", "NPL"}, {"N", "PRE"}, {"NM", "PRE"}, {"NPL", "PRE"}, {"V", "PRE"},
    {"VM", "PRE"}, {"P", "PRE"}, {"D", "PRE"}, {"DT", "PRE"}, {"CJ", "PRE"}, {"PRE", "VM"}, {"DT", "D"},
    {"CJ", "CJ"},
};

// CONTEXT_MAP of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py
inline constexpr std::pair<std::string_view, std::string_view> contextMap[] = {
    {"FUNCTION", "@func"}, {"PARAMETER", "@param"}, {"ATTRIBUTE", "@attr"}, {"DECLARATION", "@decl"},
    {"CLASS", "@class"},
};

// AVAILABLE_FEATURES of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py
inline constexpr std::string_view availableFeatures[] = {
    "context", "hungarian", "cvr", "digit", "digit_connector", "plural_suffix", "type", "type_overlap",
    "language", "sys_sim", "lead_shape", "lead_word",
};

// DEFAULT_FEATURES of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py
inline constexpr std::string_view defaultFeatures[] = {
    "context", "hungarian", "cvr", "digit", "digit_connector", "plural_suffix", "type", "type_overlap",
    "language", "sys_sim", "lead_shape",
};

// INTERNAL_FEATURES of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py
inline constexpr std::string_view internalFeatures[] = {
    "hungarian_legacy",
};

// FEATURE_ORDER of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py
inline constexpr std::string_view featureOrder[] = {
    "context", "hungarian", "hungarian_legacy", "cvr", "digit", "digit_connector", "plural_suffix", "type",
    "type_overlap", "language", "sys_sim", "lead_shape", "lead_word",
};

// LEGACY_FEATURES of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py
inline constexpr std::string_view legacyFeatures[] = {
    "context", "hungarian_legacy", "cvr", "digit",
};

// TYPE_BOOL_TOKENS of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py, sorted
inline constexpr std::string_view typeBoolTokens[] = {
    "bool", "boolean", "boolean_t", "glboolean",
};

// TYPE_VOID_TOKENS of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py, sorted
inline constexpr std::string_view typeVoidTokens[] = {
    "void",
};

// TYPE_INT_TOKENS of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py, sorted
inline constexpr std::string_view typeIntTokens[] = {
    "byte", "dword", "i16", "i32", "i64", "i8", "int", "int16_t", "int32_t", "int64_t", "int8_t", "integer",
    "isize", "long", "ptrdiff_t", "qword", "short", "size_t", "ssize_t", "u16", "u32", "u64", "u8", "uint16_t",
    "uint32_t", "uint64_t", "uint8_t", "usize", "word",
};

// TYPE_FLOAT_TOKENS of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py, sorted
inline constexpr std::string_view typeFloatTokens[] = {
    "decimal", "double", "float", "number", "numeric", "real",
};

// TYPE_CHAR_TOKENS of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py, sorted
inline constexpr std::string_view typeCharTokens[] = {
    "char", "char16_t", "char32_t", "rune", "tchar", "wchar",
};

// TYPE_STRING_TOKENS of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py, sorted
inline constexpr std::string_view typeStringTokens[] = {
    "cstring", "str", "string", "text", "utf16", "utf32", "utf8",
};

// TYPE_CONTAINER_TOKENS of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py, sorted
inline constexpr std::string_view typeContainerTokens[] = {
    "array", "buffer", "collection", "deque", "iterable", "iterator", "list", "map", "queue", "set", "slice",
    "stack", "stream", "vector",
};

// TYPE_FUNCTION_TOKENS of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py, sorted
inline constexpr std::string_view typeFunctionTokens[] = {
    "callback", "consumer", "func", "function", "lambda", "predicate", "runnable", "supplier",
};

// TYPE_QUALIFIER_TOKENS of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py, sorted
inline constexpr std::string_view typeQualifierTokens[] = {
    "class", "const", "enum", "signed", "static", "struct", "union", "unsigned", "volatile",
};

// NON_PLURAL_S_SUFFIX_STOPLIST of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py, sorted
inline constexpr std::string_view nonPluralSSuffixStoplist[] = {
    "alias", "bias", "canvas", "class", "glass", "has", "http", "https", "is", "minus", "pass", "plus",
    "status", "this", "was",
};

} // namespace scalar::release

#endif
