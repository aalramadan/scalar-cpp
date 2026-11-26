// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file featureTokens.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The feature tokens and the position tokens that precede the words of an identifier in the SCALAR model input
// Port of scalar_tagger/lm_based_tagger/distilbert_preprocessing.py of SCANL/scanl_tagger (GPL-3.0)

#include "featureTokens.hpp"

#include "asciiText.hpp"
#include "jsonReader.hpp"
#include "pythonSum.hpp"
#include "scalarRelease.hpp"
#include "sequenceMatcher.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace scalar {

namespace {

using Words  = std::vector<std::string>;
using Tokens = std::vector<std::string>;

// The word sets of distilbert_preprocessing.py, such as TYPE_BOOL_TOKENS
constexpr std::string_view typeBoolWords[] = {"bool", "boolean", "glboolean", "boolean_t"};
constexpr std::string_view typeVoidWords[] = {"void"};
constexpr std::string_view typeIntWords[] = {
    "int", "integer", "byte", "word", "dword", "qword", "short", "long",
    "size_t", "ssize_t", "ptrdiff_t", "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "u8", "u16", "u32", "u64",
    "i8", "i16", "i32", "i64", "usize", "isize"};
constexpr std::string_view typeFloatWords[] = {"float", "double", "decimal", "real", "numeric", "number"};
constexpr std::string_view typeCharWords[] = {"char", "wchar", "tchar", "char16_t", "char32_t", "rune"};
constexpr std::string_view typeStringWords[] = {"string", "str", "text", "utf8", "utf16", "utf32", "cstring"};
constexpr std::string_view typeContainerWords[] = {
    "list", "array", "vector", "map", "set", "queue", "stack", "deque",
    "collection", "iterator", "iterable", "stream", "buffer", "slice"};
constexpr std::string_view typeFunctionWords[] = {
    "func", "function", "callback", "predicate", "consumer", "supplier", "runnable", "lambda"};
constexpr std::string_view typeQualifierWords[] = {
    "const", "volatile", "static", "signed", "unsigned", "struct", "class", "enum", "union"};
constexpr std::string_view nonPluralWords[] = {
    "this", "is", "was", "has", "plus", "minus", "pass", "class", "glass",
    "status", "alias", "bias", "canvas", "http", "https"};

constexpr std::pair<std::string_view, std::string_view> contextTokens[] = {
    {"FUNCTION", "@func"}, {"PARAMETER", "@param"}, {"ATTRIBUTE", "@attr"}, {"DECLARATION", "@decl"}, {"CLASS", "@class"}};

template <std::size_t N>
bool isIn(std::string_view word, const std::string_view (&set)[N]) {
    return std::find(std::begin(set), std::end(set), word) != std::end(set);
}

template <std::size_t N>
bool anyIn(const Words& words, const std::string_view (&set)[N]) {
    return std::any_of(words.begin(), words.end(), [&set](const std::string& word) { return isIn(word, set); });
}

bool endsWith(std::string_view text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool startsWith(std::string_view text, std::string_view prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

// The \w class of the regular expressions, on ASCII
bool isWordChar(char c) {
    return isAsciiLetter(c) || isAsciiDigit(c) || c == '_';
}

// str.replace(old, new) with a non-empty old
std::string replaceAll(std::string_view text, std::string_view from, std::string_view to) {
    std::string replaced;
    std::size_t start = 0;
    for (std::size_t found = text.find(from); found != std::string_view::npos; found = text.find(from, start)) {
        replaced.append(text, start, found - start);
        replaced.append(to);
        start = found + from.size();
    }
    replaced.append(text, start);
    return replaced;
}

// _dedupe_preserve_order
Tokens dedupePreserveOrder(const Tokens& values) {
    Tokens ordered;
    for (const std::string& value : values) {
        if (std::find(ordered.begin(), ordered.end(), value) == ordered.end()) ordered.push_back(value);
    }
    return ordered;
}

// CONTEXT_MAP lookup of the context feature
std::string contextToken(std::string_view context) {
    const std::string key = toUpperAscii(pythonStrip(context));
    for (const auto& [name, token] : contextTokens) {
        if (key == name) return std::string(token);
    }
    return "@unknown";
}

// detect_hungarian_prefix_split
std::string hungarianPrefixSplit(const Words& words) {
    if (words.empty()) return "@hung_none";
    const std::string& first = words.front();
    // re.match(r"^([a-z])[A-Z][a-z]", first)
    if (first.size() >= 3 && isAsciiLower(first[0]) && isAsciiUpper(first[1]) && isAsciiLower(first[2])) {
        return "@hung_" + first.substr(0, 1);
    }
    // re.fullmatch(r"[a-z]", first) and re.match(r"[A-Z][a-z]", second)
    if (words.size() >= 2 && first.size() == 1 && isAsciiLower(first[0]) && words[1].size() >= 2 &&
        isAsciiUpper(words[1][0]) && isAsciiLower(words[1][1])) {
        return "@hung_" + first;
    }
    return "@hung_none";
}

// detect_hungarian_prefix of the first word, re.match(r'^([a-zA-Z]{1,3})[A-Z_]', word);
//   the regular expression backtracks from three letters down to one
std::string hungarianPrefixLegacy(const Words& words) {
    if (words.empty()) return "@hung_none";
    const std::string& word = words.front();
    std::size_t letters = 0;
    while (letters < 3 && letters < word.size() && isAsciiLetter(word[letters])) ++letters;
    for (std::size_t length = letters; length >= 1; --length) {
        if (length < word.size() && (isAsciiUpper(word[length]) || word[length] == '_')) {
            return "@hung_" + toLowerAscii(std::string_view(word).substr(0, length));
        }
    }
    return "@hung_none";
}

// consonant_vowel_ratio_bucket
std::string consonantVowelRatioBucket(const Words& words) {
    std::vector<double> ratios;
    for (const std::string& word : words) {
        if (!isAlphaText(word)) continue;
        std::size_t vowels     = 0;
        std::size_t consonants = 0;
        for (const char c : toLowerAscii(word)) {
            if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') ++vowels;
            else if (isAsciiLetter(c)) ++consonants;
        }
        ratios.push_back(static_cast<double>(consonants) / (static_cast<double>(vowels) + 1e-5));
    }
    if (ratios.empty()) return "@cvr_none";
    const double average = pythonSum(ratios) / static_cast<double>(ratios.size());
    if (average < 1.5) return "@cvr_low";
    if (average < 3.0) return "@cvr_mid";
    return "@cvr_high";
}

// detect_digit_feature
std::string digitFeature(const Words& words) {
    const bool hasDigit = std::any_of(words.begin(), words.end(), [](const std::string& word) {
        return std::any_of(word.begin(), word.end(), isAsciiDigit);
    });
    return hasDigit ? "@has_digit" : "@no_digit";
}

// get_digit_connector_feature_tokens
Tokens digitConnectorFeatureTokens(const Words& words) {
    Tokens tokens;
    for (std::size_t index = 0; index < words.size(); ++index) {
        if (toLowerAscii(pythonStrip(words[index])) != "2") continue;
        tokens.emplace_back("@digit_connector_2");
        if (index == 0) tokens.emplace_back("@digit_connector_head");
        else if (index == words.size() - 1) tokens.emplace_back("@digit_connector_tail");
        else tokens.emplace_back("@digit_connector_mid");

        const bool previousIsAlpha = index > 0 && isAlphaText(words[index - 1]);
        const bool nextIsAlpha     = index + 1 < words.size() && isAlphaText(words[index + 1]);
        if (previousIsAlpha && nextIsAlpha) tokens.emplace_back("@digit_connector_alpha_bridge");
        else if (previousIsAlpha || nextIsAlpha) tokens.emplace_back("@digit_connector_alpha_adjacent");
    }
    if (tokens.empty()) return {"@digit_connector_none"};
    return dedupePreserveOrder(tokens);
}

// _plural_suffix_bucket; an empty result stands for None
std::string pluralSuffixBucket(std::string_view word) {
    const std::string normalized = toLowerAscii(pythonStrip(word));
    if (normalized.size() < 4 || !isAlphaText(normalized)) return "";
    if (isIn(normalized, nonPluralWords)) return "";

    if (endsWith(normalized, "ies") && normalized.size() > 4) return "ies";
    for (const std::string_view suffix : {"ses", "xes", "zes", "ches", "shes", "oes"}) {
        if (endsWith(normalized, suffix)) return "es";
    }
    if (endsWith(normalized, "es") && !endsWith(normalized, "sses") && !endsWith(normalized, "uses") &&
        !endsWith(normalized, "ises")) {
        return "es";
    }
    if (endsWith(normalized, "s") && !endsWith(normalized, "ss") && !endsWith(normalized, "us") && !endsWith(normalized, "is")) {
        return "s";
    }
    return "";
}

// get_plural_suffix_feature_tokens
Tokens pluralSuffixFeatureTokens(const Words& words) {
    Tokens suffixes;
    Tokens positions;
    for (std::size_t index = 0; index < words.size(); ++index) {
        std::string suffix = pluralSuffixBucket(words[index]);
        if (suffix.empty()) continue;
        suffixes.push_back(std::move(suffix));
        if (index == words.size() - 1) positions.emplace_back("tail");
        else if (index == 0) positions.emplace_back("head");
        else positions.emplace_back("mid");
    }
    if (suffixes.empty()) return {"@plural_suffix_none"};

    Tokens tokens{"@plural_suffix_present"};
    for (const std::string& suffix : dedupePreserveOrder(suffixes)) tokens.push_back("@plural_suffix_" + suffix);
    for (const std::string& position : dedupePreserveOrder(positions)) tokens.push_back("@plural_suffix_" + position);
    return tokens;
}

// _split_identifier_like_text, one step per regular expression of _normalize_text_for_split
Words splitIdentifierLikeText(std::string_view text) {
    if (text.empty()) return {};

    // re.sub(r"::|->|<|>|\(|\)|\[|\]|,|\*|&|/|\\", " ", text)
    std::string separated;
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (startsWith(text.substr(i), "::") || startsWith(text.substr(i), "->")) {
            separated += ' ';
            ++i;
        }
        else if (std::string_view("<>()[],*&/\\").find(text[i]) != std::string_view::npos) {
            separated += ' ';
        }
        else {
            separated += text[i];
        }
    }

    // re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    std::string spaced;
    for (std::size_t i = 0; i < separated.size(); ++i) {
        spaced += separated[i];
        if (i + 1 < separated.size() && (isAsciiLower(separated[i]) || isAsciiDigit(separated[i])) &&
            isAsciiUpper(separated[i + 1])) {
            spaced += ' ';
            spaced += separated[++i];
        }
    }

    // re.sub(r"[_\-.]+", " ", text)
    const auto isJoiner = [](char c) { return c == '_' || c == '-' || c == '.'; };
    std::string normalized;
    for (std::size_t i = 0; i < spaced.size();) {
        if (isJoiner(spaced[i])) {
            normalized += ' ';
            while (i < spaced.size() && isJoiner(spaced[i])) ++i;
        }
        else {
            normalized += spaced[i++];
        }
    }

    // re.findall(r"[A-Za-z]+\d*|\d+[A-Za-z]*", text), each part lowercased
    Words parts;
    for (std::size_t i = 0; i < normalized.size();) {
        std::size_t end = i;
        if (isAsciiLetter(normalized[i])) {
            while (end < normalized.size() && isAsciiLetter(normalized[end])) ++end;
            while (end < normalized.size() && isAsciiDigit(normalized[end])) ++end;
        }
        else if (isAsciiDigit(normalized[i])) {
            while (end < normalized.size() && isAsciiDigit(normalized[end])) ++end;
            while (end < normalized.size() && isAsciiLetter(normalized[end])) ++end;
        }
        else {
            ++i;
            continue;
        }
        parts.push_back(toLowerAscii(std::string_view(normalized).substr(i, end - i)));
        i = end;
    }
    return parts;
}

// _normalize_type_tokens
Words typeWords(std::string_view type) {
    Words words = splitIdentifierLikeText(type);
    words.erase(std::remove_if(words.begin(), words.end(), [](const std::string& word) { return isIn(word, typeQualifierWords); }),
                words.end());
    return words;
}

// re.fullmatch(r"(?:u|i)?int\d+", word)
bool isSizedIntWord(std::string_view word) {
    const auto intWithDigits = [word](std::size_t start) {
        if (word.compare(start, 3, "int") != 0) return false;
        const std::string_view digits = word.substr(start + 3);
        return !digits.empty() && std::all_of(digits.begin(), digits.end(), isAsciiDigit);
    };
    return intWithDigits(0) || (!word.empty() && (word[0] == 'u' || word[0] == 'i') && intWithDigits(1));
}

// re.search(r"(?:^|\W)(?:u?_?int\d*_t|int\d*_t|size_t|ssize_t|ptrdiff_t)(?:$|\W)", text). Each
//   alternative has one way to match at a start, and \W also takes a newline that ends the text.
bool containsIntTypeName(std::string_view text) {
    const auto endsName = [text](std::size_t end) { return end == text.size() || !isWordChar(text[end]); };
    for (std::size_t start = 0; start < text.size(); ++start) {
        if (start > 0 && isWordChar(text[start - 1])) continue;

        // u?_?int\d*_t, which also covers int\d*_t
        std::size_t end = start;
        if (text[end] == 'u') ++end;
        if (end < text.size() && text[end] == '_') ++end;
        if (text.compare(end, 3, "int") == 0) {
            end += 3;
            while (end < text.size() && isAsciiDigit(text[end])) ++end;
            if (text.compare(end, 2, "_t") == 0 && endsName(end + 2)) return true;
        }
        for (const std::string_view name : {"size_t", "ssize_t", "ptrdiff_t"}) {
            if (text.compare(start, name.size(), name) == 0 && endsName(start + name.size())) return true;
        }
    }
    return false;
}

// _looks_int_like
bool looksIntLike(std::string_view normalizedType, const Words& words) {
    if (anyIn(words, typeIntWords)) return true;
    if (std::any_of(words.begin(), words.end(), [](const std::string& word) { return isSizedIntWord(word); })) return true;
    return containsIntTypeName(normalizedType);
}

// get_type_feature_tokens. The Python removes repeated tokens at the end, but the bucket and
//   the flags never repeat.
Tokens typeFeatureTokens(std::string_view type) {
    const std::string normalizedType = toLowerAscii(pythonStrip(type));
    if (normalizedType.empty()) return {"@type_unknown"};
    const Words words = typeWords(type);
    const auto has    = [&normalizedType](std::string_view part) { return normalizedType.find(part) != std::string::npos; };

    std::string bucket = "@type_object_like";
    if (anyIn(words, typeBoolWords)) bucket = "@type_bool";
    else if (anyIn(words, typeVoidWords)) bucket = "@type_void";
    else if (anyIn(words, typeContainerWords)) bucket = "@type_container_like";
    else if (anyIn(words, typeStringWords)) bucket = "@type_string_like";
    else if (anyIn(words, typeCharWords)) bucket = "@type_char_like";
    else if (anyIn(words, typeFloatWords)) bucket = "@type_float_like";
    else if (looksIntLike(normalizedType, words)) bucket = "@type_int_like";
    else if (has("enum")) bucket = "@type_enum_like";
    else if (anyIn(words, typeFunctionWords)) bucket = "@type_function_like";

    Tokens tokens{bucket};
    if (has("*")) tokens.emplace_back("@type_ptr");
    if (has("&")) tokens.emplace_back("@type_ref");
    if (has("[]") || std::find(words.begin(), words.end(), "array") != words.end()) tokens.emplace_back("@type_array");
    if (has("<") && has(">")) tokens.emplace_back("@type_generic");
    if (has("const")) tokens.emplace_back("@type_const");
    if (has("unsigned")) tokens.emplace_back("@type_unsigned");
    if (has("signed")) tokens.emplace_back("@type_signed");
    return tokens;
}

enum class MatchStrength { None, Prefix, Exact };

// _token_match
MatchStrength tokenMatch(std::string_view identifierWord, std::string_view lexiconWord) {
    const std::string left  = toLowerAscii(pythonStrip(identifierWord));
    const std::string right = toLowerAscii(pythonStrip(lexiconWord));
    if (left.empty() || right.empty()) return MatchStrength::None;
    if (left == right) return MatchStrength::Exact;
    if (left.size() >= 2 && right.size() >= 2 && (startsWith(left, right) || startsWith(right, left))) return MatchStrength::Prefix;
    return MatchStrength::None;
}

bool matchesAny(const std::string& word, const Words& lexicon) {
    return std::any_of(lexicon.begin(), lexicon.end(),
                       [&word](const std::string& entry) { return tokenMatch(word, entry) != MatchStrength::None; });
}

// _count_any_overlap
std::size_t countAnyOverlap(const Words& words, const Words& lexicon) {
    return static_cast<std::size_t>(
        std::count_if(words.begin(), words.end(), [&lexicon](const std::string& word) { return matchesAny(word, lexicon); }));
}

// _count_leading_overlap
std::size_t countLeadingOverlap(const Words& words, const Words& lexicon) {
    std::size_t span = 0;
    while (span < words.size() && matchesAny(words[span], lexicon)) ++span;
    return span;
}

// _bucket_overlap_count
std::string bucketOverlapCount(std::size_t count, std::string_view prefix) {
    return "@" + std::string(prefix) + (count == 0 ? "_0" : count == 1 ? "_1" : "_2plus");
}

// _get_head_match_token. As in the Python, each lexicon word replaces the strength of the
//   one before it, and only an exact match ends the loop early.
std::string headMatchToken(const Words& words, const Words& lexicon, std::string_view prefix) {
    const std::string start = "@" + std::string(prefix);
    if (words.empty() || lexicon.empty()) return start + "_head_none";
    MatchStrength strength = MatchStrength::None;
    for (const std::string& entry : lexicon) {
        strength = tokenMatch(words.front(), entry);
        if (strength == MatchStrength::Exact) break;
    }
    if (strength == MatchStrength::Exact) return start + "_head_exact";
    if (strength == MatchStrength::Prefix) return start + "_head_prefix";
    return start + "_head_none";
}

// [token.lower() for token in identifier_tokens if token]
Words lowercaseNonEmpty(const Words& words) {
    Words lowered;
    for (const std::string& word : words) {
        if (!word.empty()) lowered.push_back(toLowerAscii(word));
    }
    return lowered;
}

// get_type_overlap_feature_tokens
Tokens typeOverlapFeatureTokens(const Words& words, std::string_view type) {
    const Words identifierWords = lowercaseNonEmpty(words);
    const Words lexicon         = typeWords(type);
    if (lexicon.empty()) return {"@typeov_head_none", "@typeov_lead_0", "@typeov_any_0"};
    return {headMatchToken(identifierWords, lexicon, "typeov"),
            bucketOverlapCount(countLeadingOverlap(identifierWords, lexicon), "typeov_lead"),
            bucketOverlapCount(countAnyOverlap(identifierWords, lexicon), "typeov_any")};
}

// system_prefix_similarity
std::string systemPrefixSimilarity(std::string_view firstWord, std::string_view systemName) {
    if (firstWord.empty() || systemName.empty()) return "@sim_none";
    const std::string systemLower = toLowerAscii(pythonStrip(systemName));
    const std::string wordLower   = toLowerAscii(pythonStrip(firstWord));
    const double      ratio       = sequenceMatcherRatio(wordLower, systemLower);
    if (ratio > 0.9) return "@sim_high";
    if (ratio > 0.6) return "@sim_mid";
    if (ratio > 0.3) return "@sim_low";
    return "@sim_none";
}

// get_system_overlap_feature_tokens
Tokens systemOverlapFeatureTokens(const Words& words, std::string_view systemName) {
    const Words identifierWords = lowercaseNonEmpty(words);
    const Words systemWords     = splitIdentifierLikeText(systemName);
    if (systemWords.empty()) return {"@sys_head_none", "@sys_lead_0", "@sys_any_0", "@sim_none"};
    return {headMatchToken(identifierWords, systemWords, "sys"),
            bucketOverlapCount(countLeadingOverlap(identifierWords, systemWords), "sys_lead"),
            bucketOverlapCount(countAnyOverlap(identifierWords, systemWords), "sys_any"),
            systemPrefixSimilarity(identifierWords.empty() ? "" : identifierWords.front(), systemName)};
}

// normalize_language
std::string normalizeLanguage(std::string_view language) {
    return "@lang_" + replaceAll(replaceAll(toLowerAscii(pythonStrip(language)), "++", "pp"), "#", "sharp");
}

// get_lead_shape_feature_tokens: the length and the case of the first word, and whether the next word
//   follows it after an underscore or in camelCase
Tokens leadShapeFeatureTokens(const Words& words) {
    if (words.empty()) return {"@lead_none"};
    const std::string& first  = words.front();
    const std::string  length = first.size() == 1 ? "@lead_len1" : first.size() <= 3 ? "@lead_len2_3" : "@lead_len4";

    std::size_t letters = 0;
    std::size_t upper   = 0;
    char        firstLetter = '\0';
    for (const char c : first) {
        if (!isAsciiLetter(c)) continue;
        if (letters++ == 0) firstLetter = c;
        if (isAsciiUpper(c)) ++upper;
    }
    std::string casing;
    if (letters == 0) casing = "@lead_nocase";
    else if (upper == letters) casing = letters > 1 ? "@lead_upper" : "@lead_title";
    else if (isAsciiUpper(firstLetter)) casing = "@lead_title";
    else casing = "@lead_lower";

    if (words.size() == 1) return {length, casing, "@lead_alone"};
    const std::string& next = words[1];
    // str.isupper(): at least one letter and no lower-case letter
    const bool nextIsUpper = std::any_of(next.begin(), next.end(), isAsciiLetter) && std::none_of(next.begin(), next.end(), isAsciiLower);
    const char nextFirst = next.empty() ? '\0' : next.front();
    std::string attach;
    if (casing == "@lead_upper" && nextIsUpper) attach = "@lead_under";
    else if (isAsciiUpper(nextFirst) || isAsciiDigit(nextFirst)) attach = "@lead_camel";
    else attach = "@lead_under";
    return {length, casing, attach};
}

void appendAll(Tokens& tokens, Tokens more) {
    tokens.insert(tokens.end(), std::make_move_iterator(more.begin()), std::make_move_iterator(more.end()));
}

// One entry of FEATURE_FUNCTIONS. Each feature function that the port computes returns non-empty
//   strings, therefore the None and empty-string branches of get_feature_tokens never apply.
struct Feature {
    std::string_view name;
    void (*append)(const IdentifierRow&, const Words&, Tokens&) = nullptr;
};

// FEATURE_FUNCTIONS without lead_word, which needs the WordNet corpus of NLTK that the port lacks
constexpr std::array<Feature, 12> featureFunctions{{
    {"context", [](const IdentifierRow& row, const Words&, Tokens& tokens) { tokens.push_back(contextToken(row.context)); }},
    {"hungarian", [](const IdentifierRow&, const Words& words, Tokens& tokens) { tokens.push_back(hungarianPrefixSplit(words)); }},
    {"hungarian_legacy",
     [](const IdentifierRow&, const Words& words, Tokens& tokens) { tokens.push_back(hungarianPrefixLegacy(words)); }},
    {"cvr", [](const IdentifierRow&, const Words& words, Tokens& tokens) { tokens.push_back(consonantVowelRatioBucket(words)); }},
    {"digit", [](const IdentifierRow&, const Words& words, Tokens& tokens) { tokens.push_back(digitFeature(words)); }},
    {"digit_connector",
     [](const IdentifierRow&, const Words& words, Tokens& tokens) { appendAll(tokens, digitConnectorFeatureTokens(words)); }},
    {"plural_suffix",
     [](const IdentifierRow&, const Words& words, Tokens& tokens) { appendAll(tokens, pluralSuffixFeatureTokens(words)); }},
    {"type", [](const IdentifierRow& row, const Words&, Tokens& tokens) { appendAll(tokens, typeFeatureTokens(row.type)); }},
    {"type_overlap",
     [](const IdentifierRow& row, const Words& words, Tokens& tokens) { appendAll(tokens, typeOverlapFeatureTokens(words, row.type)); }},
    {"language", [](const IdentifierRow& row, const Words&, Tokens& tokens) { tokens.push_back(normalizeLanguage(row.language)); }},
    {"sys_sim",
     [](const IdentifierRow& row, const Words& words, Tokens& tokens) {
         appendAll(tokens, systemOverlapFeatureTokens(words, row.systemName));
     }},
    {"lead_shape", [](const IdentifierRow&, const Words& words, Tokens& tokens) { appendAll(tokens, leadShapeFeatureTokens(words)); }},
}};

const Feature* findFeature(std::string_view name) {
    const auto found = std::find_if(featureFunctions.begin(), featureFunctions.end(), [name](const Feature& feature) { return feature.name == name; });
    return found == featureFunctions.end() ? nullptr : &*found;
}

template <std::size_t size>
bool contains(const std::array<std::string_view, size>& names, std::string_view name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

template <std::size_t size>
std::vector<std::string> toStrings(const std::array<std::string_view, size>& names) {
    return {names.begin(), names.end()};
}

std::string joinNames(const std::vector<std::string>& names) {
    std::string joined;
    for (const std::string& name : names) {
        if (!joined.empty()) joined += ", ";
        joined += name;
    }
    return joined;
}

// The names of a feature list in FEATURE_ORDER, as normalize_selected_features orders them
std::vector<std::string> orderedFeatureNames(const std::vector<std::string>& selected) {
    std::vector<std::string> names;
    for (const std::string& feature : selected) {
        const std::string_view name = pythonStrip(feature);
        if (!name.empty()) names.emplace_back(name);
    }

    std::vector<std::string> unknown;
    for (const std::string& name : names) {
        const bool known = contains(release::availableFeatures, name) || contains(release::internalFeatures, name);
        if (!known && std::find(unknown.begin(), unknown.end(), name) == unknown.end()) unknown.push_back(name);
    }
    if (!unknown.empty()) {
        std::sort(unknown.begin(), unknown.end());
        throw std::invalid_argument("Unknown lm_based feature(s): " + joinNames(unknown) +
                                    ". Available features: " + joinNames(availableFeatures()));
    }

    std::vector<std::string> ordered;
    for (const std::string_view name : release::featureOrder) {
        if (std::find(names.begin(), names.end(), name) != names.end()) ordered.emplace_back(name);
    }
    return ordered;
}

} // namespace

const std::vector<std::string>& availableFeatures() {
    static const std::vector<std::string> features = toStrings(release::availableFeatures);
    return features;
}

const std::vector<std::string>& legacyFeatures() {
    static const std::vector<std::string> features = toStrings(release::legacyFeatures);
    return features;
}

std::vector<std::string> normalizeSelectedFeatures(const std::optional<std::vector<std::string>>& selected) {
    if (selected) return orderedFeatureNames(*selected);
    std::vector<std::string> defaults;
    for (const std::string_view name : release::availableFeatures) {
        if (!contains(release::optInFeatures, name)) defaults.emplace_back(name);
    }
    return defaults;
}

std::vector<std::string> modelFeatures(const std::optional<std::vector<std::string>>& savedFeatures) {
    std::vector<std::string> features = normalizeSelectedFeatures(savedFeatures ? *savedFeatures : legacyFeatures());
    for (const std::string& name : features) {
        if (findFeature(name) == nullptr) {
            throw std::runtime_error("the model selects the feature " + name + ", which the C++ port does not compute (PORT.md)");
        }
    }
    return features;
}

std::optional<std::vector<std::string>> readSavedFeatures(const std::string& configPath) {
    const JsonValue  config = readJsonFile(configPath);
    const JsonValue* saved  = config.find("selected_features");
    if (saved == nullptr || saved->kind == JsonValue::Kind::Null) return std::nullopt;
    if (saved->kind != JsonValue::Kind::Array) {
        throw std::runtime_error(configPath + ": selected_features is neither a list nor null");
    }
    std::vector<std::string> features;
    for (const JsonValue& item : saved->items) {
        if (item.kind != JsonValue::Kind::String) {
            throw std::runtime_error(configPath + ": selected_features holds a value that is not a string");
        }
        features.push_back(item.text);
    }
    return features;
}

std::vector<std::string> featureTokens(const IdentifierRow& row, const std::vector<std::string>& words,
                                       const std::vector<std::string>& features) {
    requireAscii(row.context, "context");
    requireAscii(row.systemName, "system name");
    requireAscii(row.type, "type");
    requireAscii(row.language, "language");
    for (const std::string& word : words) requireAscii(word, "word");

    Tokens tokens;
    for (const std::string& name : orderedFeatureNames(features)) {
        const Feature* feature = findFeature(name);
        if (feature == nullptr) throw std::invalid_argument("the C++ port does not compute the feature " + name);
        feature->append(row, words, tokens);
    }
    return tokens;
}

ModelInputTokens buildModelInputTokens(const IdentifierRow& row, const std::vector<std::string>& words,
                                       const std::vector<std::string>& features) {
    ModelInputTokens input;
    input.tokens       = featureTokens(row, words, features);
    input.featureCount = input.tokens.size();
    for (std::size_t index = 0; index < words.size(); ++index) {
        const bool last = index + 1 == words.size();
        input.tokens.emplace_back(last ? "@pos_2" : index == 0 ? "@pos_0" : "@pos_1");
        input.tokens.push_back(words[index]);
    }
    return input;
}

} // namespace scalar
