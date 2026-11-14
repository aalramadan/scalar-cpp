// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file englishStemmer.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */
// English Snowball stemmer that the Ronin splitter of SCALAR calls.
// Port of nltk/stem/snowball.py (EnglishStemmer) of NLTK release 3.10.3, Apache-2.0.

#include "englishStemmer.hpp"
#include "asciiText.hpp"

#include <array>
#include <cstddef>
#include <string_view>
#include <unordered_map>

namespace scalar {

namespace {

constexpr std::array<std::string_view, 3> step0Suffixes{"'s'", "'s", "'"};
constexpr std::array<std::string_view, 6> step1aSuffixes{"sses", "ied", "ies", "us", "ss", "s"};
constexpr std::array<std::string_view, 6> step1bSuffixes{"eedly", "ingly", "edly", "eed", "ing", "ed"};
constexpr std::array<std::string_view, 24> step2Suffixes{
    "ization", "ational", "fulness", "ousness", "iveness", "tional", "biliti", "lessli",
    "entli", "ation", "alism", "aliti", "ousli", "iviti", "fulli", "enci",
    "anci", "abli", "izer", "ator", "alli", "bli", "ogi", "li"};
constexpr std::array<std::string_view, 9> step3Suffixes{
    "ational", "tional", "alize", "icate", "iciti", "ative", "ical", "ness", "ful"};
constexpr std::array<std::string_view, 18> step4Suffixes{
    "ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism",
    "ate", "iti", "ous", "ive", "ize", "ion", "al", "er", "ic"};
constexpr std::array<std::string_view, 9> doubleConsonants{"bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt"};
constexpr std::string_view liEnding = "cdeghkmnrt";

const std::unordered_map<std::string, std::string>& specialWords() {
    static const std::unordered_map<std::string, std::string> words{
        {"skis", "ski"}, {"skies", "sky"}, {"dying", "die"}, {"lying", "lie"}, {"tying", "tie"},
        {"idly", "idl"}, {"gently", "gentl"}, {"ugly", "ugli"}, {"early", "earli"}, {"only", "onli"},
        {"singly", "singl"}, {"sky", "sky"}, {"news", "news"}, {"howe", "howe"}, {"atlas", "atlas"},
        {"cosmos", "cosmos"}, {"bias", "bias"}, {"andes", "andes"}, {"inning", "inning"},
        {"innings", "inning"}, {"outing", "outing"}, {"outings", "outing"}, {"canning", "canning"},
        {"cannings", "canning"}, {"herring", "herring"}, {"herrings", "herring"}, {"earring", "earring"},
        {"earrings", "earring"}, {"proceed", "proceed"}, {"proceeds", "proceed"}, {"proceeded", "proceed"},
        {"proceeding", "proceed"}, {"exceed", "exceed"}, {"exceeds", "exceed"}, {"exceeded", "exceed"},
        {"exceeding", "exceed"}, {"succeed", "succeed"}, {"succeeds", "succeed"}, {"succeeded", "succeed"},
        {"succeeding", "succeed"}};
    return words;
}

bool isVowel(char c) {
    return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

bool isOneOf(char c, std::string_view characters) {
    return characters.find(c) != std::string_view::npos;
}

bool startsWith(const std::string& text, std::string_view prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

bool endsWith(const std::string& text, std::string_view suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool containsVowel(std::string_view text) {
    for (const char c : text) {
        if (isVowel(c)) {
            return true;
        }
    }
    return false;
}

// Python's text[:-count] for a count of at least one
std::string dropLast(const std::string& text, std::size_t count) {
    return count >= text.size() ? std::string() : text.substr(0, text.size() - count);
}

// Python's text[-position]; std::out_of_range stands where Python raises IndexError
char fromEnd(const std::string& text, std::size_t position) {
    return text.at(text.size() - position);
}

// R1 or R2 of _r1r2_standard: the region after the first non-vowel that follows a vowel
std::string regionAfterVowelConsonant(const std::string& text) {
    for (std::size_t i = 1; i < text.size(); ++i) {
        if (!isVowel(text[i]) && isVowel(text[i - 1])) {
            return text.substr(i + 1);
        }
    }
    return std::string();
}

// The word with its regions R1 and R2, which NLTK keeps as three separate strings
struct Regions {
    std::string word;
    std::string r1;
    std::string r2;

    void dropLastOfAll(std::size_t count) {
        word = dropLast(word, count);
        r1 = dropLast(r1, count);
        r2 = dropLast(r2, count);
    }

    // suffix_replace on the word and on each region at least as long as the suffix; a shorter region
    // becomes its fallback, as in NLTK
    void replaceSuffixOfAll(std::string_view suffix, std::string_view replacement,
                            std::string_view r1Fallback, std::string_view r2Fallback) {
        word = dropLast(word, suffix.size()).append(replacement);
        r1 = r1.size() >= suffix.size() ? dropLast(r1, suffix.size()).append(replacement) : std::string(r1Fallback);
        r2 = r2.size() >= suffix.size() ? dropLast(r2, suffix.size()).append(replacement) : std::string(r2Fallback);
    }

    // Replaces the last character of the word and of each nonempty region by 'c'
    void replaceLastOfAll(char c) {
        word = dropLast(word, 1) + c;
        r1 = r1.empty() ? std::string() : dropLast(r1, 1) + c;
        r2 = r2.empty() ? std::string() : dropLast(r2, 1) + c;
    }
};

void step1a(Regions& w) {
    for (const std::string_view suffix : step1aSuffixes) {
        if (!endsWith(w.word, suffix)) {
            continue;
        }
        if (suffix == "sses") {
            w.dropLastOfAll(2);
        } else if (suffix == "ied" || suffix == "ies") {
            w.dropLastOfAll(w.word.size() - suffix.size() > 1 ? 2 : 1);
        } else if (suffix == "s") {
            if (containsVowel(dropLast(w.word, 2))) {
                w.dropLastOfAll(1);
            }
        }
        return;
    }
}

void step1b(Regions& w) {
    for (const std::string_view suffix : step1bSuffixes) {
        if (!endsWith(w.word, suffix)) {
            continue;
        }
        if (suffix == "eed" || suffix == "eedly") {
            if (endsWith(w.r1, suffix)) {
                w.replaceSuffixOfAll(suffix, "ee", "", "");
            }
            return;
        }
        if (!containsVowel(dropLast(w.word, suffix.size()))) {
            return;
        }
        w.dropLastOfAll(suffix.size());
        const std::string& word = w.word;
        bool doubleConsonant = false;
        for (const std::string_view pair : doubleConsonants) {
            doubleConsonant = doubleConsonant || endsWith(word, pair);
        }
        if (endsWith(word, "at") || endsWith(word, "bl") || endsWith(word, "iz")) {
            w.word += 'e';
            w.r1 += 'e';
            if (w.word.size() > 5 || w.r1.size() >= 3) {
                w.r2 += 'e';
            }
        } else if (doubleConsonant) {
            w.dropLastOfAll(1);
        } else if ((w.r1.empty() && word.size() >= 3 && !isVowel(fromEnd(word, 1)) && !isOneOf(fromEnd(word, 1), "wxY") &&
                    isVowel(fromEnd(word, 2)) && !isVowel(fromEnd(word, 3))) ||
                   (w.r1.empty() && word.size() == 2 && isVowel(word[0]) && !isVowel(word[1]))) {
            w.word += 'e';
            if (!w.r1.empty()) {
                w.r1 += 'e';
            }
            if (!w.r2.empty()) {
                w.r2 += 'e';
            }
        }
        return;
    }
}

void step2(Regions& w) {
    for (const std::string_view suffix : step2Suffixes) {
        if (!endsWith(w.word, suffix)) {
            continue;
        }
        if (!endsWith(w.r1, suffix)) {
            return;
        }
        if (suffix == "tional" || suffix == "entli" || suffix == "fulli" || suffix == "lessli") {
            w.dropLastOfAll(2);
        } else if (suffix == "enci" || suffix == "anci" || suffix == "abli") {
            w.replaceLastOfAll('e');
        } else if (suffix == "izer" || suffix == "ization") {
            w.replaceSuffixOfAll(suffix, "ize", "", "");
        } else if (suffix == "ational" || suffix == "ation" || suffix == "ator") {
            w.replaceSuffixOfAll(suffix, "ate", "", "e");
        } else if (suffix == "alism" || suffix == "aliti" || suffix == "alli") {
            w.replaceSuffixOfAll(suffix, "al", "", "");
        } else if (suffix == "fulness") {
            w.dropLastOfAll(4);
        } else if (suffix == "ousli" || suffix == "ousness") {
            w.replaceSuffixOfAll(suffix, "ous", "", "");
        } else if (suffix == "iveness" || suffix == "iviti") {
            w.replaceSuffixOfAll(suffix, "ive", "", "e");
        } else if (suffix == "biliti" || suffix == "bli") {
            w.replaceSuffixOfAll(suffix, "ble", "", "");
        } else if (suffix == "ogi" && fromEnd(w.word, 4) == 'l') {
            w.dropLastOfAll(1);
        } else if (suffix == "li" && isOneOf(fromEnd(w.word, 3), liEnding)) {
            w.dropLastOfAll(2);
        }
        return;
    }
}

void step3(Regions& w) {
    for (const std::string_view suffix : step3Suffixes) {
        if (!endsWith(w.word, suffix)) {
            continue;
        }
        if (!endsWith(w.r1, suffix)) {
            return;
        }
        if (suffix == "tional") {
            w.dropLastOfAll(2);
        } else if (suffix == "ational") {
            w.replaceSuffixOfAll(suffix, "ate", "", "");
        } else if (suffix == "alize") {
            w.dropLastOfAll(3);
        } else if (suffix == "icate" || suffix == "iciti" || suffix == "ical") {
            w.replaceSuffixOfAll(suffix, "ic", "", "");
        } else if (suffix == "ful" || suffix == "ness") {
            w.dropLastOfAll(suffix.size());
        } else if (suffix == "ative" && endsWith(w.r2, suffix)) {
            w.dropLastOfAll(5);
        }
        return;
    }
}

void step4(Regions& w) {
    for (const std::string_view suffix : step4Suffixes) {
        if (!endsWith(w.word, suffix)) {
            continue;
        }
        if (endsWith(w.r2, suffix)) {
            if (suffix != "ion") {
                w.dropLastOfAll(suffix.size());
            } else if (isOneOf(fromEnd(w.word, 4), "st")) {
                w.dropLastOfAll(3);
            }
        }
        return;
    }
}

void step5(Regions& w) {
    std::string& word = w.word;
    if (endsWith(w.r2, "l") && fromEnd(word, 2) == 'l') {
        word = dropLast(word, 1);
    } else if (endsWith(w.r2, "e")) {
        word = dropLast(word, 1);
    } else if (endsWith(w.r1, "e")) {
        if (word.size() >= 4 && (isVowel(fromEnd(word, 2)) || isOneOf(fromEnd(word, 2), "wxY") ||
                                 !isVowel(fromEnd(word, 3)) || isVowel(fromEnd(word, 4)))) {
            word = dropLast(word, 1);
        }
    }
}

}

std::string stemEnglish(const std::string& input) {
    const std::string lowered = toLowerAscii(input);
    if (lowered.size() <= 2) {
        return lowered;
    }
    if (const auto special = specialWords().find(lowered); special != specialWords().end()) {
        return special->second;
    }

    // NLTK first maps U+2018, U+2019, and U+201B to an apostrophe, which no ASCII word holds
    Regions w;
    w.word = lowered;
    if (startsWith(w.word, "'")) {
        w.word.erase(0, 1);
    }
    if (startsWith(w.word, "y")) {
        w.word[0] = 'Y';
    }
    for (std::size_t i = 1; i < w.word.size(); ++i) {
        if (isVowel(w.word[i - 1]) && w.word[i] == 'y') {
            w.word[i] = 'Y';
        }
    }

    if (startsWith(w.word, "gener") || startsWith(w.word, "commun") || startsWith(w.word, "arsen")) {
        w.r1 = w.word.substr(startsWith(w.word, "commun") ? 6 : 5);
    } else {
        w.r1 = regionAfterVowelConsonant(w.word);
    }
    w.r2 = regionAfterVowelConsonant(w.r1);

    for (const std::string_view suffix : step0Suffixes) {
        if (endsWith(w.word, suffix)) {
            w.dropLastOfAll(suffix.size());
            break;
        }
    }
    step1a(w);
    step1b(w);
    if (w.word.size() > 2 && isOneOf(w.word.back(), "yY") && !isVowel(fromEnd(w.word, 2))) {
        w.replaceLastOfAll('i');
    }
    step2(w);
    step3(w);
    step4(w);
    step5(w);

    for (char& c : w.word) {
        if (c == 'Y') {
            c = 'y';
        }
    }
    return w.word;
}

}
