// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file wordpieceTokenizer.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The DistilBERT tokenizer of SCALAR for pre-split words, on ASCII text
// Port of the BertNormalizer, BertPreTokenizer, WordPiece, added-token, truncation, and TemplateProcessing steps of tokenizers v0.23.3 (Apache-2.0), as BertTokenizer of transformers v5.19.0 (Apache-2.0) sets them up

#include "wordpieceTokenizer.hpp"

#include "asciiText.hpp"
#include "dataFile.hpp"
#include "jsonReader.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>

namespace scalar {

namespace {

// BertTokenizer of transformers 5 builds WordPiece(vocab, unk_token) with these defaults and
//   ignores the values of tokenizer.json, which the loader checks for agreement
constexpr std::size_t      maxInputCharsPerWord    = 100;
constexpr std::string_view continuingSubwordPrefix = "##";

// The special tokens of tokenizer_config.json, with the defaults of BertTokenizer
struct SpecialNames {
    std::string unknown   = "[UNK]";
    std::string separator = "[SEP]";
    std::string padding   = "[PAD]";
    std::string classification = "[CLS]";
    std::string mask      = "[MASK]";

    std::vector<std::string> all() const { return {padding, unknown, classification, separator, mask}; }
};

using Vocabulary = std::unordered_map<std::string, std::int32_t>;

[[noreturn]] void reject(const std::string& file, const std::string& setting) {
    throw std::runtime_error("SCALAR tokenizer: " + file + ": " + setting + " differs from what the C++ port follows");
}

const JsonValue& requireMember(const JsonValue& object, const std::string& file, std::string_view key) {
    const JsonValue* value = object.find(key);
    if (value == nullptr) reject(file, "missing " + std::string(key));
    return *value;
}

void requireBoolean(const JsonValue& object, const std::string& file, std::string_view key, bool expected) {
    const JsonValue& value = requireMember(object, file, key);
    if (value.kind != JsonValue::Kind::Boolean || value.boolean != expected) reject(file, std::string(key));
}

void requireString(const JsonValue& object, const std::string& file, std::string_view key, std::string_view expected) {
    const JsonValue& value = requireMember(object, file, key);
    if (value.kind != JsonValue::Kind::String || value.text != expected) reject(file, std::string(key));
}

void requireNumber(const JsonValue& object, const std::string& file, std::string_view key, double expected) {
    const JsonValue& value = requireMember(object, file, key);
    if (value.kind != JsonValue::Kind::Number || value.number != expected) reject(file, std::string(key));
}

// A setting that may be missing, or else must hold the given value
void allowMissingOrBoolean(const JsonValue& object, const std::string& file, std::string_view key, bool expected) {
    if (object.find(key) != nullptr) requireBoolean(object, file, key, expected);
}

void allowMissingOrNull(const JsonValue& object, const std::string& file, std::string_view key) {
    const JsonValue* value = object.find(key);
    if (value != nullptr && value->kind != JsonValue::Kind::Null) reject(file, std::string(key));
}

void allowMissingOrEmpty(const JsonValue& object, const std::string& file, std::string_view key) {
    const JsonValue* value = object.find(key);
    if (value == nullptr || value->kind == JsonValue::Kind::Null) return;
    const bool empty = (value->kind == JsonValue::Kind::Array || value->kind == JsonValue::Kind::Object) && value->items.empty();
    if (!empty) reject(file, std::string(key));
}

std::string readFile(const std::string& path) {
    const std::optional<DataFile> file = openDataFile(path);
    if (!file) throw std::runtime_error("SCALAR tokenizer: cannot read " + path);
    return std::string(file->bytes());
}

// The lines of vocab.txt as load_vocab of transformers reads them: text mode splits lines at
//   \n, \r, and \r\n, and rstrip("\n") removes the line end
std::vector<std::string> readVocabularyLines(const std::string& path) {
    const std::string text = readFile(path);
    std::vector<std::string> lines;
    std::string line;
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\n' || text[i] == '\r') {
            lines.push_back(line);
            line.clear();
            if (text[i] == '\r' && i + 1 < text.size() && text[i + 1] == '\n') ++i;
        }
        else {
            line += text[i];
        }
    }
    if (!line.empty()) lines.push_back(line);
    return lines;
}

// The tokens of model.vocab in tokenizer.json in the order of their ids, which must run from 0 without gaps
std::vector<std::string> readJsonVocabulary(const std::string& path) {
    const JsonValue  json  = readJsonFile(path);
    const JsonValue& vocab = requireMember(requireMember(json, path, "model"), path, "vocab");
    if (vocab.kind != JsonValue::Kind::Object) reject(path, "model.vocab");
    std::vector<std::string> tokens(vocab.items.size());
    std::vector<bool>        filled(vocab.items.size(), false);
    for (std::size_t i = 0; i < vocab.keys.size(); ++i) {
        const double id = vocab.items[i].kind == JsonValue::Kind::Number ? vocab.items[i].number : -1.0;
        const auto   index = static_cast<std::size_t>(id);
        if (id < 0 || id != static_cast<double>(index) || index >= tokens.size() || filled[index]) {
            reject(path, "model.vocab id of " + vocab.keys[i]);
        }
        tokens[index] = vocab.keys[i];
        filled[index] = true;
    }
    return tokens;
}

// One added token as tokenizer.json and tokenizer_config.json describe it: a special token of
//   tokenizer_config.json that is matched on the raw text, never normalized or stripped
void checkAddedToken(const JsonValue& token, const std::string& file, const std::string& content, const Vocabulary& vocabulary,
                     const std::vector<std::string>& specialNames) {
    if (std::find(specialNames.begin(), specialNames.end(), content) == specialNames.end()) {
        reject(file, "added token " + content + " beyond the five special tokens");
    }
    if (vocabulary.count(content) == 0) reject(file, "added token " + content + " outside the vocabulary");
    requireBoolean(token, file, "single_word", false);
    requireBoolean(token, file, "lstrip", false);
    requireBoolean(token, file, "rstrip", false);
    requireBoolean(token, file, "normalized", false);
    requireBoolean(token, file, "special", true);
}

void checkTokenizerJson(const std::string& path, const Vocabulary& vocabulary, const SpecialNames& names, std::size_t maxLength) {
    const JsonValue json = readJsonFile(path);

    const JsonValue& normalizer = requireMember(json, path, "normalizer");
    requireString(normalizer, path, "type", "BertNormalizer");
    requireBoolean(normalizer, path, "clean_text", true);
    requireBoolean(normalizer, path, "handle_chinese_chars", true);
    allowMissingOrNull(normalizer, path, "strip_accents");
    requireBoolean(normalizer, path, "lowercase", true);

    requireString(requireMember(json, path, "pre_tokenizer"), path, "type", "BertPreTokenizer");

    const JsonValue& model = requireMember(json, path, "model");
    if (model.find("type") != nullptr) requireString(model, path, "type", "WordPiece");
    requireString(model, path, "unk_token", names.unknown);
    requireString(model, path, "continuing_subword_prefix", continuingSubwordPrefix);
    requireNumber(model, path, "max_input_chars_per_word", static_cast<double>(maxInputCharsPerWord));
    const JsonValue& vocab = requireMember(model, path, "vocab");
    if (vocab.kind != JsonValue::Kind::Object || vocab.items.size() != vocabulary.size()) reject(path, "model.vocab size");
    for (std::size_t i = 0; i < vocab.keys.size(); ++i) {
        const auto entry = vocabulary.find(vocab.keys[i]);
        if (entry == vocabulary.end() || vocab.items[i].kind != JsonValue::Kind::Number ||
            vocab.items[i].number != static_cast<double>(entry->second)) {
            reject(path, "model.vocab entry " + vocab.keys[i] + " against the vocabulary");
        }
    }

    const JsonValue& addedTokens = requireMember(json, path, "added_tokens");
    const std::vector<std::string> specialNames = names.all();
    if (addedTokens.kind != JsonValue::Kind::Array || addedTokens.items.size() != specialNames.size()) {
        reject(path, "the number of added_tokens");
    }
    std::vector<std::string> seen;
    for (const JsonValue& token : addedTokens.items) {
        const JsonValue& content = requireMember(token, path, "content");
        if (content.kind != JsonValue::Kind::String) reject(path, "added token content");
        checkAddedToken(token, path, content.text, vocabulary, specialNames);
        requireNumber(token, path, "id", static_cast<double>(vocabulary.at(content.text)));
        seen.push_back(content.text);
    }
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) reject(path, "a repeated added token");

    const JsonValue& postProcessor = requireMember(json, path, "post_processor");
    requireString(postProcessor, path, "type", "TemplateProcessing");
    const JsonValue& single = requireMember(postProcessor, path, "single");
    const std::string pieceNames[] = {"SpecialToken", "Sequence", "SpecialToken"};
    const std::string pieceIds[]   = {names.classification, "A", names.separator};
    if (single.kind != JsonValue::Kind::Array || single.items.size() != 3) reject(path, "post_processor.single");
    for (std::size_t i = 0; i < 3; ++i) {
        const JsonValue& piece = requireMember(single.items[i], path, pieceNames[i]);
        requireString(piece, path, "id", pieceIds[i]);
        requireNumber(piece, path, "type_id", 0);
    }
    const JsonValue& templateTokens = requireMember(postProcessor, path, "special_tokens");
    for (const std::string& name : {names.classification, names.separator}) {
        const JsonValue& ids = requireMember(requireMember(templateTokens, path, name), path, "ids");
        if (ids.kind != JsonValue::Kind::Array || ids.items.size() != 1 || ids.items[0].kind != JsonValue::Kind::Number ||
            ids.items[0].number != static_cast<double>(vocabulary.at(name))) {
            reject(path, "post_processor.special_tokens of " + name);
        }
    }

    // The call sets the length limit, but the direction of tokenizer.json becomes truncation_side
    const JsonValue& truncation = requireMember(json, path, "truncation");
    if (truncation.kind != JsonValue::Kind::Null) {
        requireString(truncation, path, "direction", "Right");
        requireString(truncation, path, "strategy", "LongestFirst");
        requireNumber(truncation, path, "stride", 0);
        requireNumber(truncation, path, "max_length", static_cast<double>(maxLength));
    }
    if (requireMember(json, path, "padding").kind != JsonValue::Kind::Null) reject(path, "padding");
}

SpecialNames readSpecialNames(const JsonValue& config, const std::string& path) {
    SpecialNames names;
    const std::pair<std::string_view, std::string*> fields[] = {
        {"unk_token", &names.unknown}, {"sep_token", &names.separator}, {"pad_token", &names.padding},
        {"cls_token", &names.classification}, {"mask_token", &names.mask}};
    for (const auto& [key, name] : fields) {
        const JsonValue* value = config.find(key);
        if (value == nullptr) continue;
        if (value->kind != JsonValue::Kind::String || value->text.empty()) reject(path, std::string(key));
        *name = value->text;
    }
    return names;
}

// model_max_length, the limit of the call tokenizer(..., truncation=True) of SCALAR. transformers saves
//   1e30 for a tokenizer without a limit, which the port does not follow.
std::size_t readModelMaxLength(const JsonValue& config, const std::string& path) {
    const JsonValue& value = requireMember(config, path, "model_max_length");
    const bool valid = value.kind == JsonValue::Kind::Number && value.number >= 3 &&
                       value.number <= static_cast<double>(std::numeric_limits<std::int32_t>::max()) &&
                       value.number == static_cast<double>(static_cast<std::size_t>(value.number));
    if (!valid) reject(path, "model_max_length");
    return static_cast<std::size_t>(value.number);
}

void checkTokenizerConfig(const JsonValue& config, const std::string& path, const Vocabulary& vocabulary, const SpecialNames& names) {
    requireBoolean(config, path, "do_lower_case", true);
    allowMissingOrNull(config, path, "strip_accents");
    allowMissingOrBoolean(config, path, "tokenize_chinese_chars", true);
    allowMissingOrBoolean(config, path, "split_special_tokens", false);
    allowMissingOrEmpty(config, path, "additional_special_tokens");
    allowMissingOrEmpty(config, path, "extra_special_tokens");
    if (config.find("truncation_side") != nullptr) requireString(config, path, "truncation_side", "right");

    // A tokenizer saved by the tokenizers backend of transformers 5 keeps its added tokens in tokenizer.json alone
    const JsonValue* decoderEntry = config.find("added_tokens_decoder");
    if (decoderEntry == nullptr) return;
    const JsonValue& decoder = *decoderEntry;
    const std::vector<std::string> specialNames = names.all();
    if (decoder.kind != JsonValue::Kind::Object || decoder.items.size() != specialNames.size()) {
        reject(path, "the number of added_tokens_decoder entries");
    }
    std::vector<std::string> seen;
    for (std::size_t i = 0; i < decoder.items.size(); ++i) {
        const JsonValue& content = requireMember(decoder.items[i], path, "content");
        if (content.kind != JsonValue::Kind::String) reject(path, "added token content");
        checkAddedToken(decoder.items[i], path, content.text, vocabulary, specialNames);
        if (decoder.keys[i] != std::to_string(vocabulary.at(content.text))) reject(path, "the id of added token " + content.text);
        seen.push_back(content.text);
    }
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) reject(path, "a repeated added token");
}

void checkSpecialTokensMap(const std::string& path, const SpecialNames& names) {
    if (!dataFileExists(path)) return;
    const JsonValue map = readJsonFile(path);
    const std::pair<std::string_view, std::string> fields[] = {
        {"unk_token", names.unknown}, {"sep_token", names.separator}, {"pad_token", names.padding},
        {"cls_token", names.classification}, {"mask_token", names.mask}};
    if (map.kind != JsonValue::Kind::Object || map.items.size() != std::size(fields)) reject(path, "the special tokens");
    for (const auto& [key, name] : fields) requireString(map, path, key, name);
}

bool isAsciiPunctuation(char c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}

} // namespace

WordPieceTokenizer::WordPieceTokenizer(const std::string& modelDirectory) {
    const std::filesystem::path folder(modelDirectory);
    const std::string vocabPath  = (folder / "vocab.txt").string();
    const std::string jsonPath   = (folder / "tokenizer.json").string();
    const std::string configPath = (folder / "tokenizer_config.json").string();

    // Without vocab.txt, transformers builds the tokenizer from tokenizer.json alone
    const bool hasVocabFile = dataFileExists(vocabPath);
    const std::string vocabularySource = hasVocabFile ? vocabPath : jsonPath;
    const std::vector<std::string> lines = hasVocabFile ? readVocabularyLines(vocabPath) : readJsonVocabulary(jsonPath);
    for (std::size_t index = 0; index < lines.size(); ++index) {
        if (!vocabulary.emplace(lines[index], static_cast<std::int32_t>(index)).second) {
            throw std::runtime_error("SCALAR tokenizer: " + vocabularySource + " repeats the token " + lines[index]);
        }
        longestEntry = std::max(longestEntry, lines[index].size());
    }

    if (dataFileExists((folder / "added_tokens.json").string())) reject((folder / "added_tokens.json").string(), "an added token file");
    const JsonValue    config = readJsonFile(configPath);
    const SpecialNames names  = readSpecialNames(config, configPath);
    for (const std::string& name : names.all()) {
        if (vocabulary.count(name) == 0) reject(configPath, "special token " + name + " outside the vocabulary");
    }
    checkTokenizerConfig(config, configPath, vocabulary, names);
    maxLength = readModelMaxLength(config, configPath);
    checkTokenizerJson(jsonPath, vocabulary, names, maxLength);
    checkSpecialTokensMap((folder / "special_tokens_map.json").string(), names);

    for (const std::string& name : names.all()) {
        specialTokens.emplace_back(name, vocabulary.at(name));
        startsSpecialToken[static_cast<unsigned char>(name.front())] = true;
    }
    unknownId        = vocabulary.at(names.unknown);
    classificationId = vocabulary.at(names.classification);
    separatorId      = vocabulary.at(names.separator);
    for (int c = 0; c < 128; ++c) {
        const auto entry   = vocabulary.find(std::string(1, static_cast<char>(c)));
        punctuationIds[static_cast<std::size_t>(c)] = entry == vocabulary.end() ? unknownId : entry->second;
    }
}

std::optional<std::int32_t> WordPieceTokenizer::tokenId(const std::string& token) const {
    const auto entry = vocabulary.find(token);
    if (entry == vocabulary.end()) return std::nullopt;
    return entry->second;
}

Encoding WordPieceTokenizer::encode(const std::vector<std::string>& words) const {
    for (const std::string& word : words) requireAscii(word, "word");

    // Truncation keeps maxLength ids with [CLS] and [SEP]; the ids past the limit go nowhere
    const std::size_t limit = maxLength - 2;
    Encoding encoding;
    encoding.ids.push_back(classificationId);
    encoding.wordIds.emplace_back(std::nullopt);
    std::vector<std::int32_t> wordTokenIds;
    for (std::size_t index = 0; index < words.size() && encoding.ids.size() <= limit; ++index) {
        wordTokenIds.clear();
        appendWordIds(words[index], wordTokenIds);
        for (const std::int32_t id : wordTokenIds) {
            if (encoding.ids.size() > limit) break;
            encoding.ids.push_back(id);
            encoding.wordIds.emplace_back(index);
        }
    }
    encoding.ids.push_back(separatorId);
    encoding.wordIds.emplace_back(std::nullopt);
    return encoding;
}

// The added vocabulary splits the special tokens off the raw word, the leftmost match first
//   and the longest one at a position, before the rest of the word is normalized
void WordPieceTokenizer::appendWordIds(std::string_view word, std::vector<std::int32_t>& ids) const {
    std::size_t start    = 0;
    std::size_t position = 0;
    while (position < word.size()) {
        const std::pair<std::string, std::int32_t>* match = nullptr;
        if (startsSpecialToken[static_cast<unsigned char>(word[position])]) {
            for (const auto& special : specialTokens) {
                if (word.compare(position, special.first.size(), special.first) == 0 &&
                    (match == nullptr || special.first.size() > match->first.size())) {
                    match = &special;
                }
            }
        }
        if (match == nullptr) {
            ++position;
            continue;
        }
        appendTextIds(word.substr(start, position - start), ids);
        ids.push_back(match->second);
        position += match->first.size();
        start = position;
    }
    appendTextIds(word.substr(start), ids);
}

// BertNormalizer and BertPreTokenizer on ASCII: clean_text turns \t, \n, and \r into spaces and
//   removes the other control characters, lowercase maps A-Z, and the pre-tokenizer splits at
//   spaces and isolates each punctuation character
void WordPieceTokenizer::appendTextIds(std::string_view text, std::vector<std::int32_t>& ids) const {
    std::string piece;
    for (const char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || isAsciiPunctuation(c)) {
            if (!piece.empty()) appendPieceIds(piece, ids);
            piece.clear();
            if (isAsciiPunctuation(c)) ids.push_back(punctuationIds[byte]);
        }
        else if (byte >= 0x20 && byte != 0x7F) {
            piece += isAsciiUpper(c) ? static_cast<char>(c - 'A' + 'a') : c;
        }
    }
    if (!piece.empty()) appendPieceIds(piece, ids);
}

// WordPiece: the longest vocabulary entry first, ## before a piece inside the word, and one
//   [UNK] for a word that is too long or has a piece outside the vocabulary
void WordPieceTokenizer::appendPieceIds(std::string_view piece, std::vector<std::int32_t>& ids) const {
    if (piece.size() > maxInputCharsPerWord) {
        ids.push_back(unknownId);
        return;
    }
    const std::size_t first = ids.size();
    std::string candidate;
    for (std::size_t start = 0; start < piece.size();) {
        // A candidate longer than the longest vocabulary entry never matches
        const std::size_t prefixLength = start > 0 ? continuingSubwordPrefix.size() : 0;
        std::size_t end = std::min(piece.size(), start + (longestEntry > prefixLength ? longestEntry - prefixLength : 0));
        std::optional<std::int32_t> found;
        for (; end > start; --end) {
            candidate.assign(start > 0 ? continuingSubwordPrefix : std::string_view());
            candidate.append(piece, start, end - start);
            if (const auto entry = vocabulary.find(candidate); entry != vocabulary.end()) {
                found = entry->second;
                break;
            }
        }
        if (!found) {
            ids.resize(first);
            ids.push_back(unknownId);
            return;
        }
        ids.push_back(*found);
        start = end;
    }
}

} // namespace scalar
