// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file safetensors.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Validates the header of a safetensors file and reads its F32 tensors in little-endian byte order.

#include "safetensors.hpp"

#include "dataFile.hpp"
#include "json.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace scalar {

namespace {

// The safetensors library refuses larger headers, which keeps a corrupt length from allocating memory
constexpr std::uint64_t MAX_HEADER_BYTES = 100000000;

// Bytes per element of each dtype of the safetensors format; 0 marks an unknown dtype
std::size_t dtypeBytes(const std::string& dtype) {
    static const std::map<std::string, std::size_t> sizes = {
        {"BOOL", 1}, {"U8", 1},  {"I8", 1},  {"F8_E5M2", 1}, {"F8_E4M3", 1},
        {"I16", 2},  {"U16", 2}, {"F16", 2}, {"BF16", 2},
        {"I32", 4},  {"U32", 4}, {"F32", 4},
        {"I64", 8},  {"U64", 8}, {"F64", 8},
    };
    const auto found = sizes.find(dtype);
    return found == sizes.end() ? 0 : found->second;
}

[[noreturn]] void fail(const std::string& path, const std::string& reason) {
    throw std::runtime_error("invalid safetensors file '" + path + "': " + reason);
}

std::uint64_t littleEndian64(const unsigned char* bytes) {
    std::uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | bytes[i];
    }
    return value;
}

std::string shapeText(const std::vector<std::size_t>& shape) {
    std::string text = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        text += (i ? ", " : "") + std::to_string(shape[i]);
    }
    return text + "]";
}

// Parses the header and rejects a repeated key in any object, which the JSON library would overwrite
nlohmann::json parseHeader(const std::string& path, const std::string& header) {
    std::vector<std::set<std::string>> openObjects;
    const nlohmann::json::parser_callback_t checkKeys =
        [&](int, nlohmann::json::parse_event_t event, nlohmann::json& parsed) {
            if (event == nlohmann::json::parse_event_t::object_start) {
                openObjects.emplace_back();
            } else if (event == nlohmann::json::parse_event_t::object_end) {
                openObjects.pop_back();
            } else if (event == nlohmann::json::parse_event_t::key) {
                if (!openObjects.back().insert(parsed.get<std::string>()).second) {
                    fail(path, "the header repeats the key '" + parsed.get<std::string>() + "'");
                }
            }
            return true;
        };
    try {
        return nlohmann::json::parse(header, checkKeys);
    } catch (const nlohmann::json::exception& error) {
        fail(path, std::string("the header is no valid JSON (") + error.what() + ")");
    }
}

std::uint64_t unsignedValue(const std::string& path, const nlohmann::json& value, const std::string& what) {
    if (!value.is_number_unsigned()) {
        fail(path, what + " is no non-negative integer");
    }
    return value.get<std::uint64_t>();
}

TensorEntry parseEntry(const std::string& path, const std::string& name, const nlohmann::json& value) {
    if (!value.is_object() || value.size() != 3 || !value.contains("dtype") || !value.contains("shape") ||
        !value.contains("data_offsets")) {
        fail(path, "tensor '" + name + "' needs exactly the fields dtype, shape, and data_offsets");
    }
    TensorEntry entry;
    if (!value["dtype"].is_string() || dtypeBytes(value["dtype"].get<std::string>()) == 0) {
        fail(path, "tensor '" + name + "' has an unknown dtype");
    }
    entry.dtype = value["dtype"].get<std::string>();

    const nlohmann::json& shape = value["shape"];
    if (!shape.is_array()) {
        fail(path, "the shape of tensor '" + name + "' is no array");
    }
    std::uint64_t elements = 1;
    for (const nlohmann::json& dimension : shape) {
        const std::uint64_t size = unsignedValue(path, dimension, "a dimension of tensor '" + name + "'");
        if (size > std::numeric_limits<std::size_t>::max() ||
            (size != 0 && elements > std::numeric_limits<std::uint64_t>::max() / size)) {
            fail(path, "the shape of tensor '" + name + "' overflows");
        }
        elements *= size;
        entry.shape.push_back(static_cast<std::size_t>(size));
    }

    const nlohmann::json& offsets = value["data_offsets"];
    if (!offsets.is_array() || offsets.size() != 2) {
        fail(path, "the data_offsets of tensor '" + name + "' are no pair");
    }
    entry.begin = unsignedValue(path, offsets[0], "the first data offset of tensor '" + name + "'");
    entry.end   = unsignedValue(path, offsets[1], "the second data offset of tensor '" + name + "'");
    if (entry.end < entry.begin) {
        fail(path, "the data_offsets of tensor '" + name + "' end before they begin");
    }
    const std::uint64_t elementBytes = dtypeBytes(entry.dtype);
    if (elements > std::numeric_limits<std::uint64_t>::max() / elementBytes ||
        entry.end - entry.begin != elements * elementBytes) {
        fail(path, "the byte range of tensor '" + name + "' does not fit its shape " + shapeText(entry.shape) +
                       " and dtype " + entry.dtype);
    }
    return entry;
}

}

SafetensorsFile::SafetensorsFile(const std::string& filePath) : path(filePath) {
    std::uint64_t fileBytes = 0;
    if (isBuiltInPath(path)) {
        const std::optional<DataFile> file = openDataFile(path);
        if (!file) {
            fail(path, "the binary holds no file of this name");
        }
        builtInBytes = file->bytes();
        fileBytes = builtInBytes->size();
    } else {
        std::error_code error;
        fileBytes = std::filesystem::file_size(path, error);
        if (error) {
            fail(path, "the file cannot be read (" + error.message() + ")");
        }
    }
    std::vector<unsigned char> buffer;
    const unsigned char* lengthBytes = fileBytes < 8 ? nullptr : bytesAt(0, 8, buffer);
    if (!lengthBytes) {
        fail(path, "the file is shorter than its 8-byte header length");
    }
    const std::uint64_t headerBytes = littleEndian64(lengthBytes);
    if (headerBytes > MAX_HEADER_BYTES || headerBytes > fileBytes - 8) {
        fail(path, "the header length " + std::to_string(headerBytes) + " exceeds the file or the limit");
    }
    const unsigned char* headerStart = bytesAt(8, headerBytes, buffer);
    if (!headerStart) {
        fail(path, "the header cannot be read");
    }
    const std::string header(reinterpret_cast<const char*>(headerStart), static_cast<std::size_t>(headerBytes));
    if (header.empty() || header.front() != '{') {
        fail(path, "the header does not start with '{'");
    }
    const nlohmann::json root = parseHeader(path, header);
    if (!root.is_object()) {
        fail(path, "the header is no JSON object");
    }

    for (const auto& item : root.items()) {
        if (item.key() == "__metadata__") {
            if (!item.value().is_object()) {
                fail(path, "__metadata__ is no object");
            }
            for (const auto& field : item.value().items()) {
                if (!field.value().is_string()) {
                    fail(path, "the __metadata__ value of '" + field.key() + "' is no string");
                }
            }
            continue;
        }
        entries.emplace(item.key(), parseEntry(path, item.key(), item.value()));
    }

    // The byte ranges must tile the data section, as the format forbids gaps and overlaps
    dataStart = 8 + headerBytes;
    const std::uint64_t dataBytes = fileBytes - dataStart;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& item : entries) {
        ranges.emplace_back(item.second.begin, item.second.end);
    }
    std::sort(ranges.begin(), ranges.end());
    std::uint64_t covered = 0;
    for (const auto& range : ranges) {
        if (range.first != covered) {
            fail(path, "the tensor data has a gap or an overlap at byte " + std::to_string(range.first));
        }
        covered = range.second;
    }
    if (covered != dataBytes) {
        fail(path, "the tensors cover " + std::to_string(covered) + " of the " + std::to_string(dataBytes) +
                       " bytes of the data section");
    }
}

std::vector<std::string> SafetensorsFile::tensorNames() const {
    std::vector<std::string> names;
    for (const auto& item : entries) {
        names.push_back(item.first);
    }
    return names;
}

const TensorEntry& SafetensorsFile::entry(const std::string& name) const {
    const auto found = entries.find(name);
    if (found == entries.end()) {
        fail(path, "it has no tensor '" + name + "'");
    }
    return found->second;
}

std::vector<float> SafetensorsFile::readF32(const std::string& name, const std::vector<std::size_t>& expectedShape) const {
    const TensorEntry& tensor = entry(name);
    if (tensor.dtype != "F32") {
        fail(path, "tensor '" + name + "' has the dtype " + tensor.dtype + " in place of F32");
    }
    if (tensor.shape != expectedShape) {
        fail(path, "tensor '" + name + "' has the shape " + shapeText(tensor.shape) + " in place of " +
                       shapeText(expectedShape));
    }
    std::vector<unsigned char> buffer;
    const unsigned char* bytes = bytesAt(dataStart + tensor.begin, tensor.end - tensor.begin, buffer);
    if (!bytes) {
        fail(path, "the data of tensor '" + name + "' cannot be read");
    }
    std::vector<float> values(static_cast<std::size_t>((tensor.end - tensor.begin) / 4));
    for (std::size_t i = 0; i < values.size(); ++i) {
        const unsigned char* b = bytes + 4 * i;
        const std::uint32_t bits = static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8) |
                                   (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
        std::memcpy(&values[i], &bits, sizeof(float));
    }
    return values;
}

const unsigned char* SafetensorsFile::bytesAt(std::uint64_t offset, std::uint64_t count, std::vector<unsigned char>& buffer) const {
    if (builtInBytes) {
        // The constructor checked the header against the size of the file and that the byte ranges of
        // the tensors tile its data section
        return reinterpret_cast<const unsigned char*>(builtInBytes->data()) + offset;
    }
    // One tensor at a time, since a copy of the whole file would double the memory of the model
    buffer.resize(static_cast<std::size_t>(count));
    std::ifstream file(path, std::ios::binary);
    file.seekg(static_cast<std::streamoff>(offset));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(count))) {
        return nullptr;
    }
    return buffer.data();
}

}
