// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file safetensors.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Reads the F32 tensors of a safetensors file after it validates the header of the file.

#ifndef SCALAR_SAFETENSORS_HPP
#define SCALAR_SAFETENSORS_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace scalar {

// The dtype, the shape, and the byte range of one tensor; the range is relative to the data section.
struct TensorEntry {
    std::string              dtype;
    std::vector<std::size_t> shape;
    std::uint64_t            begin = 0;
    std::uint64_t            end   = 0;
};

// A safetensors file with a validated header. The constructor checks the header length, the JSON
// syntax, unique keys, the known dtypes, the shapes against the byte ranges, and that the byte ranges
// tile the data section without a gap or an overlap. Tensor values are read on request, in place for
// a built-in file and from disk otherwise.
class SafetensorsFile {
public:
    // Throws std::runtime_error when the file cannot be read or breaks the format.
    explicit SafetensorsFile(const std::string& path);

    // Names of the tensors in sorted order, without the __metadata__ entry.
    std::vector<std::string> tensorNames() const;

    // The entry of the named tensor. Throws std::runtime_error when the file has no such tensor.
    const TensorEntry& entry(const std::string& name) const;

    // Values of the named tensor in row-major order. Throws std::runtime_error unless the tensor
    // has the dtype F32 and exactly the expected shape.
    std::vector<float> readF32(const std::string& name, const std::vector<std::size_t>& expectedShape) const;

private:
    // The bytes at the offset, in place for a built-in file or read from disk into the buffer, or
    // nullptr when the read fails
    const unsigned char* bytesAt(std::uint64_t offset, std::uint64_t count, std::vector<unsigned char>& buffer) const;

    std::string                        path;
    std::optional<std::string_view>    builtInBytes;
    std::uint64_t                      dataStart = 0;
    std::map<std::string, TensorEntry> entries;
};

}

#endif
