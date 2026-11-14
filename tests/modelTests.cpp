// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file modelTests.cpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// Checks the C++ SCALAR model against the golden files of the Python model; each call runs one named test.

#include "crfDecoder.hpp"
#include "dataFile.hpp"
#include "distilbertCrf.hpp"
#include "encoderKernels.hpp"
#include "goldenData.hpp"
#include "kernelDispatch.hpp"
#include "linearLayer.hpp"
#include "safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using scalar::DistilBertCrf;
using scalar::ModelInput;
using scalar::ReferenceRow;

struct TestPaths {
    std::string modelDirectory;
    std::string dataDirectory;
};

// Counts the failed checks of a test and prints the first of them
class Checker {
public:
    void check(bool condition, const std::string& message) {
        if (condition) {
            return;
        }
        if (++failures <= 20) {
            std::cout << "  FAILED: " << message << "\n";
        }
    }
    bool passed() const {
        if (failures > 20) {
            std::cout << "  ... " << failures << " failed checks in all\n";
        }
        return failures == 0;
    }

private:
    std::size_t failures = 0;
};

template <typename Exception, typename Action>
bool throws(Action action) {
    try {
        action();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

std::string labelText(const DistilBertCrf& model, const std::vector<int>& labels) {
    std::string text;
    for (const int label : labels) {
        text += (text.empty() ? "" : " ") + model.labelNames().at(static_cast<std::size_t>(label));
    }
    return text;
}

std::vector<ReferenceRow> goldenRows(const TestPaths& paths) {
    return scalar::readReferenceRows(paths.dataDirectory + "/golden_rows.tsv");
}

// Labels or emissions of all rows, computed by several threads that call decodeBatch or wordEmissions on their own chunks
template <typename Result, typename Compute>
std::vector<Result> computeInParallel(const std::vector<ModelInput>& inputs, std::size_t batchSize, Compute compute) {
    std::vector<Result> results(inputs.size());
    const std::size_t threadCount = std::max<std::size_t>(1, std::min<std::size_t>(8, std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < threadCount; ++t) {
        threads.emplace_back([&, t] {
            for (std::size_t start = t * batchSize; start < inputs.size(); start += threadCount * batchSize) {
                const std::size_t end = std::min(inputs.size(), start + batchSize);
                const std::vector<ModelInput> batch(inputs.begin() + static_cast<std::ptrdiff_t>(start),
                                                    inputs.begin() + static_cast<std::ptrdiff_t>(end));
                const std::vector<Result> batchResults = compute(batch);
                std::copy(batchResults.begin(), batchResults.end(), results.begin() + static_cast<std::ptrdiff_t>(start));
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    return results;
}

// A path in the temporary folder, with a number of its own for each run of the tests, therefore two
//  runs at once from two build folders never share a file
std::filesystem::path temporaryPath(const std::string& name) {
    static const std::string run = std::to_string(std::random_device{}());
    return std::filesystem::temp_directory_path() / ("scalarModelTests-" + run + "-" + name);
}

// The bytes of a safetensors file: the header length in eight little-endian bytes, the header, and the data
std::string safetensorsBytes(const std::string& header, const std::vector<unsigned char>& data, std::uint64_t headerLength) {
    std::string bytes;
    for (int i = 0; i < 8; ++i) {
        bytes += static_cast<char>((headerLength >> (8 * i)) & 0xff);
    }
    bytes += header;
    bytes.append(reinterpret_cast<const char*>(data.data()), data.size());
    return bytes;
}

// A file of the test, removed when the guard leaves scope
class TemporaryFile {
public:
    TemporaryFile(const std::string& name, const std::string& bytes)
        : path(temporaryPath(name).string()) {
        std::ofstream(path, std::ios::binary) << bytes;
    }
    TemporaryFile(const std::string& name, const std::string& header, const std::vector<unsigned char>& data,
                  std::uint64_t headerLength)
        : TemporaryFile(name, safetensorsBytes(header, data, headerLength)) {}
    TemporaryFile(const std::string& name, const std::string& header, const std::vector<unsigned char>& data)
        : TemporaryFile(name, header, data, header.size()) {}
    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile& operator=(const TemporaryFile&) = delete;
    ~TemporaryFile() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
    const std::string path;
};

std::vector<unsigned char> floatBytes(const std::vector<float>& values) {
    std::vector<unsigned char> bytes(values.size() * 4);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, &values[i], sizeof bits);
        for (int b = 0; b < 4; ++b) {
            bytes[4 * i + static_cast<std::size_t>(b)] = static_cast<unsigned char>((bits >> (8 * b)) & 0xff);
        }
    }
    return bytes;
}

bool dataFilesViewBuiltInFilesInPlace(const TestPaths&) {
    Checker checker;
    static const std::string text = "first line\nsecond line\n";
    const scalar::BuiltInFile file{"tests/lines.txt", text};
    scalar::addBuiltInFiles(&file, 1);
    const std::optional<scalar::DataFile> found = scalar::openDataFile("builtin:tests/lines.txt");
    checker.check(found && found->bytes().data() == text.data() && found->bytes().size() == text.size(),
                  "a built-in file is a view of its bytes");
    const std::optional<scalar::DataFile> joined = scalar::openDataFile("builtin:tests\\lines.txt");
    checker.check(joined && joined->bytes() == text, "a backslash, which std::filesystem::path joins with on Windows, counts as a slash");
    checker.check(!scalar::openDataFile("builtin:tests/missing.txt"), "a missing built-in file opens as no file");
    checker.check(scalar::dataFileExists("builtin:tests/lines.txt"), "a built-in file exists");
    checker.check(!scalar::dataFileExists("builtin:tests/missing.txt"), "a missing built-in file does not exist");
    return checker.passed();
}

bool dataFilesReadFilesOnDiskWhole(const TestPaths&) {
    Checker checker;
    // A zero byte and a carriage return, repeated past the buffer of a stream
    std::string bytes("a\0b\r\nc", 6);
    for (int i = 0; i < 16; ++i) {
        bytes += bytes;
    }
    std::string path;
    {
        const TemporaryFile file("dataFile.bin", bytes);
        path = file.path;
        const std::optional<scalar::DataFile> read = scalar::openDataFile(path);
        checker.check(read && read->bytes() == bytes, "a file on disk keeps every byte");
        checker.check(scalar::dataFileExists(path), "a file on disk exists");
    }
    checker.check(!scalar::openDataFile(path), "a removed file opens as no file");
    checker.check(!scalar::dataFileExists(path), "a removed file does not exist");
    checker.check(!scalar::openDataFile(std::filesystem::temp_directory_path().string()), "a folder opens as no file");
    return checker.passed();
}

// The bytes of a valid file: tensor a is F32 [2, 3], b is an F32 scalar, and c is I64 [2]
std::string validTensorsFile() {
    std::vector<unsigned char> data = floatBytes({1, 2, 3, 4, 5, 6, 7.5f});
    data.resize(data.size() + 16, 0);
    const std::string header = R"({"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
                               R"("b":{"dtype":"F32","shape":[],"data_offsets":[24,28]},)"
                               R"("c":{"dtype":"I64","shape":[2],"data_offsets":[28,44]}}  )";
    return safetensorsBytes(header, data, header.size());
}

void checkValidTensors(Checker& checker, const scalar::SafetensorsFile& tensors) {
    checker.check(tensors.tensorNames() == std::vector<std::string>{"a", "b", "c"}, "the names are a, b, and c");
    checker.check(tensors.entry("a").shape == std::vector<std::size_t>{2, 3}, "tensor a has the shape [2, 3]");
    checker.check(tensors.readF32("a", {2, 3}) == std::vector<float>{1, 2, 3, 4, 5, 6}, "tensor a holds 1 to 6");
    checker.check(tensors.readF32("b", {}) == std::vector<float>{7.5f}, "the scalar tensor b holds 7.5");
    checker.check(throws<std::runtime_error>([&] { tensors.readF32("c", {2}); }), "reading the I64 tensor as F32 throws");
    checker.check(throws<std::runtime_error>([&] { tensors.readF32("a", {3, 2}); }), "reading a with another shape throws");
    checker.check(throws<std::runtime_error>([&] { tensors.entry("d"); }), "a missing tensor throws");
}

bool safetensorsReadsValidTensors(const TestPaths&) {
    Checker checker;
    const TemporaryFile file("valid.safetensors", validTensorsFile());
    checkValidTensors(checker, scalar::SafetensorsFile(file.path));
    return checker.passed();
}

bool safetensorsReadsBuiltInTensors(const TestPaths&) {
    Checker checker;
    static const std::string bytes = validTensorsFile();
    const scalar::BuiltInFile file{"tests/valid.safetensors", bytes};
    scalar::addBuiltInFiles(&file, 1);
    checkValidTensors(checker, scalar::SafetensorsFile("builtin:tests/valid.safetensors"));
    checker.check(throws<std::runtime_error>([&] { scalar::SafetensorsFile tensors("builtin:tests/missing.safetensors"); }),
                  "a missing built-in file throws");
    return checker.passed();
}

bool safetensorsRejectsMalformedFiles(const TestPaths&) {
    Checker checker;
    const std::vector<unsigned char> data = floatBytes({1, 2, 3, 4, 5, 6});
    const std::string tensorA = R"("a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]})";
    const std::vector<std::pair<std::string, std::string>> headers = {
        {"no JSON", R"({"a": })"},
        {"leading space", " {" + tensorA + "}"},
        {"repeated key", "{" + tensorA + "," + tensorA + "}"},
        {"unknown dtype", R"({"a":{"dtype":"F33","shape":[2,3],"data_offsets":[0,24]}})"},
        {"shape larger than the range", R"({"a":{"dtype":"F32","shape":[2,4],"data_offsets":[0,24]}})"},
        {"range past the data", R"({"a":{"dtype":"F32","shape":[2,4],"data_offsets":[0,32]}})"},
        {"gap", R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[1],"data_offsets":[20,24]}})"},
        {"overlap", R"({"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"b":{"dtype":"F32","shape":[1],"data_offsets":[20,24]}})"},
        {"data beyond the tensors", R"({"a":{"dtype":"F32","shape":[5],"data_offsets":[0,20]}})"},
        {"negative offset", R"({"a":{"dtype":"F32","shape":[2,3],"data_offsets":[-4,20]}})"},
        {"end before begin", R"({"a":{"dtype":"F32","shape":[0],"data_offsets":[24,0]}})"},
        {"missing shape", R"({"a":{"dtype":"F32","data_offsets":[0,24]}})"},
        {"extra field", R"({"a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24],"extra":1}})"},
        {"fractional dimension", R"({"a":{"dtype":"F32","shape":[2.5,3],"data_offsets":[0,24]}})"},
        {"entry no object", R"({"a":5})"},
        {"metadata no strings", R"({"__metadata__":{"format":1},)" + tensorA + "}"},
    };
    for (std::size_t i = 0; i < headers.size(); ++i) {
        const TemporaryFile file("malformed" + std::to_string(i) + ".safetensors", headers[i].second, data);
        checker.check(throws<std::runtime_error>([&] { scalar::SafetensorsFile tensors(file.path); }),
                      "a header with " + headers[i].first + " throws");
    }
    const std::string valid = "{" + tensorA + "}";
    const TemporaryFile tooLong("lengthPastEnd.safetensors", valid, data, 4096);
    checker.check(throws<std::runtime_error>([&] { scalar::SafetensorsFile tensors(tooLong.path); }),
                  "a header length past the end of the file throws");
    const TemporaryFile huge("lengthHuge.safetensors", valid, data, std::uint64_t{1} << 62);
    checker.check(throws<std::runtime_error>([&] { scalar::SafetensorsFile tensors(huge.path); }),
                  "a header length above the limit throws");
    {
        const std::string path = temporaryPath("short.safetensors").string();
        std::ofstream(path, std::ios::binary) << "abcd";
        checker.check(throws<std::runtime_error>([&] { scalar::SafetensorsFile tensors(path); }),
                      "a file shorter than 8 bytes throws");
        std::filesystem::remove(path);
    }
    checker.check(throws<std::runtime_error>([&] { scalar::SafetensorsFile tensors("/nonexistent/model.safetensors"); }),
                  "a missing file throws");
    return checker.passed();
}

bool crfMatchesTorchcrfOnSyntheticEmissions(const TestPaths& paths) {
    Checker checker;
    const std::vector<scalar::CrfCaseSet> sets = scalar::readCrfCases(paths.dataDirectory + "/crf_cases.txt");
    std::size_t cases = 0;
    for (std::size_t s = 0; s < sets.size(); ++s) {
        for (std::size_t c = 0; c < sets[s].cases.size(); ++c) {
            const scalar::CrfCase& entry = sets[s].cases[c];
            ++cases;
            checker.check(entry.emissions.size() == entry.length * sets[s].decoder.tagCount(),
                          "CRF " + std::to_string(s) + " case " + std::to_string(c) + " has one emission per step and tag");
            checker.check(sets[s].decoder.decode(entry.emissions.data(), entry.length) == entry.path,
                          "CRF " + std::to_string(s) + " case " + std::to_string(c) + " gives the torchcrf path");
        }
    }
    std::cout << "  " << cases << " cases of " << sets.size() << " CRFs\n";
    checker.check(cases > 0, "the file holds cases");
    return checker.passed();
}

bool crfDecodesPythonEmissionsToPythonLabels(const TestPaths& paths) {
    Checker checker;
    const DistilBertCrf model(paths.modelDirectory);
    std::size_t rows = 0;
    for (const ReferenceRow& row : goldenRows(paths)) {
        if (row.tooLong) {
            continue;
        }
        ++rows;
        checker.check(model.crf().decode(row.emissions.data(), row.labels.size()) == row.labels,
                      "the Python emissions of '" + row.name + "' decode to the Python labels");
    }
    std::cout << "  " << rows << " rows\n";
    return checker.passed();
}

bool bitEqual(const std::vector<float>& left, const std::vector<float>& right) {
    return left.size() == right.size() &&
           (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0);
}

std::size_t countBitDifferences(const std::vector<float>& left, const std::vector<float>& right) {
    std::size_t differences = 0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        differences += std::memcmp(&left[i], &right[i], sizeof(float)) != 0;
    }
    return differences;
}

bool kernelsMatchTorchBitForBit(const TestPaths& paths) {
    Checker checker;
    for (const scalar::KernelCase& entry : scalar::readKernelCases(paths.dataDirectory + "/kernel_cases.txt")) {
        std::vector<float> result;
        const std::vector<float>* expected = &entry.arrays.back();
        if (entry.name == "gelu") {
            result = entry.arrays[0];
            scalar::geluInPlace(result.data(), result.size());
        } else if (entry.name == "layernorm") {
            result = entry.arrays[0];
            scalar::layerNormRows(result.data(), entry.sizes[0], entry.sizes[1], entry.arrays[1].data(),
                                  entry.arrays[2].data(), 1e-12f);
        } else if (entry.name == "linear") {
            const scalar::LinearLayer layer(entry.arrays[1], entry.arrays[2], entry.sizes[1]);
            checker.check(layer.inputCount() == entry.sizes[1] && layer.outputCount() == entry.sizes[2],
                          "the linear layer has the sizes of its weights");
            result.resize(entry.sizes[0] * layer.outputCount());
            layer.apply(entry.arrays[0].data(), entry.sizes[0], result.data());
        } else if (entry.name == "attention") {
            const std::size_t length = entry.sizes[0];
            const std::size_t width  = entry.sizes[1] * entry.sizes[2];
            std::vector<float> queryKeyValue(length * 3 * width);
            for (std::size_t t = 0; t < length; ++t) {
                for (std::size_t part = 0; part < 3; ++part) {
                    std::copy_n(entry.arrays[part].data() + t * width, width, queryKeyValue.data() + (t * 3 + part) * width);
                }
            }
            std::vector<std::size_t> queries(length);
            for (std::size_t t = 0; t < length; ++t) {
                queries[t] = t;
            }
            result.resize(length * width);
            scalar::attendIdentifier(queryKeyValue.data(), length, entry.sizes[1], entry.sizes[2], 0.125f, queries.data(),
                                     length, result.data());
        } else {
            checker.check(false, "unknown kernel case " + entry.name);
            continue;
        }
        const std::size_t differences = countBitDifferences(result, *expected);
        std::cout << "  " << entry.name << ": " << differences << " of " << expected->size() << " values differ\n";
        checker.check(result.size() == expected->size() && differences == 0, entry.name + " equals torch bit for bit");
    }
    return checker.passed();
}

std::uint32_t floatBits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

float floatWithBits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

// A float of random sign and mantissa whose biased exponent lies in [lowest, highest]
float randomFloat(std::mt19937& generator, std::uint32_t lowest, std::uint32_t highest) {
    const auto exponent = static_cast<std::uint32_t>(lowest + generator() % (highest - lowest + 1));
    return floatWithBits(static_cast<std::uint32_t>(generator() & 0x807FFFFFu) | (exponent << 23));
}

// One multiply-add of the emulation against std::fma; any NaN equals any NaN
class FmaComparison {
public:
    explicit FmaComparison(Checker& failures) : checks(failures) {}

    void compare(float a, float b, float c) {
        const float expected = std::fma(a, b, c);
        const float result   = scalar::emulatedFusedMultiplyAdd(a, b, c);
        // A double sum rounded once to float, the shortcut that the emulation must not take on a boundary
        const float doubleSum = static_cast<float>(static_cast<double>(a) * static_cast<double>(b) + static_cast<double>(c));
        ++cases;
        doubleSumWrong += !std::isnan(expected) && floatBits(doubleSum) != floatBits(expected);
        if (!(std::isnan(expected) && std::isnan(result)) && floatBits(expected) != floatBits(result)) {
            checks.check(false, "the emulated fma of " + hex(a) + ", " + hex(b) + ", and " + hex(c) + " gives " + hex(result) +
                                    " in place of " + hex(expected));
        }
    }

    std::size_t cases          = 0;
    std::size_t doubleSumWrong = 0;

private:
    static std::string hex(float value) {
        std::ostringstream text;
        text << std::hex << floatBits(value);
        return text.str();
    }

    Checker& checks;
};

bool emulatedFmaEqualsStdFmaOnHardCases(const TestPaths&) {
    Checker checker;
    FmaComparison comparison(checker);
    std::vector<float> special = {0.0f, 1.0f, 1.5f, 3.0f, 0.1f, 0x1.000002p0f, 0x1p-24f, 0x1p-126f, 0x1p-149f, 0x1.8p-148f,
                                  0x1.fffffcp-127f, 0x1.fffffep127f, 0x1p127f, 1e-30f, 3e38f,
                                  std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
    for (std::size_t i = 0, count = special.size(); i < count; ++i) {
        special.push_back(-special[i]);
    }
    for (const float a : special) {
        for (const float b : special) {
            for (const float c : special) {
                comparison.compare(a, b, c);
            }
        }
    }
    std::mt19937 generator(2026);
    for (int i = 0; i < 300000; ++i) {
        // Products of two 13-bit mantissas, many on a midpoint between two floats, plus an addend far
        // below the last bit of a double
        const float a = std::ldexp(static_cast<float>(4096 + generator() % 4096), static_cast<int>(generator() % 60) - 40);
        const float b = std::ldexp(static_cast<float>(4096 + generator() % 4096), static_cast<int>(generator() % 60) - 40);
        const double product = static_cast<double>(a) * static_cast<double>(b);
        const float tiny = static_cast<float>(std::ldexp(product, -30 - static_cast<int>(generator() % 40)));
        comparison.compare(a, b, (generator() & 1) != 0 ? tiny : -tiny);
        comparison.compare(a, b, (generator() & 1) != 0 ? 0x1p-149f : -0x1p-149f);
    }
    for (int i = 0; i < 300000; ++i) {
        // Sums next to the midpoint above the float nearest the product
        const float a = randomFloat(generator, 100, 154);
        const float b = randomFloat(generator, 100, 154);
        const double product  = static_cast<double>(a) * static_cast<double>(b);
        const float  nearest  = static_cast<float>(product);
        const double midpoint = (static_cast<double>(nearest) + std::nextafter(nearest, std::numeric_limits<float>::infinity())) / 2;
        comparison.compare(a, b, static_cast<float>(midpoint - product));
    }
    for (int i = 0; i < 1000000; ++i) {
        // Any exponents, with overflow, subnormal results, and cancellation
        comparison.compare(randomFloat(generator, 0, 254), randomFloat(generator, 0, 254), randomFloat(generator, 0, 254));
        comparison.compare(randomFloat(generator, 90, 164), randomFloat(generator, 90, 164), randomFloat(generator, 50, 200));
    }
    std::cout << "  " << comparison.cases << " cases, " << comparison.doubleSumWrong << " where one rounding of the double sum fails\n";
    checker.check(comparison.doubleSumWrong > 1000, "the cases hold many sums that a double sum rounds wrong");
    return checker.passed();
}

// The order of LinearLayer: each output starts from its bias and adds the FMA chain of each run of 256 inputs
std::vector<float> chainOrderProduct(const std::vector<float>& input, std::size_t rows, const std::vector<float>& weights,
                                     const std::vector<float>& bias, std::size_t inputs) {
    const std::size_t outputs = bias.size();
    std::vector<float> result(rows * outputs);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t out = 0; out < outputs; ++out) {
            float total = bias[out];
            for (std::size_t first = 0; first < inputs; first += 256) {
                float chain = 0.0f;
                for (std::size_t k = first; k < std::min(inputs, first + 256); ++k) {
                    chain = std::fma(input[row * inputs + k], weights[out * inputs + k], chain);
                }
                total += chain;
            }
            result[row * outputs + out] = total;
        }
    }
    return result;
}

bool linearLayerFollowsTheChainOrderOnEveryShape(const TestPaths&) {
    Checker checker;
    std::mt19937 generator(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const std::size_t rowCounts[]   = {1, 2, 3, 4, 5, 6, 7, 9, 13, 200};
    const std::size_t inputCounts[] = {1, 7, 256, 257, 600};
    const std::size_t outputCounts[] = {1, 5, 16, 21, 33};
    std::size_t shapes = 0;
    for (const std::size_t rows : rowCounts) {
        for (const std::size_t inputs : inputCounts) {
            for (const std::size_t outputs : outputCounts) {
                std::vector<float> weights(outputs * inputs);
                std::vector<float> bias(outputs);
                std::vector<float> input(rows * inputs);
                for (std::vector<float>* values : {&weights, &bias, &input}) {
                    std::generate(values->begin(), values->end(), [&] { return normal(generator); });
                }
                const scalar::LinearLayer layer(weights, bias, inputs);
                std::vector<float> output(rows * outputs);
                layer.apply(input.data(), rows, output.data());
                ++shapes;
                checker.check(bitEqual(output, chainOrderProduct(input, rows, weights, bias, inputs)),
                              std::to_string(rows) + " rows of " + std::to_string(inputs) + " inputs and " +
                                  std::to_string(outputs) + " outputs follow the chain order");
            }
        }
    }
    std::cout << "  " << shapes << " shapes\n";
    return checker.passed();
}

bool modelRejectsInvalidInputs(const TestPaths& paths) {
    Checker checker;
    checker.check(throws<std::runtime_error>([] { DistilBertCrf model("/nonexistent"); }), "a missing model folder throws");
    {
        const std::filesystem::path folder = temporaryPath("config");
        std::filesystem::create_directories(folder);
        std::ofstream(folder / "config.json") << R"({"dim": 768, "n_heads": 12})";
        checker.check(throws<std::runtime_error>([&] { DistilBertCrf model(folder.string()); }),
                      "a config without id2label throws");
        std::ofstream(folder / "config.json") << R"({"dim": 768, "n_heads": 7, "id2label": {"0": "N"}})";
        checker.check(throws<std::runtime_error>([&] { DistilBertCrf model(folder.string()); }),
                      "a config whose heads do not divide dim throws");
        std::filesystem::remove_all(folder);
    }
    const DistilBertCrf model(paths.modelDirectory);
    const std::int32_t cls = 101, sep = 102, word = 2131;
    const std::vector<std::pair<std::string, ModelInput>> invalid = {
        {"two tokens", {{cls, sep}, {}}},
        {"more tokens than the position embeddings", {std::vector<std::int32_t>(model.maxInputLength() + 1, word), {1}}},
        {"a negative id", {{cls, -1, sep}, {1}}},
        {"an id past the vocabulary", {{cls, static_cast<std::int32_t>(model.vocabularySize()), sep}, {1}}},
        {"a word at [CLS]", {{cls, word, word, sep}, {0}}},
        {"a word at [SEP]", {{cls, word, word, sep}, {3}}},
        {"a repeated word position", {{cls, word, word, sep}, {1, 1}}},
        {"decreasing word positions", {{cls, word, word, sep}, {2, 1}}},
    };
    for (const auto& item : invalid) {
        checker.check(throws<std::invalid_argument>([&] { model.decode(item.second.inputIds, item.second.wordPositions); }),
                      "decode throws for " + item.first);
        checker.check(throws<std::invalid_argument>([&] { model.decodeBatch({{{cls, word, sep}, {1}}, item.second}); }),
                      "decodeBatch throws for " + item.first);
    }
    checker.check(model.decode(std::vector<std::int32_t>(model.maxInputLength(), word), {1, model.maxInputLength() - 2}).size() == 2,
                  "as many tokens as the position embeddings allow decode");
    return checker.passed();
}

bool modelReturnsNoLabelsWithoutWords(const TestPaths& paths) {
    Checker checker;
    const DistilBertCrf model(paths.modelDirectory);
    checker.check(model.decode({101, 1030, 102}, {}).empty(), "an input without words gives no labels");
    std::size_t rows = 0;
    for (const ReferenceRow& row : goldenRows(paths)) {
        if (!row.tooLong && row.input.wordPositions.empty()) {
            ++rows;
            checker.check(model.decode(row.input.inputIds, row.input.wordPositions).empty(),
                          "'" + row.name + "' without words gives no labels");
        }
    }
    const std::vector<std::vector<int>> mixed = model.decodeBatch({{{101, 1030, 102}, {}}, {{101, 2131, 102}, {1}}});
    checker.check(mixed.size() == 2 && mixed[0].empty() && mixed[1].size() == 1,
                  "decodeBatch keeps an input without words in its place");
    std::cout << "  " << rows << " golden rows without words\n";
    return checker.passed();
}

bool modelGivesPythonLabelsOnGoldenRows(const TestPaths& paths) {
    Checker checker;
    const DistilBertCrf model(paths.modelDirectory);
    std::vector<ReferenceRow> rows = goldenRows(paths);
    rows.erase(std::remove_if(rows.begin(), rows.end(), [](const ReferenceRow& row) { return row.tooLong; }), rows.end());
    std::vector<ModelInput> inputs;
    for (const ReferenceRow& row : rows) {
        inputs.push_back(row.input);
    }
    const std::vector<std::vector<int>> labels = computeInParallel<std::vector<int>>(
        inputs, 16, [&](const std::vector<ModelInput>& batch) { return model.decodeBatch(batch); });
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const bool equal = labels[i] == rows[i].labels;
        mismatches += !equal;
        checker.check(equal, "'" + rows[i].name + "' (" + rows[i].context + ") gets " + labelText(model, labels[i]) +
                                 " in place of " + labelText(model, rows[i].labels));
    }
    std::cout << "  " << rows.size() << " rows, " << mismatches << " with other labels\n";
    return checker.passed();
}

bool modelEmissionsStayCloseToPython(const TestPaths& paths) {
    Checker checker;
    // Well above the differences of the summation orders of MKL that the port does not follow, and far
    // below the Viterbi margins of the corpus
    constexpr double TOLERANCE = 1e-4;
    const DistilBertCrf model(paths.modelDirectory);
    std::vector<ReferenceRow> rows = goldenRows(paths);
    rows.erase(std::remove_if(rows.begin(), rows.end(), [](const ReferenceRow& row) { return row.tooLong; }), rows.end());
    std::vector<ModelInput> inputs;
    for (const ReferenceRow& row : rows) {
        inputs.push_back(row.input);
    }
    const std::vector<std::vector<float>> emissions = computeInParallel<std::vector<float>>(
        inputs, 16, [&](const std::vector<ModelInput>& batch) { return model.wordEmissions(batch); });
    double largest = 0.0;
    std::size_t values = 0, equalValues = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) {
        checker.check(emissions[i].size() == rows[i].emissions.size(), "'" + rows[i].name + "' has one emission per word and label");
        for (std::size_t v = 0; v < std::min(emissions[i].size(), rows[i].emissions.size()); ++v) {
            const double difference = std::fabs(static_cast<double>(emissions[i][v]) - rows[i].emissions[v]);
            largest = std::max(largest, difference);
            ++values;
            equalValues += emissions[i][v] == rows[i].emissions[v];
            checker.check(difference <= TOLERANCE, "an emission of '" + rows[i].name + "' differs by " + std::to_string(difference));
        }
    }
    std::cout << "  " << values << " emissions, " << equalValues << " equal to Python, largest difference " << largest << "\n";
    return checker.passed();
}

bool decodeBatchEqualsDecodeBitForBit(const TestPaths& paths) {
    Checker checker;
    const DistilBertCrf model(paths.modelDirectory);
    std::vector<ModelInput> inputs;
    for (const ReferenceRow& row : goldenRows(paths)) {
        if (!row.tooLong && inputs.size() < 48) {
            inputs.push_back(row.input);
        }
    }
    inputs.insert(inputs.begin() + 7, ModelInput{{101, 1030, 102}, {}});
    std::vector<std::vector<float>> alone;
    for (const ModelInput& input : inputs) {
        alone.push_back(model.wordEmissions({input}).front());
    }
    const std::vector<std::vector<float>> together = model.wordEmissions(inputs);
    std::vector<ModelInput> reversed(inputs.rbegin(), inputs.rend());
    const std::vector<std::vector<float>> backwards = model.wordEmissions(reversed);
    const std::vector<std::vector<float>> parallel = computeInParallel<std::vector<float>>(
        inputs, 5, [&](const std::vector<ModelInput>& batch) { return model.wordEmissions(batch); });
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const std::string which = "input " + std::to_string(i);
        checker.check(bitEqual(together[i], alone[i]), which + " in one batch equals the input alone");
        checker.check(bitEqual(backwards[inputs.size() - 1 - i], alone[i]), which + " in the reversed batch equals the input alone");
        checker.check(bitEqual(parallel[i], alone[i]), which + " in batches of 5 on several threads equals the input alone");
    }
    const std::vector<std::vector<int>> batchLabels = model.decodeBatch(inputs);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        checker.check(batchLabels[i] == model.decode(inputs[i].inputIds, inputs[i].wordPositions),
                      "decodeBatch gives input " + std::to_string(i) + " the labels of decode");
    }
    std::cout << "  " << inputs.size() << " inputs\n";
    return checker.passed();
}

}

int main(int argc, char* argv[]) {
    const std::map<std::string, std::function<bool(const TestPaths&)>> tests = {
        {"dataFilesViewBuiltInFilesInPlace", dataFilesViewBuiltInFilesInPlace},
        {"dataFilesReadFilesOnDiskWhole", dataFilesReadFilesOnDiskWhole},
        {"safetensorsReadsValidTensors", safetensorsReadsValidTensors},
        {"safetensorsReadsBuiltInTensors", safetensorsReadsBuiltInTensors},
        {"safetensorsRejectsMalformedFiles", safetensorsRejectsMalformedFiles},
        {"crfMatchesTorchcrfOnSyntheticEmissions", crfMatchesTorchcrfOnSyntheticEmissions},
        {"crfDecodesPythonEmissionsToPythonLabels", crfDecodesPythonEmissionsToPythonLabels},
        {"kernelsMatchTorchBitForBit", kernelsMatchTorchBitForBit},
        {"emulatedFmaEqualsStdFmaOnHardCases", emulatedFmaEqualsStdFmaOnHardCases},
        {"linearLayerFollowsTheChainOrderOnEveryShape", linearLayerFollowsTheChainOrderOnEveryShape},
        {"modelRejectsInvalidInputs", modelRejectsInvalidInputs},
        {"modelReturnsNoLabelsWithoutWords", modelReturnsNoLabelsWithoutWords},
        {"modelGivesPythonLabelsOnGoldenRows", modelGivesPythonLabelsOnGoldenRows},
        {"modelEmissionsStayCloseToPython", modelEmissionsStayCloseToPython},
        {"decodeBatchEqualsDecodeBitForBit", decodeBatchEqualsDecodeBitForBit},
    };
    if (argc != 4 || tests.count(argv[1]) == 0) {
        std::cerr << "usage: scalarModelTests <test> <model folder> <golden data folder>\ntests:";
        for (const auto& test : tests) {
            std::cerr << " " << test.first;
        }
        std::cerr << "\n";
        return 2;
    }
    try {
        const bool passed = tests.at(argv[1])({argv[2], argv[3]});
        std::cout << (passed ? "PASS " : "FAIL ") << argv[1] << "\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "FAIL " << argv[1] << ": " << error.what() << "\n";
        return 1;
    }
}
