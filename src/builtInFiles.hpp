// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file builtInFiles.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

// The files that scalar_embed_files of CMakeLists.txt puts into the binary of a program: the model files and
// the NLTK words that CMake downloads, the Ronin tables, and the files that the program adds.
// cmake/embedFiles.cmake writes the definition of registerBuiltInFiles to the build folder.

#ifndef BUILT_IN_FILES_HPP
#define BUILT_IN_FILES_HPP

// Makes each built-in file readable through scalar::openDataFile under a path such as
//  builtin:model/config.json
void registerBuiltInFiles();

#endif
