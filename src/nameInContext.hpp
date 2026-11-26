// SPDX-License-Identifier: GPL-3.0-only
/**
 * @file nameInContext.hpp
 *
 * @copyright Copyright (C) 2021-2026 srcML, LLC. (www.srcML.org)
 *
 * This file is part of scalar-cpp.
 */

#ifndef SCALAR_NAME_IN_CONTEXT_HPP
#define SCALAR_NAME_IN_CONTEXT_HPP

#include <string>

namespace scalar {

// A name with the context of its category, such as FUNCTION or PARAMETER, and the other inputs that the
//   server reads from its query, which stay empty where unknown. The README of SCALAR says which context
//   and type to send for each kind of name. A new input of a newer SCALAR becomes one more member here,
//   which the feature functions of featureTokens.cpp read.
struct NameInContext {
    std::string name;
    std::string context;
    std::string type{};      // the declared type as the parser gives it, such as std::vector<Foo>, or the return type
    std::string language{};  // such as C++ or Java
    std::string system{};    // the project name, such as gimp
};

} // namespace scalar

#endif
