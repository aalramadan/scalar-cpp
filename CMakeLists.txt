# @file CMakeLists.txt
# CMake files for scalar-cpp, a C++ port of the SCALAR part-of-speech tagger for identifier names

cmake_minimum_required(VERSION 3.28)
project(scalar-cpp VERSION 2.0.0 LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(SCALAR_BUILD_TESTS "Build the tests and tools/scalarTagCorpus" ${PROJECT_IS_TOP_LEVEL})

find_package(Threads REQUIRED)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/scalarRelease.cmake")

# The downloads go to the cache folder of the user, therefore a new build folder downloads nothing
if(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
    set(SCALAR_CACHE_BASE "$ENV{XDG_CACHE_HOME}")
elseif(WIN32 AND DEFINED ENV{LOCALAPPDATA})
    set(SCALAR_CACHE_BASE "$ENV{LOCALAPPDATA}")
else()
    set(SCALAR_CACHE_BASE "$ENV{HOME}/.cache")
endif()
string(SUBSTRING "${SCALAR_MODEL_REVISION}" 0 7 SCALAR_SHORT_REVISION)

# Forward slashes, since the folder also becomes a string literal of the C++ code. Unlike
# file(TO_CMAKE_PATH), cmake_path keeps a colon of a path on Linux and macOS.
cmake_path(SET SCALAR_DATA_PATH NORMALIZE "${SCALAR_CACHE_BASE}/scalar-cpp/model-${SCALAR_SHORT_REVISION}")
cmake_path(SET SCALAR_RONIN_PATH NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/data/ronin")

# Downloads a file unless a copy with the expected SHA-256 is already in place
function(scalar_fetch url destination sha256)
    if(EXISTS "${destination}")
        file(SHA256 "${destination}" present)
        if(present STREQUAL sha256)
            return()
        endif()
    endif()
    message(STATUS "Downloading ${url}")
    # The hash is checked after the download, since EXPECTED_HASH turns a failed download into an
    # error of its own that skips the message below
    file(DOWNLOAD "${url}" "${destination}.part" TLS_VERIFY ON STATUS status SHOW_PROGRESS)
    list(GET status 0 code)
    list(GET status 1 reason)
    if(code EQUAL 0)
        file(SHA256 "${destination}.part" received)
        if(NOT received STREQUAL sha256)
            set(code 1)
            set(reason "SHA-256 ${received} instead of ${sha256}")
        endif()
    endif()
    if(NOT code EQUAL 0)
        file(REMOVE "${destination}.part")
        message(FATAL_ERROR "Could not download ${url} (${reason}).\n"
            "Check the network connection and run cmake -B build again.")
    endif()
    file(RENAME "${destination}.part" "${destination}")
endfunction()

file(MAKE_DIRECTORY "${SCALAR_DATA_PATH}/nltk")
# Two first configures at once, such as of two build folders, share the folder: one downloads, the
# other waits and finds the files
file(LOCK "${SCALAR_DATA_PATH}/download.lock" GUARD PROCESS TIMEOUT 600)
set(files ${SCALAR_MODEL_FILES})
while(files)
    list(POP_FRONT files name sha256)
    scalar_fetch("https://huggingface.co/${SCALAR_MODEL_REPO}/resolve/${SCALAR_MODEL_REVISION}/${name}"
                 "${SCALAR_DATA_PATH}/${name}" ${sha256})
endwhile()
scalar_fetch("${NLTK_WORDS_URL}" "${SCALAR_DATA_PATH}/nltk/words.zip" ${NLTK_WORDS_SHA256})
if(NOT EXISTS "${SCALAR_DATA_PATH}/nltk/words/en" OR NOT EXISTS "${SCALAR_DATA_PATH}/nltk/words/en-basic")
    file(ARCHIVE_EXTRACT INPUT "${SCALAR_DATA_PATH}/nltk/words.zip" DESTINATION "${SCALAR_DATA_PATH}/nltk"
         PATTERNS words/en words/en-basic)
endif()
file(LOCK "${SCALAR_DATA_PATH}/download.lock" RELEASE)

# The files that the tagger reads, which scalar_embed_files puts into a binary: the name that the code
#   reads each file by, then the file
set(SCALAR_BUILT_IN_FILES)
set(files ${SCALAR_MODEL_FILES})
while(files)
    list(POP_FRONT files name sha256)
    list(APPEND SCALAR_BUILT_IN_FILES "model/${name}" "${SCALAR_DATA_PATH}/${name}")
endwhile()
foreach(name IN ITEMS nltk/words/en nltk/words/en-basic)
    list(APPEND SCALAR_BUILT_IN_FILES "model/${name}" "${SCALAR_DATA_PATH}/${name}")
endforeach()
# A file that disappears from the cache folder makes the next build configure again, which downloads it
set(files ${SCALAR_BUILT_IN_FILES})
while(files)
    list(POP_FRONT files name path)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
endwhile()
file(GLOB SCALAR_RONIN_TABLES CONFIGURE_DEPENDS RELATIVE "${SCALAR_RONIN_PATH}" "${SCALAR_RONIN_PATH}/*.txt")
foreach(name IN LISTS SCALAR_RONIN_TABLES)
    list(APPEND SCALAR_BUILT_IN_FILES "ronin/${name}" "${SCALAR_RONIN_PATH}/${name}")
endforeach()
set_property(GLOBAL PROPERTY SCALAR_BUILT_IN_FILES "${SCALAR_BUILT_IN_FILES}")
set_property(GLOBAL PROPERTY SCALAR_EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embedFiles.cmake")

# scalar_embed_files(<target> [<name> <file>]...) puts the files of the tagger and the given files into
#   the binary of the target. The program then calls registerBuiltInFiles() of builtInFiles.hpp once and
#   loads the tagger from scalar::builtInFiles(), therefore the binary runs from any folder.
function(scalar_embed_files target)
    get_property(files GLOBAL PROPERTY SCALAR_BUILT_IN_FILES)
    get_property(script GLOBAL PROPERTY SCALAR_EMBED_SCRIPT)
    list(APPEND files ${ARGN})
    set(paths)
    set(rest ${files})
    while(rest)
        list(POP_FRONT rest name path)
        list(APPEND paths "${path}")
    endwhile()
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${target}BuiltInFiles")
    # The list goes to embedFiles.cmake through a file, which keeps the command line short
    file(CONFIGURE OUTPUT "${output}.list" CONTENT "set(BUILT_IN_FILES [==[@files@]==])\n" @ONLY)
    set(sources "${output}.cpp")
    # MSVC has no .incbin, therefore its build puts the files into a resource script
    if(MSVC)
        list(APPEND sources "${output}.rc")
    endif()
    add_custom_command(
        OUTPUT ${sources}
        COMMAND "${CMAKE_COMMAND}" "-DLIST=${output}.list" "-DOUTPUT=${output}" "-DRESOURCES=$<BOOL:${MSVC}>" -P "${script}"
        DEPENDS "${script}" "${output}.list" ${paths}
        COMMENT "Building the model files of SCALAR and the other data files into ${target}"
        VERBATIM
    )
    target_sources(${target} PRIVATE ${sources})
endfunction()

# The tagger library
file(GLOB SCALAR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)
add_library(scalar STATIC ${SCALAR_SOURCES})
add_library(scalar::scalar ALIAS scalar)
target_compile_features(scalar PUBLIC cxx_std_17)
target_include_directories(scalar
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PRIVATE "${PROJECT_SOURCE_DIR}/include"
)
target_compile_definitions(scalar PUBLIC
    "SCALAR_MODEL_REVISION=\"${SCALAR_MODEL_REVISION}\""
)
target_link_libraries(scalar PUBLIC Threads::Threads)
# The sources hold UTF-8 text, which MSVC reads only with /utf-8
set(SCALAR_WARNINGS
    "$<$<CXX_COMPILER_ID:MSVC>:/W4;/permissive-;/utf-8>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wunused>"
    "$<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Wshadow-all;-Wunused>"
)
target_compile_options(scalar PRIVATE ${SCALAR_WARNINGS})

# The kernels of the model round every product and every sum where torch does, and they fuse a
# multiply-add only through std::fma or the FMA intrinsic, never through the compiler. Like the build
# of torch, they ignore floating-point traps and errno. The kernels pick AVX2 and FMA at run time,
# therefore the build needs no flag for the CPU.
target_compile_options(scalar PRIVATE
    "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off;-fno-trapping-math;-fno-math-errno>"
)

# Builds the plain C++ kernels that Arm, MSVC, and CPUs without AVX2 run, for a test of these kernels
# on any machine. They give the same tags about 60 times slower.
option(SCALAR_PORTABLE_KERNELS "Build the tagger without its AVX2 kernels" OFF)
if(SCALAR_PORTABLE_KERNELS)
    target_compile_definitions(scalar PRIVATE SCALAR_PORTABLE_KERNELS)
endif()

if(SCALAR_BUILD_TESTS)
    # The tests and the corpus tool read the downloaded files and the Ronin tables in their folders
    set(SCALAR_FOLDER_DEFINITIONS
        "SCALAR_DATA_DIR=\"${SCALAR_DATA_PATH}\""
        "SCALAR_RONIN_DIR=\"${SCALAR_RONIN_PATH}\""
    )

    # Tags the names of a corpus, such as for the comparison with the Python SCALAR after an upgrade (PORT.md)
    add_executable(scalarTagCorpus tools/scalarTagCorpus.cpp)
    target_link_libraries(scalarTagCorpus PRIVATE scalar)
    target_include_directories(scalarTagCorpus PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_definitions(scalarTagCorpus PRIVATE ${SCALAR_FOLDER_DEFINITIONS})
    target_compile_options(scalarTagCorpus PRIVATE ${SCALAR_WARNINGS})

    enable_testing()
    add_subdirectory(tests)
endif()

