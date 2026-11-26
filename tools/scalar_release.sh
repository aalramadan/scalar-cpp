# Shell functions that read the release of cmake/scalarRelease.cmake, sourced by the scripts that compare the
# port with the Python SCALAR and that write the golden files of ctest.
# shellcheck shell=bash

scalar_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The value of a set() of cmake/scalarRelease.cmake, such as SCALAR_UPSTREAM_TAG, without the CR of its line end
release_value() { tr -d '\r' < "$scalar_repo/cmake/scalarRelease.cmake" | sed -n "s/^set($1 *\"\(.*\)\")$/\1/p"; }

# The model folder of CMakeLists.txt on Linux and macOS
model_folder() {
    local revision
    revision="$(release_value SCALAR_MODEL_REVISION)"
    echo "${XDG_CACHE_HOME:-$HOME/.cache}/scalar-cpp/model-${revision:0:7}"
}

# Stops the script unless the installed Python SCALAR is the release that the port follows
require_installed_release() {
    local tag installed
    tag="$(release_value SCALAR_UPSTREAM_TAG)"
    installed="$(python3 -c 'from scalar_tagger.version import __version__; print(__version__)')"
    if [[ "v$installed" != "$tag" ]]; then
        echo "the installed SCALAR is $installed, while the port follows $tag:" >&2
        echo "  pip install \"scalar-tagger @ git+https://github.com/SCANL/scanl_tagger.git@$tag\"" >&2
        exit 1
    fi
}
