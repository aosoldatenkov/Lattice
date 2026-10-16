cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Eigen: use the system copy if there is one, otherwise download a pinned release
find_package(Eigen3 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
    include(FetchContent)
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        SOURCE_SUBDIR do-not-build)
    FetchContent_MakeAvailable(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE ${eigen_SOURCE_DIR})
endif()

# GMP with C++ bindings (system package, e.g. libgmp-dev): via pkg-config if available,
# otherwise search the standard locations directly
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(GMPXX QUIET IMPORTED_TARGET gmpxx gmp)
endif()
if(TARGET PkgConfig::GMPXX)
    add_library(GMP::GMPXX ALIAS PkgConfig::GMPXX)
else()
    find_path(GMPXX_INCLUDE_DIR gmpxx.h)
    find_path(GMP_INCLUDE_DIR gmp.h)
    find_library(GMPXX_LIBRARY gmpxx)
    find_library(GMP_LIBRARY gmp)
    if(NOT (GMPXX_INCLUDE_DIR AND GMP_INCLUDE_DIR AND GMPXX_LIBRARY AND GMP_LIBRARY))
        message(FATAL_ERROR "GMP with C++ bindings not found (install libgmp-dev or equivalent)")
    endif()
    add_library(GMP::GMPXX INTERFACE IMPORTED)
    target_include_directories(GMP::GMPXX INTERFACE ${GMPXX_INCLUDE_DIR} ${GMP_INCLUDE_DIR})
    target_link_libraries(GMP::GMPXX INTERFACE ${GMPXX_LIBRARY} ${GMP_LIBRARY})
    message(STATUS "Found GMP: ${GMP_LIBRARY}, ${GMPXX_LIBRARY}")
endif()

pybind11_add_module(fp_search_cpp src/cpp/fp_search_module.cpp)
target_link_libraries(fp_search_cpp PRIVATE Eigen3::Eigen)

pybind11_add_module(vsearch_cpp src/cpp/vsearch_module.cpp)
target_link_libraries(vsearch_cpp PRIVATE Eigen3::Eigen GMP::GMPXX Threads::Threads)

foreach(target fp_search_cpp vsearch_cpp)
    target_compile_options(${target} PRIVATE -Wall)
endforeach()

install(TARGETS fp_search_cpp vsearch_cpp LIBRARY DESTINATION lattice)
