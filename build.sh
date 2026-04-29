#!/usr/bin/env bash

c++ -O3 -Wall -shared -std=c++17 -fPIC \
    -I ~/Github/Lattice \
    $(python3 -m pybind11 --includes) \
    fp_search.cpp \
    -o fp_search_cpp$(.venv/bin/python3-config --extension-suffix)

c++ -O3 -Wall -shared -std=c++17 -fPIC \
    -I ~/Github/Lattice \
    $(python3 -m pybind11 --includes) \
    vsearch.cpp \
    -o vsearch_cpp$(.venv/bin/python3-config --extension-suffix) \
    -lgmpxx -lgmp