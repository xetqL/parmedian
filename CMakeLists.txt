cmake_minimum_required(VERSION 3.17)
project(parallel_median)

find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILER smpicxx)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(med main.cpp median.hpp)
