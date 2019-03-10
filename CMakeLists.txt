cmake_minimum_required(VERSION 3.13)
project(PageRankGPU)

set(CMAKE_CXX_STANDARD 14)

add_executable(PageRankGPU main.cpp Matrix.cpp Matrix.h PageRank.cpp PageRank.h)