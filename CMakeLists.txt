cmake_minimum_required(VERSION 3.15)
project(Valen VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cc" "src/*.cpp" "main.cc")

add_executable(valen ${SOURCES})