cmake_minimum_required(VERSION 3.15)
project(Valen VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

add_compile_definitions(ASIO_STANDALONE)

file(GLOB_RECURSE SOURCES "src/*.cc" "src/*.cpp" "main.cc")

add_executable(valen ${SOURCES})

# Target-based include directories
target_include_directories(valen PRIVATE
    include
    include/network_include/include
)

# Required for threading support in Asio / Crow
find_package(Threads REQUIRED)
target_link_libraries(valen PRIVATE Threads::Threads)