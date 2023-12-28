cmake_minimum_required(VERSION 3.27)
project(DeepLearningZero)

set(CMAKE_CXX_STANDARD 23)

# json ライブラリを追加
set(JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/json/include)

# 行列演算用のライブラリを追加
add_subdirectory(nagatolib)
file(GLOB_RECURSE SOURCES
        src/*.cpp)
file(GLOB_RECURSE HEADERS
        src/*.hpp)

message(STATUS "## SOURCES: ${SOURCES}")
message(STATUS "## HEADERS: ${HEADERS}")

# DeepLearningZero library
add_library(
        nagato_deep_learning_zero_lib
        STATIC
        ${SOURCES}
        ${HEADERS}
)
target_link_libraries(nagato_deep_learning_zero_lib
        nagato
)
target_include_directories(nagato_deep_learning_zero_lib
        PRIVATE
        nagatolib/src
        ${JSON_INCLUDE_DIR}
)
set_target_properties(nagato_deep_learning_zero_lib
        PROPERTIES
        LINKER_LANGUAGE CXX
)
# コンパイルオプションを追加
# DEBUG
target_compile_options(nagato_deep_learning_zero_lib
        PRIVATE
        -g
        -O0
        -Wall
)
# RELEASE
target_compile_options(nagato_deep_learning_zero_lib
        PRIVATE
        -O3
        -Wall
        -march=native
)


# DeepLearningZero executable
add_executable(DeepLearningZero main.cpp)
target_link_libraries(DeepLearningZero
        nagato
        nagato_deep_learning_zero_lib
)

target_include_directories(DeepLearningZero
        PRIVATE
        nagatolib/src
        src
        ${JSON_INCLUDE_DIR}
)

