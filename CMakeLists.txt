cmake_minimum_required(VERSION 3.27)
project(DeepLearningZero)

set(CMAKE_CXX_STANDARD 23)

# OpenMPのfindに必要な変数設定
set(OpenMP_HOME "/opt/homebrew/Cellar/libomp/17.0.6")
# 見つかったか？
message(STATUS "## OpenMP_C_Found   : ${OpenMP_C_FOUND}")
message(STATUS "## OpenMP_CXX_Found : ${OpenMP_CXX_FOUND}")

# json ライブラリを追加
set(JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/json/include)

# 行列演算用のライブラリを追加
add_subdirectory(nagatolib)
file(GLOB_RECURSE SOURCES
        src/*.cpp)
file(GLOB_RECURSE HEADERS
        src/*.hpp)

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
target_compile_options(
        nagato_deep_learning_zero_lib PRIVATE
        $<$<CONFIG:Release>: -O3 -march=native>
        $<$<CONFIG:Debug>: -O0 -g3 -Wall>
)
target_compile_options(nagato_deep_learning_zero_lib PRIVATE -Xpreprocessor -fopenmp)
target_include_directories(nagato_deep_learning_zero_lib PRIVATE /opt/homebrew/opt/libomp/include)
target_link_libraries(nagato_deep_learning_zero_lib /opt/homebrew/opt/libomp/lib/libomp.a)


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
target_compile_options(
        DeepLearningZero PRIVATE
        $<$<CONFIG:Release>: -O3 -march=native>
        $<$<CONFIG:Debug>: -O0 -g3 -Wall>
)

target_compile_options(DeepLearningZero PRIVATE -Xpreprocessor -fopenmp)
target_include_directories(DeepLearningZero PRIVATE /opt/homebrew/opt/libomp/include)
target_link_libraries(DeepLearningZero /opt/homebrew/opt/libomp/lib/libomp.a)