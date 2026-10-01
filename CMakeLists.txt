cmake_minimum_required(VERSION 3.20)
project(model_inference_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(inference src/model.cpp)
target_include_directories(inference PUBLIC include)
target_compile_options(inference PRIVATE -Wall -Wextra -Wpedantic)

add_executable(model-runtime src/main.cpp)
target_link_libraries(model-runtime PRIVATE inference)

enable_testing()
add_executable(inference-tests tests/test_model.cpp)
target_link_libraries(inference-tests PRIVATE inference)
add_test(NAME inference-tests COMMAND inference-tests)

