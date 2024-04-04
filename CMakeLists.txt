cmake_minimum_required(VERSION 3.5)

project(make_NN)

set(CMAKE_PREFIX_PATH /Users/taewoohong/libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

add_executable(main src/main.cpp)

target_link_libraries(main ${TORCH_LIBRARIES})

set_property(TARGET main PROPERTY CXX_STANDARD 17)