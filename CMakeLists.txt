cmake_minimum_required(VERSION 3.21)
project(collision_detection CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(collision_detection main.cu scene.cpp scene.h core.cu core.cuh utils.h render.cpp render.h dataGenerator.cpp dataGenerator.h)

set_target_properties(collision_detection PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
