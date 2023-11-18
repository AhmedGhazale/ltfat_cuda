cmake_minimum_required(VERSION 3.26)
project(LTFAT CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(LTFAT main.cu ltfat_cpu.h
        ltfat_gpu1.cuh)

set_target_properties(LTFAT PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
