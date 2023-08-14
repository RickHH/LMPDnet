cmake_minimum_required(VERSION 3.9.0)
project(parallelproj_cuda LANGUAGES C CXX CUDA)

set(SOURCES lcx_projector_kernels.cu)

add_library(parallelproj_cuda SHARED ${SOURCES})