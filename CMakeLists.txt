cmake_minimum_required(VERSION 3.21)
project(Umgebung LANGUAGES CXX CUDA)

enable_testing()

# Project wide setup
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# Disable MSBuild file tracking if using caching tools in Visual Studio[cite: 3]
list(APPEND CMAKE_VS_GLOBALS TrackFileAccess=false)

# Main targets built by this project
add_subdirectory(src)

# Wrap auxiliary directories to prevent execution if built as a subproject[cite: 3]
if(PROJECT_IS_TOP_LEVEL)
    add_subdirectory(tests)
    add_subdirectory(packaging)
endif()