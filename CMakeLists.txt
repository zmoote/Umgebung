cmake_minimum_required(VERSION 3.25.2)
project(Umgebung VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 60-virtual 70-virtual 75-real 89-real)

# vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/submodules/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")

# Source files
file(GLOB_RECURSIVE SOURCES src/*.cpp src/*.cu)
add_executable(${PROJECT_NAME} ${SOURCES})

# Dependencies
find_package(bgfx CONFIG REQUIRED)
find_package(Boost COMPONENTS multiprecision REQUIRED)
find_package(EnTT CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(unofficial-omniverse-physx-sdk CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    bgfx::bgfx bgfx::bimg bgfx::bx
    Boost::multiprecision
    EnTT::EnTT
    glfw
    glm::glm
    imgui::imgui
    nlohmann_json::nlohmann_json
    unofficial::omniverse-physx-sdk::sdk
    CUDA::cudart CUDA::cublas
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE include)

# Copy assets
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})

# PhysX GPU libraries
if(TARGET unofficial::omniverse-physx-sdk::gpu-library)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-library>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()