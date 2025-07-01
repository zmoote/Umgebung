cmake_minimum_required(VERSION 3.25.2)

project("Umgebung" VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/submodules/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 60-virtual 70-virtual 75-real 89-real)

if(WIN32)
    set(VCPKG_TARGET_TRIPLET x64-windows)
elseif(UNIX AND NOT APPLE)
    set(VCPKG_TARGET_TRIPLET x64-linux)
endif()

add_executable(${PROJECT_NAME} src/main.cpp)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Find dependencies
find_package(bgfx CONFIG REQUIRED)
find_package(boost_multiprecision CONFIG REQUIRED)
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
    bgfx::bx bgfx::bgfx bgfx::bimg bgfx::bimg_decode
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
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Copy PhysX GPU libraries
if(TARGET unofficial::omniverse-physx-sdk::gpu-library)
    if(UNIX)
        set_target_properties(${PROJECT_NAME} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
        )
    endif()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-library>
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    if(WIN32)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-device-library>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endif()
else()
    message(FATAL_ERROR "PhysX GPU library not found. Ensure vcpkg has PhysX with GPU support enabled.")
endif()