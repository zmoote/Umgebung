cmake_minimum_required (VERSION 3.21)

# Allow triplet and CRT linkage to be controlled via presets or command line
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
    if(WIN32)
        set(VCPKG_TARGET_TRIPLET x64-windows)
    elseif(UNIX)
        set(VCPKG_TARGET_TRIPLET x64-linux)
    endif()
endif()

set(VCPKG_CRT_LINKAGE dynamic)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/submodules/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")

project ("Umgebung" LANGUAGES CXX CUDA VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_ARCHITECTURES 75 80 89)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

include_directories(${CMAKE_SOURCE_DIR}/include)

# Toggle to disable UI for headless HPC batch jobs
option(UMGEBUNG_BUILD_GUI "Build with Dear ImGui and GLFW desktop UI" ON)

# Source files common to both headless and GUI targets
set(UMGEBUNG_SOURCES src/Main.cpp)
if(WIN32)
    list(APPEND UMGEBUNG_SOURCES assets/icon/Umgebung.rc)
endif()

add_executable (${PROJECT_NAME} ${UMGEBUNG_SOURCES})

find_package(CUDAToolkit REQUIRED)
find_package(nvrhi CONFIG REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    CUDA::cuda_driver
    CUDA::cudart
    nvrhi
    cxxopts::cxxopts
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

# Conditionally add ImGui and GLFW for desktop builds with the GUI feature enabled
if(UMGEBUNG_BUILD_GUI)
    find_package(imgui CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE imgui::imgui glfw)
    target_compile_definitions(${PROJECT_NAME} PRIVATE UMGEBUNG_HAS_GUI=1)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE UMGEBUNG_HAS_GUI=0)
endif()

# Post-build copy steps (Windows DLLs and assets)
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>>,copy,true>
        $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND_EXPAND_LISTS
    )
endif()

add_custom_command(
    TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/assets
            $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets
    COMMENT "Copying assets directory to build output"
)