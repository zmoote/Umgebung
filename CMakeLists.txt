# ==============================================================================
# @file CMakeLists.txt
# @brief Top-level CMake build definition for the Umgebung Reality Simulation Engine.
#
# @details
# Configures cross-platform C++20 and CUDA 20 compilation using Ninja and vcpkg
# manifest mode (`vcpkg.json`). Organizes engine subsystems into a reusable
# static library (`UmgebungEngine`) consumed by both the primary `Umgebung`
# executable (`src/Main.cpp`) and the CTest verification suite (`tests/`).
#
# Supports two primary execution modes from a single codebase:
#   1. Desktop Interactive GUI (`UMGEBUNG_BUILD_GUI=ON`):
#      Compiles `umgebung::graphics::Window` (and future Vulkan/NVRHI/ImGui
#      renderers) and links GLFW, Vulkan, NVIDIA NVRHI, and Dear ImGui.
#   2. Headless HPC Batch (`UMGEBUNG_BUILD_GUI=OFF`):
#      Omits all windowing and rendering sources/dependencies so the engine
#      builds cleanly on Linux GPU compute clusters (ARCSIM, Premise, SPORC).
# ==============================================================================

# Require CMake >= 3.21 for first-class CUDA 20 standard support, CMakePresets v3,
# and modern generator expressions (such as $<TARGET_RUNTIME_DLLS:...>).
cmake_minimum_required(VERSION 3.21)

# ------------------------------------------------------------------------------
# Section 1: vcpkg Toolchain & Triplet Configuration (Must precede `project()`)
# ------------------------------------------------------------------------------
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
    if(WIN32)
        set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
    elseif(UNIX)
        set(VCPKG_TARGET_TRIPLET "x64-linux" CACHE STRING "vcpkg target triplet")
    endif()
endif()

# Dynamic CRT linkage keeps host C++, CUDA runtime, and Vulkan loader aligned.
set(VCPKG_CRT_LINKAGE dynamic)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE
        "${CMAKE_SOURCE_DIR}/submodules/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Path to the vcpkg CMake toolchain file"
    )
endif()

# ------------------------------------------------------------------------------
# Section 2: Project Declaration & Language Standards
# ------------------------------------------------------------------------------
project("Umgebung"
    VERSION 0.0.1
    DESCRIPTION "GPU-accelerated multi-scale reality simulation engine and computational physics testbed"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Target NVIDIA GPU Compute Capabilities covering every system in GEMINI.md's
# Hardware Topology Matrix:
#   - 70 : Volta        (UNH Premise NVIDIA V100 — supported on CUDA <= 12.x; removed in CUDA 13.0+)
#   - 75 : Turing       (Personal Desktop NVIDIA TITAN RTX; UMaine ARCSIM RTX)
#   - 80 : Ampere       (UMaine ARCSIM DGX A100; UNH Premise A100; RIT SPORC A100)
#   - 89 : Ada Lovelace (Personal Laptop NVIDIA RTX 4070 Laptop; ARCSIM RTX)
#
# Automatically adapts to the detected `nvcc` major version so CUDA 13.x+ hosts
# do not fail on dropped Volta (`sm_70`) while CUDA 12.x V100 cluster toolchains
# automatically include `70`. Can still be overridden explicitly via
# `-DUMGEBUNG_CUDA_ARCHITECTURES=80` (e.g., for fast single-arch cluster builds).
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "13.0")
    set(_UMGEBUNG_DEFAULT_CUDA_ARCHS "70;75;80;89")
else()
    set(_UMGEBUNG_DEFAULT_CUDA_ARCHS "75;80;89")
endif()
set(UMGEBUNG_CUDA_ARCHITECTURES "${_UMGEBUNG_DEFAULT_CUDA_ARCHS}" CACHE STRING
    "Target NVIDIA CUDA SM architectures compiled into UmgebungEngine"
)
set(CMAKE_CUDA_ARCHITECTURES "${UMGEBUNG_CUDA_ARCHITECTURES}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ------------------------------------------------------------------------------
# Section 3: Build Options & Feature Toggles
# ------------------------------------------------------------------------------
option(UMGEBUNG_BUILD_GUI
    "Build with interactive desktop UI (GLFW, Vulkan, NVRHI, Dear ImGui)"
    ON
)

option(UMGEBUNG_BUILD_TESTS
    "Build and register automated CTest verification and regression tests"
    ON
)

option(UMGEBUNG_VERIFY_PROOFS
    "Verify Lean 4 formal specifications and proofs in formal/ using Lake"
    OFF
)

option(UMGEBUNG_USE_SYSTEM_VULKAN
    "Prefer the system Vulkan loader over vcpkg's bundled loader on Linux"
    ON
)

# ------------------------------------------------------------------------------
# Section 4: Source File Organization by Architectural Module
# ------------------------------------------------------------------------------
# Core runtime (startup hardware discovery/adaptation, CLI parsing, Application lifecycle).
set(UMGEBUNG_CORE_SOURCES
    src/core/SystemProfiler.cpp
    src/core/Application.cpp
)

# GPU compute layer (CUDA device/context/stream/memory management, multi-GPU dispatch).
set(UMGEBUNG_COMPUTE_SOURCES
    src/compute/CUDAContext.cpp
)

# Multiverse/Universe/Scenario hierarchy & `.umgebung` scene schema serialization.
set(UMGEBUNG_SCENE_SOURCES
    src/scene/UniverseScene.cpp
    src/scene/Multiverse.cpp
    src/scene/FlowerOfLifeMultiverse.cpp
)

# Pillar 1: Verified Mainstream Physics solvers, ModelRegistry & multi-domain engines.
set(UMGEBUNG_MODELS_MAINSTREAM_SOURCES
    src/models/ModelRegistry.cpp
    src/models/SolverFactory.cpp
    src/models/FieldGrid2DSolver.cu
    src/models/RadialScaleProfileSolver.cpp
    src/models/mainstream/NewtonianNBodySolver.cu
    src/models/mainstream/DarkMatterNFWSolver.cu
)

# Pillar 2 (Tier 2): Standalone Verbatim Thinker modules (`models/thinkers/`).
set(UMGEBUNG_MODELS_THINKERS_SOURCES
    src/models/thinkers/EssonneSolver.cu
    src/models/thinkers/HarameinSolver.cu
    src/models/thinkers/PaisSolver.cu
    src/models/thinkers/DanaanSolver.cu
    src/models/thinkers/VogelWillisSolver.cu
)

# Pillar 2 (Tier 3): Unified Synthesis composing all Thinker modules (`models/unified/`).
set(UMGEBUNG_MODELS_UNIFIED_SOURCES
    src/models/unified/UnifiedSynthesisSolver.cu
)

# Desktop-only windowing, rendering, and UI layer (compiled strictly when UMGEBUNG_BUILD_GUI=ON).
set(UMGEBUNG_GRAPHICS_SOURCES
    src/graphics/Window.cpp
    src/graphics/RenderDevice.cpp
)

# ------------------------------------------------------------------------------
# Section 5: Core Engine Static Library (`UmgebungEngine`)
# ------------------------------------------------------------------------------
# Building the engine modules into `UmgebungEngine` lets both the main `Umgebung`
# binary and CTest executables link directly against engine classes without
# duplicating compilation.
add_library(UmgebungEngine STATIC
    ${UMGEBUNG_CORE_SOURCES}
    ${UMGEBUNG_COMPUTE_SOURCES}
    ${UMGEBUNG_SCENE_SOURCES}
    ${UMGEBUNG_MODELS_MAINSTREAM_SOURCES}
    ${UMGEBUNG_MODELS_THINKERS_SOURCES}
    ${UMGEBUNG_MODELS_UNIFIED_SOURCES}
)

# Propagate public headers (`include/umgebung/...`) to all downstream targets.
target_include_directories(UmgebungEngine PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)

# Enable strict host C++ warnings without passing unsupported warning flags to nvcc.
target_compile_options(UmgebungEngine PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
)

# Propagate platform feature macros across all targets linking `UmgebungEngine`.
if(WIN32)
    target_compile_definitions(UmgebungEngine PUBLIC UMGEBUNG_PLATFORM_WINDOWS=1)
elseif(UNIX)
    target_compile_definitions(UmgebungEngine PUBLIC UMGEBUNG_PLATFORM_LINUX=1)
endif()

# ------------------------------------------------------------------------------
# Section 6: Core Dependencies (Shared Across Desktop & Headless HPC Modes)
# ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

target_link_libraries(UmgebungEngine PUBLIC
    CUDA::cuda_driver
    CUDA::cudart
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

# `cxxopts` is used internally by `Application.cpp` for CLI parsing.
target_link_libraries(UmgebungEngine PRIVATE
    cxxopts::cxxopts
)

# ------------------------------------------------------------------------------
# Section 7: Conditional Desktop GUI Stack (`UMGEBUNG_BUILD_GUI=ON`)
# ------------------------------------------------------------------------------
if(UMGEBUNG_BUILD_GUI)
    # On Linux, prefer the host system Vulkan loader library (`libvulkan.so`)
    # across standard FHS paths so host NVIDIA ICD manifests under
    # `/usr/share/vulkan/icd.d/` are discovered reliably, while `Vulkan_INCLUDE_DIR`
    # resolves from vcpkg's version-matched `vulkan-headers`.
    if(UNIX AND UMGEBUNG_USE_SYSTEM_VULKAN)
        find_library(UMGEBUNG_SYSTEM_VULKAN_LIB
            NAMES vulkan
            PATHS /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu
            NO_DEFAULT_PATH
        )
        if(UMGEBUNG_SYSTEM_VULKAN_LIB)
            set(Vulkan_LIBRARY "${UMGEBUNG_SYSTEM_VULKAN_LIB}" CACHE FILEPATH "System Vulkan loader library" FORCE)
            message(STATUS "Umgebung: Using system Vulkan loader at ${UMGEBUNG_SYSTEM_VULKAN_LIB}")
        else()
            message(STATUS "Umgebung: System Vulkan loader not found; falling back to default FindVulkan search.")
        endif()
    endif()

    find_package(Vulkan REQUIRED)
    find_package(imgui CONFIG REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
    find_package(nvrhi CONFIG REQUIRED)

    target_sources(UmgebungEngine PRIVATE ${UMGEBUNG_GRAPHICS_SOURCES})

    target_link_libraries(UmgebungEngine PUBLIC
        imgui::imgui
        glfw
        nvrhi_vk
        nvrhi
        Vulkan::Vulkan
    )

    target_compile_definitions(UmgebungEngine PUBLIC UMGEBUNG_HAS_GUI=1)
else()
    # Explicitly define UMGEBUNG_HAS_GUI=0 so `#if UMGEBUNG_HAS_GUI` guards stay
    # unambiguous in headless HPC configurations.
    target_compile_definitions(UmgebungEngine PUBLIC UMGEBUNG_HAS_GUI=0)
endif()

# ------------------------------------------------------------------------------
# Section 8: Primary Executable Target (`Umgebung`)
# ------------------------------------------------------------------------------
set(UMGEBUNG_APP_SOURCES src/Main.cpp)
if(WIN32)
    list(APPEND UMGEBUNG_APP_SOURCES assets/icon/Umgebung.rc)
endif()

add_executable(${PROJECT_NAME} ${UMGEBUNG_APP_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE UmgebungEngine)
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
)

# ------------------------------------------------------------------------------
# Section 9: Post-Build Packaging Steps (Runtime DLLs & Assets)
# ------------------------------------------------------------------------------
if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E
            $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>>,copy,true>
            $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND_EXPAND_LISTS
        COMMENT "Copying Windows runtime DLL dependencies to executable directory"
    )
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/assets")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets"
        COMMENT "Copying assets directory to build output"
    )
endif()

# Mirror the repository `scenes/` directory next to the compiled executable so
# bundled `.umgebung` benchmark universes auto-discover reliably from any CWD.
if(EXISTS "${CMAKE_SOURCE_DIR}/scenes")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/scenes"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/scenes"
        COMMENT "Copying scenes directory to build output"
    )
endif()

# ------------------------------------------------------------------------------
# Section 10: Pillar 3 — Optional Lean 4 Formal Proof Verification (`formal/`)
# ------------------------------------------------------------------------------
if(UMGEBUNG_VERIFY_PROOFS)
    find_program(UMGEBUNG_LAKE_EXECUTABLE
        NAMES lake
        HINTS "$ENV{HOME}/.elan/bin" "$ENV{USERPROFILE}/.elan/bin"
    )
    if(NOT UMGEBUNG_LAKE_EXECUTABLE)
        message(FATAL_ERROR
            "UMGEBUNG_VERIFY_PROOFS is ON, but the Lean 4 `lake` executable was not found in PATH. "
            "Install Lean 4 via elan or disable -DUMGEBUNG_VERIFY_PROOFS=OFF."
        )
    endif()

    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/formal/lakefile.toml" AND NOT EXISTS "${CMAKE_SOURCE_DIR}/formal/lakefile.lean")
        message(FATAL_ERROR
            "UMGEBUNG_VERIFY_PROOFS is ON, but no Lake manifest (lakefile.toml or lakefile.lean) "
            "exists in ${CMAKE_SOURCE_DIR}/formal/ yet."
        )
    endif()

    add_custom_target(umgebung_verify_proofs
        COMMAND "${UMGEBUNG_LAKE_EXECUTABLE}" build
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/formal"
        COMMENT "Verifying Lean 4 formal specifications and proofs via Lake"
        USES_TERMINAL
    )
endif()

# ------------------------------------------------------------------------------
# Section 11: Automated Verification & Regression Suite (CTest)
# ------------------------------------------------------------------------------
if(UMGEBUNG_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
