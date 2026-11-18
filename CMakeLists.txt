# ==============================================================================
# @file CMakeLists.txt
# @brief Top-level CMake build definition for the Umgebung Reality Simulation Engine.
#
# @details
# Configures cross-platform C++20 and CUDA 20 compilation using Ninja and vcpkg
# manifest mode (`vcpkg.json`). Supports two primary build modes from a single
# codebase:
#   1. Desktop Interactive GUI (`UMGEBUNG_BUILD_GUI=ON`):
#      Links GLFW, Vulkan, NVIDIA NVRHI, and Dear ImGui for interactive 2D/3D
#      visualization and multiverse hierarchy inspection on personal workstations.
#   2. Headless HPC Batch (`UMGEBUNG_BUILD_GUI=OFF`):
#      Omits all windowing and rendering dependencies so the engine builds cleanly
#      on Linux GPU compute clusters (UMaine ARCSIM, UNH Premise, RIT SPORC).
#
# Additionally provides:
#   - Automated CTest numerical/regression verification (`UMGEBUNG_BUILD_TESTS`).
#   - Optional offline Lean 4 formal proof checking (`UMGEBUNG_VERIFY_PROOFS`),
#     strictly decoupled from GPU/HPC runtime builds.
# ==============================================================================

# Require CMake >= 3.21 for first-class CUDA 20 standard support, CMakePresets v3,
# and modern generator expressions (such as $<TARGET_RUNTIME_DLLS:...>).
cmake_minimum_required(VERSION 3.21)

# ------------------------------------------------------------------------------
# Section 1: vcpkg Toolchain & Triplet Configuration (Must precede `project()`)
# ------------------------------------------------------------------------------
# Allow the vcpkg target triplet to be overridden by CMakePresets.json or via
# `-DVCPKG_TARGET_TRIPLET=<triplet>` on the command line (e.g., custom HPC
# triplets). Fall back to standard x64 desktop/server triplets by host OS.
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
    if(WIN32)
        set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
    elseif(UNIX)
        set(VCPKG_TARGET_TRIPLET "x64-linux" CACHE STRING "vcpkg target triplet")
    endif()
endif()

# Use dynamic C/C++ runtime linkage so shared libraries (CUDA runtime, Vulkan
# loader, and system drivers) share a single consistent CRT/allocator boundary.
set(VCPKG_CRT_LINKAGE dynamic)

# Point CMake to the vendored vcpkg submodule toolchain unless the user or an
# HPC build script explicitly provided an external toolchain file.
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

# Enforce strict ISO C++20 on host code without compiler-specific extensions
# (ensures portability across MSVC on Windows and GCC/Clang on Linux/HPC).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enforce CUDA C++20 on device/kernel translation units (.cu) so shared headers
# between host C++ and CUDA kernels compile under a uniform language standard.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Target NVIDIA GPU Compute Capabilities covering every system in GEMINI.md's
# Hardware Topology Matrix:
#   - 70 : Volta        (UNH Premise NVIDIA V100)
#   - 75 : Turing       (Personal Desktop NVIDIA TITAN RTX; UMaine ARCSIM RTX)
#   - 80 : Ampere       (UMaine ARCSIM DGX A100; UNH Premise A100; RIT SPORC A100)
#   - 89 : Ada Lovelace (Personal Laptop NVIDIA RTX 4070 Laptop; ARCSIM RTX)
#
# Guarded with `if(NOT DEFINED ...)` so HPC jobs or developers can narrow to a
# single architecture (e.g., `-DCMAKE_CUDA_ARCHITECTURES=80`) for faster builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 89 CACHE STRING "Target NVIDIA CUDA SM architectures")
endif()

# Place all build artifacts in predictable, uniform subdirectories beneath the
# active preset's binary directory (`out/build/<preset>/bin` and `lib`).
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ------------------------------------------------------------------------------
# Section 3: Build Options & Feature Toggles
# ------------------------------------------------------------------------------
# Desktop GUI toggle: ON for interactive workstations, OFF for headless HPC nodes.
option(UMGEBUNG_BUILD_GUI
    "Build with interactive desktop UI (GLFW, Vulkan, NVRHI, Dear ImGui)"
    ON
)

# Verification test suite toggle: registers CTest targets in `tests/`.
option(UMGEBUNG_BUILD_TESTS
    "Build and register automated CTest verification and regression tests"
    ON
)

# Optional Lean 4 formal verification toggle: executes `lake build` inside
# `formal/`. Kept OFF by default so HPC nodes never require `elan`/`lake`.
option(UMGEBUNG_VERIFY_PROOFS
    "Verify Lean 4 formal specifications and proofs in formal/ using Lake"
    OFF
)

# On Linux desktops (e.g., Arch/Manjaro with NVIDIA proprietary drivers), vcpkg's
# isolated Vulkan loader can miss system NVIDIA ICD manifests under /usr/share/vulkan.
# Preferring the system Vulkan loader resolves ICD discovery automatically.
option(UMGEBUNG_USE_SYSTEM_VULKAN
    "Prefer the system Vulkan loader over vcpkg's bundled loader on Linux"
    ON
)

# ------------------------------------------------------------------------------
# Section 4: Source File Organization by Architectural Module
# ------------------------------------------------------------------------------
# Sources are grouped to mirror `include/umgebung/` and `src/` (see GEMINI.md).
# As each architectural module is implemented, add its `.cpp`/`.cu` files to the
# corresponding list below.

# Core runtime (EngineContext, SystemProfiler, CLI argument parsing, logging, timing).
set(UMGEBUNG_CORE_SOURCES
    # e.g., src/core/SystemProfiler.cpp
    # e.g., src/core/EngineContext.cpp
)

# GPU compute layer (CUDA device/context/stream/memory management, multi-GPU dispatch).
set(UMGEBUNG_COMPUTE_SOURCES
    # e.g., src/compute/CUDAContext.cpp
)

# Multiverse/Universe hierarchy & `.umgebung` scene schema serialization.
set(UMGEBUNG_SCENE_SOURCES
    # e.g., src/scene/UniverseScene.cpp
)

# Pillar 1: Verified Mainstream Physics solvers & textbook numerical baselines.
set(UMGEBUNG_MODELS_MAINSTREAM_SOURCES
    # e.g., src/models/mainstream/NewtonianNBody.cu
)

# Pillar 2: Unified Research Models synthesized verbatim from submodules/research/Thinkers.
set(UMGEBUNG_MODELS_UNIFIED_SOURCES
    # e.g., src/models/unified/UnifiedFieldModel.cu
)

# Desktop-only rendering & UI layer (compiled strictly when UMGEBUNG_BUILD_GUI=ON).
set(UMGEBUNG_GRAPHICS_SOURCES
    # e.g., src/graphics/RenderDevice.cpp
)

# Aggregate all non-GUI translation units common to both Desktop and Headless HPC.
set(UMGEBUNG_SOURCES
    src/Main.cpp
    ${UMGEBUNG_CORE_SOURCES}
    ${UMGEBUNG_COMPUTE_SOURCES}
    ${UMGEBUNG_SCENE_SOURCES}
    ${UMGEBUNG_MODELS_MAINSTREAM_SOURCES}
    ${UMGEBUNG_MODELS_UNIFIED_SOURCES}
)

# Embed the Windows application icon resource into the `.exe` on Win32 builds.
if(WIN32)
    list(APPEND UMGEBUNG_SOURCES assets/icon/Umgebung.rc)
endif()

# ------------------------------------------------------------------------------
# Section 5: Primary Executable Target & Compiler Configuration
# ------------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${UMGEBUNG_SOURCES})

# Expose project public headers (`include/umgebung/...`) scoped cleanly to this target.
target_include_directories(${PROJECT_NAME} PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

# Enable strict host C++ warnings without passing unsupported warning flags to nvcc.
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
)

# Define consistent platform feature macros available in both GUI and HPC builds.
if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE UMGEBUNG_PLATFORM_WINDOWS=1)
elseif(UNIX)
    target_compile_definitions(${PROJECT_NAME} PRIVATE UMGEBUNG_PLATFORM_LINUX=1)
endif()

# ------------------------------------------------------------------------------
# Section 6: Core Dependencies (Shared Across Desktop & Headless HPC Modes)
# ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(cxxopts CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    cxxopts::cxxopts
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

# ------------------------------------------------------------------------------
# Section 7: Conditional Desktop GUI Stack (`UMGEBUNG_BUILD_GUI=ON`)
# ------------------------------------------------------------------------------
if(UMGEBUNG_BUILD_GUI)
    # On Linux, prefer the host system Vulkan loader library (`libvulkan.so`)
    # across standard FHS paths (Arch/Manjaro `/usr/lib`, RHEL/Rocky `/usr/lib64`,
    # Debian/Ubuntu `/usr/lib/x86_64-linux-gnu`) so host NVIDIA ICD manifests
    # under `/usr/share/vulkan/icd.d/` are discovered reliably, while still
    # allowing `Vulkan_INCLUDE_DIR` to resolve from vcpkg's `vulkan-headers`
    # (which stay version-matched to vcpkg's `nvrhi`).
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

    target_link_libraries(${PROJECT_NAME} PRIVATE
        imgui::imgui
        glfw
        nvrhi_vk
        nvrhi
        Vulkan::Vulkan
    )

    # Attach desktop graphics translation units when present.
    if(UMGEBUNG_GRAPHICS_SOURCES)
        target_sources(${PROJECT_NAME} PRIVATE ${UMGEBUNG_GRAPHICS_SOURCES})
    endif()

    target_compile_definitions(${PROJECT_NAME} PRIVATE UMGEBUNG_HAS_GUI=1)
else()
    # Explicitly define UMGEBUNG_HAS_GUI=0 so `#if UMGEBUNG_HAS_GUI` guards stay
    # unambiguous under `-Wundef` and headless HPC configurations.
    target_compile_definitions(${PROJECT_NAME} PRIVATE UMGEBUNG_HAS_GUI=0)
endif()

# ------------------------------------------------------------------------------
# Section 8: Post-Build Packaging Steps (Runtime DLLs & Assets)
# ------------------------------------------------------------------------------
# On Windows, copy all dynamically linked vcpkg/runtime DLLs next to Umgebung.exe
# so the binary launches directly from `out/build/<preset>/bin` without PATH edits.
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

# Mirror the repository `assets/` directory next to the compiled executable so
# relative asset paths (icons, shaders, fonts) resolve identically in all modes.
if(EXISTS "${CMAKE_SOURCE_DIR}/assets")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/assets"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets"
        COMMENT "Copying assets directory to build output"
    )
endif()

# ------------------------------------------------------------------------------
# Section 9: Pillar 3 — Optional Lean 4 Formal Proof Verification (`formal/`)
# ------------------------------------------------------------------------------
# When enabled, invokes `lake build` inside `formal/` to ensure all Lean 4
# mathematical models and integrator proofs compile cleanly with zero errors.
# Kept strictly opt-in (`OFF` by default) to preserve HPC cluster portability.
if(UMGEBUNG_VERIFY_PROOFS)
    find_program(UMGEBUNG_LAKE_EXECUTABLE lake)
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
# Section 10: Automated Verification & Regression Suite (CTest)
# ------------------------------------------------------------------------------
if(UMGEBUNG_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
