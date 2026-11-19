# ==============================================================================
# @file tests/CMakeLists.txt
# @brief CTest verification, conservation-law, hardware-adaptation, and
#        regression test registration for Umgebung.
#
# @details
# Builds and registers automated CTest targets executed via
# `ctest --preset <preset-name>`. All tests run cleanly in both Desktop GUI
# (`UMGEBUNG_BUILD_GUI=ON`) and Headless HPC (`UMGEBUNG_BUILD_GUI=OFF`) presets
# without requiring an interactive display server.
# ==============================================================================

# Helper macro applying strict host C++ warnings to test targets consistently.
function(umgebung_configure_test_target target_name)
    target_link_libraries(${target_name} PRIVATE UmgebungEngine)
    target_compile_options(${target_name} PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall -Wextra -Wpedantic>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
    )
endfunction()

# ------------------------------------------------------------------------------
# 1. Core Hardware Discovery & Cross-Topology Adaptation Unit Tests
# ------------------------------------------------------------------------------
add_executable(UmgebungSystemProfilerTests SystemProfilerTests.cpp)
umgebung_configure_test_target(UmgebungSystemProfilerTests)

add_test(
    NAME Core.SystemProfilerUnitTests
    COMMAND $<TARGET_FILE:UmgebungSystemProfilerTests>
)
set_tests_properties(Core.SystemProfilerUnitTests PROPERTIES
    LABELS "core;unit;hardware"
)

# ------------------------------------------------------------------------------
# 2. Compute Layer Tests (`CUDAContext`, `DeviceBuffer<T>`, VRAM Budget)
# ------------------------------------------------------------------------------
add_executable(UmgebungComputeTests ComputeTests.cpp)
umgebung_configure_test_target(UmgebungComputeTests)

add_test(
    NAME Compute.CudaContextAndBufferTests
    COMMAND $<TARGET_FILE:UmgebungComputeTests>
)
set_tests_properties(Compute.CudaContextAndBufferTests PROPERTIES
    LABELS "compute;unit;cuda"
)

# ------------------------------------------------------------------------------
# 3. Scene Schema & Multiverse Loader Tests (`.umgebung` JSON v1)
# ------------------------------------------------------------------------------
add_executable(UmgebungSceneTests SceneTests.cpp)
umgebung_configure_test_target(UmgebungSceneTests)

add_test(
    NAME Scene.UniverseAndMultiverseTests
    COMMAND $<TARGET_FILE:UmgebungSceneTests> "${CMAKE_SOURCE_DIR}/scenes"
)
set_tests_properties(Scene.UniverseAndMultiverseTests PROPERTIES
    LABELS "scene;unit"
)

# ------------------------------------------------------------------------------
# 4. Pillar 1 Mainstream Physics Verification (FP64 CUDA Velocity Verlet N-Body)
# ------------------------------------------------------------------------------
add_executable(UmgebungNewtonianNBodyTests NewtonianNBodyTests.cpp)
umgebung_configure_test_target(UmgebungNewtonianNBodyTests)

add_test(
    NAME Models.MainstreamNewtonianNBodyTests
    COMMAND $<TARGET_FILE:UmgebungNewtonianNBodyTests> "${CMAKE_SOURCE_DIR}/scenes"
)
set_tests_properties(Models.MainstreamNewtonianNBodyTests PROPERTIES
    LABELS "models;mainstream;physics;cuda"
)

# ------------------------------------------------------------------------------
# 5. Pillar 2 Unified Research Models & 3-Way A/B Comparison Verification
# ------------------------------------------------------------------------------
add_executable(UmgebungUnifiedModelTests UnifiedModelTests.cpp)
umgebung_configure_test_target(UmgebungUnifiedModelTests)

add_test(
    NAME Models.UnifiedThinkersAndABTests
    COMMAND $<TARGET_FILE:UmgebungUnifiedModelTests> "${CMAKE_SOURCE_DIR}/scenes"
)
set_tests_properties(Models.UnifiedThinkersAndABTests PROPERTIES
    LABELS "models;unified;physics;cuda"
)

# ------------------------------------------------------------------------------
# 6. Three-Tier Model Registry, Mainstream NFW & All 5 Standalone Thinkers
# ------------------------------------------------------------------------------
add_executable(UmgebungStandaloneThinkersAndNFWTests StandaloneThinkersAndNFWTests.cpp)
umgebung_configure_test_target(UmgebungStandaloneThinkersAndNFWTests)

add_test(
    NAME Models.StandaloneThinkersAndNFWTests
    COMMAND $<TARGET_FILE:UmgebungStandaloneThinkersAndNFWTests> "${CMAKE_SOURCE_DIR}/scenes"
)
set_tests_properties(Models.StandaloneThinkersAndNFWTests PROPERTIES
    LABELS "models;thinkers;mainstream;physics;cuda"
)

# ------------------------------------------------------------------------------
# 7. CLI & End-to-End Headless Batch + Simultaneous A/B Overlay Smoke Tests
# ------------------------------------------------------------------------------
add_test(
    NAME Core.ExecutableProfileSmokeTest
    COMMAND $<TARGET_FILE:Umgebung> --profile-only
)
set_tests_properties(Core.ExecutableProfileSmokeTest PROPERTIES
    LABELS "core;smoke;hardware"
)

add_test(
    NAME Core.ExecutableHeadlessSceneTest
    COMMAND $<TARGET_FILE:Umgebung>
            --headless
            --scene "${CMAKE_SOURCE_DIR}/scenes/scenarios/kepler_two_body.scenario.umgebung"
            --steps 500
)
set_tests_properties(Core.ExecutableHeadlessSceneTest PROPERTIES
    LABELS "core;smoke;physics;cuda"
)

add_test(
    NAME Core.ExecutableHeadlessABOverlayTest
    COMMAND $<TARGET_FILE:Umgebung>
            --headless
            --scene "${CMAKE_SOURCE_DIR}/scenes/scenarios/galaxy_periphery.scenario.umgebung"
            --model "Mainstream::Newtonian"
            --compare-model "Thinker::Essonne"
            --steps 500
)
set_tests_properties(Core.ExecutableHeadlessABOverlayTest PROPERTIES
    LABELS "core;smoke;unified;ab;physics;cuda"
)

add_test(
    NAME Core.ExecutableHeadlessUnifiedUniverseTest
    COMMAND $<TARGET_FILE:Umgebung>
            --headless
            --scene "${CMAKE_SOURCE_DIR}/scenes/universes/sol_prime.umgebung"
            --model "Unified::Synthesis"
            --steps 250
)
set_tests_properties(Core.ExecutableHeadlessUnifiedUniverseTest PROPERTIES
    LABELS "core;smoke;unified;physics;cuda"
)

add_test(
    NAME Core.ExecutableHelpTest
    COMMAND $<TARGET_FILE:Umgebung> --help
)
set_tests_properties(Core.ExecutableHelpTest PROPERTIES
    LABELS "core;smoke;cli"
)

add_test(
    NAME Core.ExecutableVersionTest
    COMMAND $<TARGET_FILE:Umgebung> --version
)
set_tests_properties(Core.ExecutableVersionTest PROPERTIES
    LABELS "core;smoke;cli"
)

# ------------------------------------------------------------------------------
# 6. Optional Formal Proof Verification Test (Lean 4 via Lake)
# ------------------------------------------------------------------------------
if(UMGEBUNG_VERIFY_PROOFS AND UMGEBUNG_LAKE_EXECUTABLE)
    add_test(
        NAME Formal.Lean4ProofCheck
        COMMAND "${UMGEBUNG_LAKE_EXECUTABLE}" build
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/formal"
    )
    set_tests_properties(Formal.Lean4ProofCheck PROPERTIES
        LABELS "formal;lean4"
    )
endif()
