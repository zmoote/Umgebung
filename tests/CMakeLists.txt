# ==============================================================================
# @file tests/CMakeLists.txt
# @brief CTest verification, hardware-adaptation, and regression test registration.
#
# @details
# Builds and registers automated CTest targets executed via
# `ctest --preset <preset-name>`. All tests in this suite execute cleanly in
# both Desktop GUI (`UMGEBUNG_BUILD_GUI=ON`) and Headless HPC
# (`UMGEBUNG_BUILD_GUI=OFF`) presets without requiring an interactive display.
# ==============================================================================

# ------------------------------------------------------------------------------
# Unit Tests: Hardware Discovery & Cross-Topology Adaptation (`SystemProfiler`)
# ------------------------------------------------------------------------------
add_executable(UmgebungSystemProfilerTests
    SystemProfilerTests.cpp
)
target_link_libraries(UmgebungSystemProfilerTests PRIVATE
    UmgebungEngine
)
target_compile_options(UmgebungSystemProfilerTests PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
)

add_test(
    NAME Core.SystemProfilerUnitTests
    COMMAND $<TARGET_FILE:UmgebungSystemProfilerTests>
)
set_tests_properties(Core.SystemProfilerUnitTests PROPERTIES
    LABELS "core;unit;hardware"
)

# ------------------------------------------------------------------------------
# CLI & End-to-End Smoke Tests on the Primary `Umgebung` Executable
# ------------------------------------------------------------------------------
# Verifies `--profile-only` runs full hardware discovery, CUDA device binding,
# and runtime adaptation cleanly in both GUI and Headless builds.
add_test(
    NAME Core.ExecutableProfileSmokeTest
    COMMAND $<TARGET_FILE:Umgebung> --profile-only
)
set_tests_properties(Core.ExecutableProfileSmokeTest PROPERTIES
    LABELS "core;smoke;hardware"
)

# Verifies `--help` prints CLI documentation and exits with code 0.
add_test(
    NAME Core.ExecutableHelpTest
    COMMAND $<TARGET_FILE:Umgebung> --help
)
set_tests_properties(Core.ExecutableHelpTest PROPERTIES
    LABELS "core;smoke;cli"
)

# Verifies `--version` prints build mode/version and exits with code 0.
add_test(
    NAME Core.ExecutableVersionTest
    COMMAND $<TARGET_FILE:Umgebung> --version
)
set_tests_properties(Core.ExecutableVersionTest PROPERTIES
    LABELS "core;smoke;cli"
)

# ------------------------------------------------------------------------------
# Optional Formal Proof Verification Test (Lean 4 via Lake)
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
