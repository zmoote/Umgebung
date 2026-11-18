# ==============================================================================
# @file tests/CMakeLists.txt
# @brief CTest verification, conservation-law, and regression test registration.
#
# @details
# Registers automated tests executed via `ctest --preset <preset-name>`.
# As Mainstream Physics solvers (`models/mainstream`), Unified Research Models
# (`models/unified`), and Lean 4 reference oracles (`formal/`) come online,
# their corresponding analytical benchmarks and invariant checks are added here.
# ==============================================================================

# ------------------------------------------------------------------------------
# Smoke Test: Verify the primary Umgebung executable launches and exits cleanly.
# ------------------------------------------------------------------------------
add_test(
    NAME Core.ExecutableSmokeTest
    COMMAND $<TARGET_FILE:Umgebung>
)
set_tests_properties(Core.ExecutableSmokeTest PROPERTIES
    LABELS "core;smoke"
)

# ------------------------------------------------------------------------------
# Optional Formal Proof Verification Test: Runs `lake build` under CTest when
# `UMGEBUNG_VERIFY_PROOFS=ON` and a Lean 4 Lake workspace is configured.
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
