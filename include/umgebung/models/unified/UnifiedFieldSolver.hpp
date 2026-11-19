/**
 * @file UnifiedFieldSolver.hpp
 * @brief Compatibility alias forwarding `UnifiedFieldSolver` to the refactored
 *        Tier 3 `UnifiedSynthesisSolver` (`Unified::Synthesis`).
 */

#pragma once

#include "umgebung/models/unified/UnifiedSynthesisSolver.hpp"

namespace umgebung::models::unified {

    /** @brief Backwards-compatible alias for `UnifiedSynthesisSolver`. */
    using UnifiedFieldSolver = UnifiedSynthesisSolver;

} // namespace umgebung::models::unified
