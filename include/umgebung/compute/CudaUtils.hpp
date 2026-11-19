/**
 * @file CudaUtils.hpp
 * @brief Low-level CUDA error checking, 1D launch geometry helpers, and byte
 *        formatting utilities shared across all Umgebung GPU modules.
 *
 * @details
 * Provides deterministic, zero-overhead diagnostics for CUDA Runtime API calls
 * and kernel launches. Every CUDA call in `compute/`, `models/mainstream/`,
 * and `models/unified/` should route through `CheckCudaError` or
 * `UMGEBUNG_CUDA_CHECK` so device failures surface immediately with exact
 * source file, line number, expression text, and CUDA error string via `spdlog`.
 *
 * @note
 * Intentionally header-only and safe to include from both ISO C++20 host
 * translation units (`.cpp`) and CUDA device translation units (`.cu`).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

/**
 * @namespace umgebung::compute
 * @brief GPU compute infrastructure: CUDA context & stream management,
 *        budget-tracked RAII device buffers, and multi-GPU dispatch utilities.
 */
namespace umgebung::compute {

    /**
     * @brief Validates a `cudaError_t` return code and logs rich diagnostics
     *        through `spdlog` on failure.
     *
     * @param status   CUDA runtime status returned by a CUDA API call.
     * @param exprText Stringified C/CUDA call expression for log traceability.
     * @param filePath Source file path (`__FILE__`) where the call occurred.
     * @param lineNum  Source line number (`__LINE__`) where the call occurred.
     *
     * @return `true` if `status == cudaSuccess`; `false` otherwise.
     */
    [[nodiscard]] inline bool CheckCudaError(
        cudaError_t status,
        const char* exprText,
        const char* filePath,
        int lineNum
    ) noexcept {
        if (status == cudaSuccess) {
            return true;
        }

        spdlog::error(
            "[CUDA Error] {} failed at {}:{} -> {} ({})",
            exprText ? exprText : "(unknown expression)",
            filePath ? filePath : "(unknown file)",
            lineNum,
            cudaGetErrorString(status),
            static_cast<int>(status)
        );
        return false;
    }

    /**
     * @brief Validates a `cudaError_t` return code and throws `std::runtime_error`
     *        on failure after logging through `spdlog`.
     *
     * @param status   CUDA runtime status returned by a CUDA API call.
     * @param exprText Stringified call expression.
     * @param filePath Source file path (`__FILE__`).
     * @param lineNum  Source line number (`__LINE__`).
     *
     * @throws std::runtime_error If `status != cudaSuccess`.
     */
    inline void CheckCudaErrorOrThrow(
        cudaError_t status,
        const char* exprText,
        const char* filePath,
        int lineNum
    ) {
        if (!CheckCudaError(status, exprText, filePath, lineNum)) {
            throw std::runtime_error(
                std::string("CUDA error in ") +
                (exprText ? exprText : "unknown") + " at " +
                (filePath ? filePath : "unknown") + ":" + std::to_string(lineNum) +
                " -> " + cudaGetErrorString(status)
            );
        }
    }

    /**
     * @brief Computes the ceiling 1D CUDA grid dimension (`blocksPerGrid`) needed
     *        to cover `elementCount` work items with `blockSize` threads per block.
     *
     * @details
     * Implements integer ceiling division:
     *   $$\text{blocks} = \left\lceil \frac{\text{elementCount}}{\text{blockSize}} \right\rceil = \frac{\text{elementCount} + \text{blockSize} - 1}{\text{blockSize}}$$
     * Returns `0` cleanly when `elementCount == 0` so callers can guard against
     * empty kernel launches.
     *
     * @param elementCount Total number of elements/entities to process on the GPU.
     * @param blockSize    Number of threads per CUDA block (must be `> 0`).
     * @return Number of 1D thread blocks required, or `0` if `elementCount == 0` or `blockSize <= 0`.
     */
    [[nodiscard]] constexpr unsigned int ComputeGridSize1D(
        std::size_t elementCount,
        int blockSize
    ) noexcept {
        if (elementCount == 0U || blockSize <= 0) {
            return 0U;
        }
        const std::size_t bs = static_cast<std::size_t>(blockSize);
        return static_cast<unsigned int>((elementCount + bs - 1U) / bs);
    }

    /**
     * @brief Formats a byte count into a human-readable binary unit string
     *        (`B`, `KiB`, `MiB`, or `GiB`) for diagnostic logs and UI telemetry.
     *
     * @param bytes Raw byte count to format.
     * @return Formatted string (e.g., `"256.00 MiB"` or `"19.21 GiB"`).
     */
    [[nodiscard]] inline std::string FormatBytes(std::size_t bytes) {
        constexpr double kKiB = 1024.0;
        constexpr double kMiB = 1024.0 * 1024.0;
        constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

        char buffer[64] = {};
        if (static_cast<double>(bytes) >= kGiB) {
            std::snprintf(buffer, sizeof(buffer), "%.2f GiB", static_cast<double>(bytes) / kGiB);
        } else if (static_cast<double>(bytes) >= kMiB) {
            std::snprintf(buffer, sizeof(buffer), "%.2f MiB", static_cast<double>(bytes) / kMiB);
        } else if (static_cast<double>(bytes) >= kKiB) {
            std::snprintf(buffer, sizeof(buffer), "%.2f KiB", static_cast<double>(bytes) / kKiB);
        } else {
            std::snprintf(buffer, sizeof(buffer), "%zu B", bytes);
        }
        return std::string(buffer);
    }

} // namespace umgebung::compute

/**
 * @def UMGEBUNG_CUDA_CHECK(call)
 * @brief Evaluates a CUDA runtime API expression and returns `bool` (`true` on
 *        `cudaSuccess`, `false` after logging file/line/error diagnostics).
 */
#define UMGEBUNG_CUDA_CHECK(call) \
    (::umgebung::compute::CheckCudaError((call), #call, __FILE__, __LINE__))

/**
 * @def UMGEBUNG_CUDA_CHECK_LAST_ERROR()
 * @brief Inspects `cudaGetLastError()` immediately following a `<<<...>>>`
 *        kernel launch to catch invalid launch configurations or device faults.
 */
#define UMGEBUNG_CUDA_CHECK_LAST_ERROR() \
    (::umgebung::compute::CheckCudaError(::cudaGetLastError(), "cudaGetLastError()", __FILE__, __LINE__))
