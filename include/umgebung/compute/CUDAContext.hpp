/**
 * @file CUDAContext.hpp
 * @brief RAII CUDA runtime context, non-blocking stream pool, and atomic VRAM
 *        budget governor driven by `SystemProfiler`'s `HardwareAdaptationPlan`.
 *
 * @details
 * `umgebung::compute::CUDAContext` is the foundational GPU compute resource
 * shared across all simulation models (`models/mainstream/`, `models/unified/`)
 * in both Desktop Interactive GUI builds and Headless HPC builds.
 *
 * Key responsibilities:
 *   1. **Device Binding:** Activates the primary NVIDIA GPU chosen by
 *      `core::SystemProfiler::BuildAdaptationPlan()` (`cudaSetDevice`).
 *   2. **Non-Blocking Stream Pool:** Allocates `plan.recommendedCudaStreams`
 *      non-blocking CUDA streams (`cudaStreamNonBlocking`) so kernels and
 *      host-device memory copies never implicitly serialize on the legacy
 *      default stream 0.
 *   3. **Atomic VRAM Budget Enforcement:** Enforces the hardware-adapted
 *      `plan.simulationVRAMBudgetBytes` ceiling via lock-free atomic
 *      compare-exchange (`TryReserveVRAM` / `ReleaseVRAM`), preventing desktop
 *      runs from starving the Vulkan compositor or HPC jobs from hitting
 *      unexpected out-of-memory faults mid-sweep.
 *
 * @note
 * Zero dependency on GLFW, Vulkan, NVRHI, or Dear ImGui. Compiles identically
 * on headless HPC clusters (ARCSIM, Premise, SPORC) and desktop workstations.
 */

#pragma once

#include "umgebung/core/SystemProfiler.hpp"

#include <atomic>
#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

namespace umgebung::compute {

    /**
     * @class CUDAContext
     * @brief Owns the active CUDA device binding, a pool of non-blocking CUDA
     *        streams, and an atomic VRAM allocation budget tracker.
     */
    class CUDAContext final {
    public:
        /** @brief Constructs an uninitialized `CUDAContext`; invoke `Initialize()` before use. */
        CUDAContext() = default;

        /** @brief Destructor; synchronizes the device and releases all owned CUDA streams. */
        ~CUDAContext();

        // Non-copyable and non-movable because `DeviceBuffer<T>` instances hold
        // a non-owning pointer back to their owning `CUDAContext` for VRAM accounting.
        CUDAContext(const CUDAContext&) = delete;
        CUDAContext& operator=(const CUDAContext&) = delete;
        CUDAContext(CUDAContext&&) = delete;
        CUDAContext& operator=(CUDAContext&&) = delete;

        /**
         * @brief Initializes the CUDA compute context on the primary GPU selected
         *        by `plan`, creates the non-blocking stream pool, and configures
         *        the VRAM budget ceiling.
         *
         * @param caps Hardware snapshot captured at startup by `SystemProfiler`.
         * @param plan Adaptation plan containing primary GPU index, stream count,
         *             recommended block size, and simulation VRAM budget.
         *
         * @return `true` if the device was bound and all streams were created;
         *         `false` on invalid adaptation plan or any CUDA driver/runtime failure.
         *
         * @pre  `IsInitialized() == false` and `plan.computeReady == true`.
         * @post On `true`, `GetStreamCount() >= 1` and active CUDA device equals `plan.primaryGpuIndex`.
         */
        [[nodiscard]] bool Initialize(
            const core::HardwareCapabilities& caps,
            const core::HardwareAdaptationPlan& plan
        );

        /**
         * @brief Synchronizes all owned streams and destroys them. Logs a warning
         *        if any `DeviceBuffer<T>` reservations are still unreleased at teardown.
         *
         * @details
         * Idempotent: safe to call explicitly during `Application::Shutdown()`
         * and again inside `~CUDAContext()`.
         */
        void Shutdown() noexcept;

        /**
         * @brief Ensures the calling host thread's active CUDA device matches
         *        this context's bound `m_deviceIndex`.
         *
         * @details
         * Important when host worker threads or CTest threads invoke CUDA APIs,
         * because `cudaSetDevice` is thread-local in the CUDA Runtime API.
         *
         * @return `true` on success; `false` if uninitialized or `cudaSetDevice` failed.
         */
        [[nodiscard]] bool MakeCurrent() const noexcept;

        /**
         * @brief Blocks the calling host thread until all enqueued work on the
         *        specified managed stream has completed.
         *
         * @param streamIndex Zero-based index into the context's stream pool.
         * @return `true` on `cudaSuccess`; `false` on out-of-range index or device error.
         */
        [[nodiscard]] bool SynchronizeStream(std::size_t streamIndex = 0U) const noexcept;

        /**
         * @brief Blocks the calling host thread until all work across the entire
         *        bound CUDA device has completed (`cudaDeviceSynchronize`).
         *
         * @return `true` on `cudaSuccess`; `false` on device error.
         */
        [[nodiscard]] bool SynchronizeDevice() const noexcept;

        /**
         * @brief Atomically reserves `bytes` against this context's adapted
         *        simulation VRAM budget before issuing a `cudaMalloc`.
         *
         * @details
         * Uses a lock-free `compare_exchange_weak` loop so concurrent host
         * threads allocating GPU buffers cannot race past `m_vramBudgetBytes`.
         *
         * @param bytes Number of bytes requested for GPU device allocation.
         * @return `true` if the reservation succeeded within budget; `false` if
         *         granting `bytes` would exceed `m_vramBudgetBytes` (no change made).
         */
        [[nodiscard]] bool TryReserveVRAM(std::size_t bytes) noexcept;

        /**
         * @brief Releases a previously reserved byte count back to the VRAM budget.
         *
         * @details
         * Uses saturating atomic subtraction so defensive double-releases cannot
         * wrap `m_trackedAllocatedBytes` around `std::size_t` underflow.
         *
         * @param bytes Number of bytes freed via `cudaFree`.
         */
        void ReleaseVRAM(std::size_t bytes) noexcept;

        /**
         * @brief Queries the live free and total VRAM currently reported by the
         *        CUDA driver on the bound GPU (`cudaMemGetInfo`).
         *
         * @param[out] outFreeBytes  Populated with currently free VRAM in bytes.
         * @param[out] outTotalBytes Populated with total GPU VRAM in bytes.
         * @return `true` on success; `false` if uninitialized or driver query failed.
         */
        [[nodiscard]] bool QueryLiveDeviceVRAM(
            std::size_t& outFreeBytes,
            std::size_t& outTotalBytes
        ) const noexcept;

        /** @brief Returns `true` if `Initialize()` succeeded and `Shutdown()` has not run. */
        [[nodiscard]] bool IsInitialized() const noexcept { return m_initialized; }

        /** @brief Returns the CUDA ordinal (`0..N-1`) bound by this context (`-1` if uninitialized). */
        [[nodiscard]] int GetDeviceIndex() const noexcept { return m_deviceIndex; }

        /** @brief Returns the cached `GPUDeviceInfo` for the bound primary GPU. */
        [[nodiscard]] const core::GPUDeviceInfo& GetPrimaryGpuInfo() const noexcept { return m_primaryGpuInfo; }

        /** @brief Returns the hardware-adapted default 1D CUDA block size (threads/block). */
        [[nodiscard]] int GetRecommendedBlockSize() const noexcept { return m_recommendedBlockSize; }

        /** @brief Returns the number of owned non-blocking CUDA streams in the pool. */
        [[nodiscard]] std::size_t GetStreamCount() const noexcept { return m_streams.size(); }

        /**
         * @brief Retrieves a non-blocking CUDA stream handle from the pool.
         *
         * @param streamIndex Zero-based stream index; wraps modulo `GetStreamCount()`
         *                    if out of bounds when initialized, or returns `nullptr` if uninitialized.
         * @return Valid `cudaStream_t` handle, or `nullptr` if uninitialized.
         */
        [[nodiscard]] cudaStream_t GetStream(std::size_t streamIndex = 0U) const noexcept;

        /** @brief Returns the total simulation VRAM ceiling (bytes) configured by `HardwareAdaptationPlan`. */
        [[nodiscard]] std::size_t GetVRAMBudgetBytes() const noexcept { return m_vramBudgetBytes; }

        /** @brief Returns the number of VRAM bytes currently reserved by active `DeviceBuffer<T>` allocations. */
        [[nodiscard]] std::size_t GetTrackedAllocatedBytes() const noexcept {
            return m_trackedAllocatedBytes.load(std::memory_order_acquire);
        }

        /** @brief Returns the remaining available VRAM headroom within the simulation budget. */
        [[nodiscard]] std::size_t GetRemainingVRAMBudgetBytes() const noexcept {
            const std::size_t used = GetTrackedAllocatedBytes();
            return (used < m_vramBudgetBytes) ? (m_vramBudgetBytes - used) : 0U;
        }

    private:
        /** @brief True after `Initialize()` completes cleanly. */
        bool m_initialized{false};

        /** @brief Bound primary CUDA device ordinal (`-1` when uninitialized). */
        int m_deviceIndex{-1};

        /** @brief Snapshot of the bound GPU's architectural properties. */
        core::GPUDeviceInfo m_primaryGpuInfo{};

        /** @brief Adapted default 1D CUDA thread block size copied from `HardwareAdaptationPlan`. */
        int m_recommendedBlockSize{256};

        /** @brief Maximum simulation VRAM bytes permitted on this GPU by `HardwareAdaptationPlan`. */
        std::size_t m_vramBudgetBytes{0U};

        /** @brief Lock-free counter of VRAM bytes currently held by tracked `DeviceBuffer<T>` instances. */
        std::atomic<std::size_t> m_trackedAllocatedBytes{0U};

        /** @brief Pool of owned non-blocking CUDA streams created via `cudaStreamCreateWithFlags`. */
        std::vector<cudaStream_t> m_streams;
    };

} // namespace umgebung::compute
