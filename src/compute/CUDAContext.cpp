/**
 * @file CUDAContext.cpp
 * @brief Implementation of the RAII CUDA compute context, non-blocking stream
 *        pool, and atomic VRAM budget governor (`umgebung::compute::CUDAContext`).
 *
 * @details
 * Connects `core::SystemProfiler`'s startup hardware adaptation directly to live
 * NVIDIA CUDA runtime resources: binds the selected primary GPU, constructs a
 * pool of non-blocking `cudaStream_t` queues sized to the GPU's SM width, and
 * maintains lock-free accounting of simulation VRAM allocations.
 */

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/compute/CudaUtils.hpp"

#include <algorithm>

#include <spdlog/spdlog.h>

namespace umgebung::compute {

    CUDAContext::~CUDAContext() {
        Shutdown();
    }

    bool CUDAContext::Initialize(
        const core::HardwareCapabilities& caps,
        const core::HardwareAdaptationPlan& plan
    ) {
        if (m_initialized) {
            spdlog::warn("CUDAContext::Initialize called on an already initialized CUDAContext.");
            return false;
        }

        if (!plan.computeReady || plan.primaryGpuIndex < 0) {
            spdlog::error("CUDAContext::Initialize failed: HardwareAdaptationPlan indicates no compute-ready GPU.");
            return false;
        }

        // Locate the matching `GPUDeviceInfo` descriptor for metadata & diagnostics.
        bool foundGpuInfo = false;
        for (const auto& gpu : caps.gpus) {
            if (gpu.deviceIndex == plan.primaryGpuIndex) {
                m_primaryGpuInfo = gpu;
                foundGpuInfo = true;
                break;
            }
        }
        if (!foundGpuInfo) {
            spdlog::error(
                "CUDAContext::Initialize failed: primary GPU index {} not found in HardwareCapabilities.",
                plan.primaryGpuIndex
            );
            return false;
        }

        // Bind the calling thread's CUDA runtime state to the adapted primary GPU.
        if (!UMGEBUNG_CUDA_CHECK(cudaSetDevice(plan.primaryGpuIndex))) {
            return false;
        }

        m_deviceIndex = plan.primaryGpuIndex;
        m_recommendedBlockSize = std::max(32, plan.recommendedCudaBlockSize);
        m_vramBudgetBytes = plan.simulationVRAMBudgetBytes;
        m_trackedAllocatedBytes.store(0U, std::memory_order_release);

        // Allocate the adapted number of non-blocking CUDA streams (at least 1).
        // `cudaStreamNonBlocking` avoids implicit synchronization with the legacy
        // NULL stream (stream 0), allowing future H2D/kernel/D2H overlap.
        const std::size_t streamCount = static_cast<std::size_t>(std::max(1, plan.recommendedCudaStreams));
        m_streams.reserve(streamCount);

        for (std::size_t i = 0U; i < streamCount; ++i) {
            cudaStream_t stream = nullptr;
            if (!UMGEBUNG_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking))) {
                spdlog::error("CUDAContext: Failed creating CUDA stream {} of {}. Rolling back.", i, streamCount);
                Shutdown();
                return false;
            }
            m_streams.push_back(stream);
        }

        m_initialized = true;
        spdlog::info(
            "CUDAContext initialized on GPU [{}] '{}' (SM {}.{}, {} streams, blockSize={}, VRAM budget={}).",
            m_deviceIndex,
            m_primaryGpuInfo.name,
            m_primaryGpuInfo.computeCapabilityMajor,
            m_primaryGpuInfo.computeCapabilityMinor,
            m_streams.size(),
            m_recommendedBlockSize,
            FormatBytes(m_vramBudgetBytes)
        );
        return true;
    }

    void CUDAContext::Shutdown() noexcept {
        if (m_deviceIndex >= 0) {
            (void)cudaSetDevice(m_deviceIndex);
        }

        // Warn if any tracked `DeviceBuffer<T>` allocations outlived the context.
        const std::size_t leakedBytes = m_trackedAllocatedBytes.load(std::memory_order_acquire);
        if (leakedBytes > 0U) {
            spdlog::warn(
                "CUDAContext::Shutdown detected {} of tracked VRAM still reserved; "
                "ensure all DeviceBuffer<T> objects are destroyed before CUDAContext::Shutdown().",
                FormatBytes(leakedBytes)
            );
        }

        // Synchronize and destroy every owned CUDA stream cleanly.
        for (cudaStream_t stream : m_streams) {
            if (stream != nullptr) {
                (void)cudaStreamSynchronize(stream);
                (void)cudaStreamDestroy(stream);
            }
        }
        m_streams.clear();

        if (m_initialized) {
            spdlog::info("CUDAContext on GPU [{}] shut down cleanly.", m_deviceIndex);
        }

        m_initialized = false;
        m_deviceIndex = -1;
        m_vramBudgetBytes = 0U;
        m_trackedAllocatedBytes.store(0U, std::memory_order_release);
    }

    bool CUDAContext::MakeCurrent() const noexcept {
        if (!m_initialized || m_deviceIndex < 0) {
            return false;
        }
        return UMGEBUNG_CUDA_CHECK(cudaSetDevice(m_deviceIndex));
    }

    bool CUDAContext::SynchronizeStream(std::size_t streamIndex) const noexcept {
        if (!m_initialized || streamIndex >= m_streams.size()) {
            return false;
        }
        if (!MakeCurrent()) {
            return false;
        }
        return UMGEBUNG_CUDA_CHECK(cudaStreamSynchronize(m_streams[streamIndex]));
    }

    bool CUDAContext::SynchronizeDevice() const noexcept {
        if (!MakeCurrent()) {
            return false;
        }
        return UMGEBUNG_CUDA_CHECK(cudaDeviceSynchronize());
    }

    bool CUDAContext::TryReserveVRAM(std::size_t bytes) noexcept {
        if (!m_initialized) {
            return false;
        }
        if (bytes == 0U) {
            return true;
        }

        // Lock-free atomic reservation enforcing `current + bytes <= m_vramBudgetBytes`
        // while guarding against unsigned integer overflow on `current + bytes`.
        std::size_t current = m_trackedAllocatedBytes.load(std::memory_order_relaxed);
        while (true) {
            if (bytes > m_vramBudgetBytes || current > (m_vramBudgetBytes - bytes)) {
                spdlog::error(
                    "CUDAContext: VRAM budget exceeded! Requested {}, tracked used {}, budget {}.",
                    FormatBytes(bytes),
                    FormatBytes(current),
                    FormatBytes(m_vramBudgetBytes)
                );
                return false;
            }

            const std::size_t desired = current + bytes;
            if (m_trackedAllocatedBytes.compare_exchange_weak(
                    current,
                    desired,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed
                )) {
                return true;
            }
        }
    }

    void CUDAContext::ReleaseVRAM(std::size_t bytes) noexcept {
        if (bytes == 0U) {
            return;
        }

        // Saturating atomic subtraction prevents unsigned underflow.
        std::size_t current = m_trackedAllocatedBytes.load(std::memory_order_relaxed);
        while (true) {
            const std::size_t desired = (bytes >= current) ? 0U : (current - bytes);
            if (m_trackedAllocatedBytes.compare_exchange_weak(
                    current,
                    desired,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed
                )) {
                return;
            }
        }
    }

    bool CUDAContext::QueryLiveDeviceVRAM(
        std::size_t& outFreeBytes,
        std::size_t& outTotalBytes
    ) const noexcept {
        outFreeBytes = 0U;
        outTotalBytes = 0U;
        if (!MakeCurrent()) {
            return false;
        }
        return UMGEBUNG_CUDA_CHECK(cudaMemGetInfo(&outFreeBytes, &outTotalBytes));
    }

    cudaStream_t CUDAContext::GetStream(std::size_t streamIndex) const noexcept {
        if (!m_initialized || m_streams.empty()) {
            return nullptr;
        }
        return m_streams[streamIndex % m_streams.size()];
    }

} // namespace umgebung::compute
