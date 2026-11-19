/**
 * @file DeviceBuffer.hpp
 * @brief Move-only RAII GPU device memory container integrated with
 *        `umgebung::compute::CUDAContext` VRAM budget tracking.
 *
 * @details
 * Provides a strongly typed, Structure-of-Arrays (SoA)-friendly GPU array
 * abstraction (`DeviceBuffer<T>`) that eliminates manual `cudaMalloc`/`cudaFree`
 * boilerplate and guarantees:
 *   1. **Compile-time safety:** `T` must be trivially copyable so raw byte
 *      DMA transfers between host RAM and NVIDIA VRAM are well-defined.
 *   2. **Budget enforcement:** Every `Allocate()` atomically reserves its byte
 *      footprint from `CUDAContext::TryReserveVRAM()` *before* invoking
 *      `cudaMalloc`, and rolls back the reservation automatically if the CUDA
 *      driver rejects the allocation.
 *   3. **Deterministic RAII teardown:** `Release()` and `~DeviceBuffer()` free
 *      device memory and immediately credit the released bytes back to
 *      `CUDAContext::ReleaseVRAM()`.
 *   4. **Sync and async stream transfers:** Supports synchronous and stream-ordered
 *      host-to-device (`CopyFromHost`), device-to-host (`CopyToHost`), and
 *      device-side zeroing (`ZeroFill`).
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/compute/CudaUtils.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

namespace umgebung::compute {

    /**
     * @class DeviceBuffer
     * @tparam T Trivially copyable element type stored contiguously in GPU global VRAM
     *           (e.g., `double` for FP64 physics SoA arrays, `float`, `std::uint32_t`).
     *
     * @brief RAII owning handle for a contiguous device-side array of `T`.
     */
    template <typename T>
    class DeviceBuffer final {
        static_assert(
            std::is_trivially_copyable_v<T>,
            "umgebung::compute::DeviceBuffer<T> requires a trivially copyable element type."
        );

    public:
        /** @brief Constructs an empty, unallocated `DeviceBuffer`. */
        DeviceBuffer() = default;

        /** @brief Destructor; releases GPU memory and updates `CUDAContext` VRAM accounting. */
        ~DeviceBuffer() {
            Release();
        }

        // Non-copyable to prevent accidental double-free or hidden expensive D2D copies.
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        /**
         * @brief Move constructor transferring GPU pointer ownership and VRAM tracking.
         *
         * @param other Source buffer left in a clean empty state.
         */
        DeviceBuffer(DeviceBuffer&& other) noexcept
            : m_context(std::exchange(other.m_context, nullptr))
            , m_devicePtr(std::exchange(other.m_devicePtr, nullptr))
            , m_count(std::exchange(other.m_count, 0U))
            , m_sizeBytes(std::exchange(other.m_sizeBytes, 0U)) {}

        /**
         * @brief Move assignment operator; frees any existing allocation before
         *        taking ownership from `other`.
         *
         * @param other Source buffer left in a clean empty state.
         * @return Reference to `*this`.
         */
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            if (this != &other) {
                Release();
                m_context = std::exchange(other.m_context, nullptr);
                m_devicePtr = std::exchange(other.m_devicePtr, nullptr);
                m_count = std::exchange(other.m_count, 0U);
                m_sizeBytes = std::exchange(other.m_sizeBytes, 0U);
            }
            return *this;
        }

        /**
         * @brief Allocates space for `count` elements of `T` in GPU global VRAM
         *        under `context`'s adapted VRAM budget.
         *
         * @details
         * Releases any prior allocation held by `*this` first so reallocations
         * never temporarily double-count budget headroom. Calling `Allocate(ctx, 0)`
         * releases any existing memory and returns `true`.
         *
         * @param context Initialized `CUDAContext` owning the target GPU and budget.
         * @param count   Number of `T` elements to allocate.
         * @return `true` on success; `false` if context is uninitialized, VRAM
         *         budget would be exceeded, or `cudaMalloc` failed.
         */
        [[nodiscard]] bool Allocate(CUDAContext& context, std::size_t count) {
            // Free existing buffer first so reallocation credits old VRAM before reserving new VRAM.
            Release();

            if (!context.IsInitialized()) {
                spdlog::error("DeviceBuffer::Allocate failed: CUDAContext is not initialized.");
                return false;
            }

            if (count == 0U) {
                return true;
            }

            // Guard against size multiplication overflow.
            if (count > (static_cast<std::size_t>(-1) / sizeof(T))) {
                spdlog::error("DeviceBuffer::Allocate failed: element count {} overflows size_t.", count);
                return false;
            }

            const std::size_t requestedBytes = count * sizeof(T);
            if (!context.TryReserveVRAM(requestedBytes)) {
                return false;
            }

            if (!context.MakeCurrent()) {
                context.ReleaseVRAM(requestedBytes);
                return false;
            }

            T* rawPtr = nullptr;
            if (!UMGEBUNG_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&rawPtr), requestedBytes))) {
                context.ReleaseVRAM(requestedBytes);
                return false;
            }

            m_context = &context;
            m_devicePtr = rawPtr;
            m_count = count;
            m_sizeBytes = requestedBytes;
            return true;
        }

        /**
         * @brief Frees the underlying GPU allocation (if any) and returns its
         *        byte count to the owning `CUDAContext` budget.
         *
         * @details Idempotent; safe to invoke on an already-empty buffer.
         */
        void Release() noexcept {
            if (m_devicePtr != nullptr) {
                if (m_context != nullptr && m_context->IsInitialized()) {
                    (void)m_context->MakeCurrent();
                }
                (void)cudaFree(m_devicePtr);
                m_devicePtr = nullptr;
            }

            if (m_context != nullptr && m_sizeBytes > 0U) {
                m_context->ReleaseVRAM(m_sizeBytes);
            }

            m_context = nullptr;
            m_count = 0U;
            m_sizeBytes = 0U;
        }

        /**
         * @brief Zero-fills the entire allocated device buffer (`0x00` bytes).
         *
         * @param stream Optional CUDA stream for asynchronous zeroing (`nullptr` = synchronous).
         * @return `true` on success; `false` on CUDA error.
         */
        [[nodiscard]] bool ZeroFill(cudaStream_t stream = nullptr) {
            if (m_devicePtr == nullptr || m_sizeBytes == 0U) {
                return true;
            }
            if (m_context == nullptr || !m_context->MakeCurrent()) {
                return false;
            }
            if (stream != nullptr) {
                return UMGEBUNG_CUDA_CHECK(cudaMemsetAsync(m_devicePtr, 0, m_sizeBytes, stream));
            }
            return UMGEBUNG_CUDA_CHECK(cudaMemset(m_devicePtr, 0, m_sizeBytes));
        }

        /**
         * @brief Uploads `count` elements from a host pointer into GPU VRAM.
         *
         * @param hostSrc Pointer to contiguous host memory containing at least `count` elements.
         * @param count   Number of elements to copy (must satisfy `count <= Count()`).
         * @param stream  Optional CUDA stream; when non-null, executes `cudaMemcpyAsync`.
         * @return `true` on success; `false` on bounds mismatch or CUDA transfer failure.
         */
        [[nodiscard]] bool CopyFromHost(
            const T* hostSrc,
            std::size_t count,
            cudaStream_t stream = nullptr
        ) {
            if (count == 0U) {
                return true;
            }
            if (hostSrc == nullptr || m_devicePtr == nullptr || count > m_count) {
                spdlog::error(
                    "DeviceBuffer::CopyFromHost invalid arguments (count={}, capacity={}).",
                    count,
                    m_count
                );
                return false;
            }
            if (m_context == nullptr || !m_context->MakeCurrent()) {
                return false;
            }

            const std::size_t copyBytes = count * sizeof(T);
            if (stream != nullptr) {
                return UMGEBUNG_CUDA_CHECK(
                    cudaMemcpyAsync(m_devicePtr, hostSrc, copyBytes, cudaMemcpyHostToDevice, stream)
                );
            }
            return UMGEBUNG_CUDA_CHECK(
                cudaMemcpy(m_devicePtr, hostSrc, copyBytes, cudaMemcpyHostToDevice)
            );
        }

        /**
         * @brief Convenience overload uploading an entire `std::vector<T>` to GPU VRAM.
         *
         * @param hostVec Host vector whose `size()` must not exceed `Count()`.
         * @param stream  Optional CUDA stream for async upload.
         * @return `true` on success; `false` on bounds mismatch or CUDA failure.
         */
        [[nodiscard]] bool CopyFromHost(
            const std::vector<T>& hostVec,
            cudaStream_t stream = nullptr
        ) {
            return CopyFromHost(hostVec.data(), hostVec.size(), stream);
        }

        /**
         * @brief Downloads `count` elements from GPU VRAM into a host memory buffer.
         *
         * @param hostDst Destination host pointer with room for at least `count` elements.
         * @param count   Number of elements to download (`count <= Count()`).
         * @param stream  Optional CUDA stream; when non-null, executes `cudaMemcpyAsync`
         *                (caller must synchronize `stream` before reading `hostDst`).
         * @return `true` on success; `false` on bounds mismatch or CUDA failure.
         */
        [[nodiscard]] bool CopyToHost(
            T* hostDst,
            std::size_t count,
            cudaStream_t stream = nullptr
        ) const {
            if (count == 0U) {
                return true;
            }
            if (hostDst == nullptr || m_devicePtr == nullptr || count > m_count) {
                spdlog::error(
                    "DeviceBuffer::CopyToHost invalid arguments (count={}, capacity={}).",
                    count,
                    m_count
                );
                return false;
            }
            if (m_context == nullptr || !m_context->MakeCurrent()) {
                return false;
            }

            const std::size_t copyBytes = count * sizeof(T);
            if (stream != nullptr) {
                return UMGEBUNG_CUDA_CHECK(
                    cudaMemcpyAsync(hostDst, m_devicePtr, copyBytes, cudaMemcpyDeviceToHost, stream)
                );
            }
            return UMGEBUNG_CUDA_CHECK(
                cudaMemcpy(hostDst, m_devicePtr, copyBytes, cudaMemcpyDeviceToHost)
            );
        }

        /**
         * @brief Convenience overload resizing `outHostVec` to `Count()` and
         *        downloading the entire buffer from GPU VRAM.
         *
         * @param[out] outHostVec Host vector resized and populated with GPU data.
         * @param      stream     Optional CUDA stream for async transfer.
         * @return `true` on success; `false` on CUDA transfer failure.
         */
        [[nodiscard]] bool CopyToHost(
            std::vector<T>& outHostVec,
            cudaStream_t stream = nullptr
        ) const {
            outHostVec.resize(m_count);
            return CopyToHost(outHostVec.data(), m_count, stream);
        }

        /** @brief Returns the raw mutable device pointer (usable directly in CUDA kernel launches). */
        [[nodiscard]] T* Data() noexcept { return m_devicePtr; }

        /** @brief Returns the raw read-only device pointer. */
        [[nodiscard]] const T* Data() const noexcept { return m_devicePtr; }

        /** @brief Returns the number of allocated `T` elements. */
        [[nodiscard]] std::size_t Count() const noexcept { return m_count; }

        /** @brief Returns the total allocated size in bytes (`Count() * sizeof(T)`). */
        [[nodiscard]] std::size_t SizeBytes() const noexcept { return m_sizeBytes; }

        /** @brief Returns `true` when zero elements are allocated. */
        [[nodiscard]] bool Empty() const noexcept { return m_count == 0U; }

        /** @brief Returns `true` when backing GPU memory is currently owned. */
        [[nodiscard]] bool IsAllocated() const noexcept { return m_devicePtr != nullptr; }

    private:
        /** @brief Non-owning back-pointer to the `CUDAContext` tracking this allocation's VRAM. */
        CUDAContext* m_context{nullptr};

        /** @brief Device global-memory pointer returned by `cudaMalloc`. */
        T* m_devicePtr{nullptr};

        /** @brief Number of elements of type `T` currently allocated. */
        std::size_t m_count{0U};

        /** @brief Total allocated footprint in bytes (`m_count * sizeof(T)`). */
        std::size_t m_sizeBytes{0U};
    };

} // namespace umgebung::compute
