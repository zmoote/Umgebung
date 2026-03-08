#pragma once
#include <cuda.h>
#include <vector>
#include "umgebung/util/LogMacros.hpp"

namespace Umgebung::util {

    /**
     * @brief A simple RAII wrapper for CUDA device memory using the Driver API.
     * @tparam T The type of data stored in device memory.
     */
    template<typename T>
    class DeviceBuffer {
    public:
        DeviceBuffer() : ptr_(0), count_(0) {}
        
        explicit DeviceBuffer(size_t count) : ptr_(0), count_(0) {
            allocate(count);
        }

        ~DeviceBuffer() {
            free();
        }

        // Disable copy
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        // Enable move
        DeviceBuffer(DeviceBuffer&& other) noexcept 
            : ptr_(other.ptr_), count_(other.count_) {
            other.ptr_ = 0;
            other.count_ = 0;
        }

        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            if (this != &other) {
                free();
                ptr_ = other.ptr_;
                count_ = other.count_;
                other.ptr_ = 0;
                other.count_ = 0;
            }
            return *this;
        }

        void allocate(size_t count) {
            if (count == count_ && ptr_ != 0) return;
            free();
            if (count == 0) return;

            CUresult res = cuMemAlloc(&ptr_, count * sizeof(T));
            if (res != CUDA_SUCCESS) {
                UMGEBUNG_LOG_ERROR("CUDA: Failed to allocate device memory (count: {})", count);
                ptr_ = 0;
                count_ = 0;
                return;
            }
            count_ = count;
        }

        void free() {
            if (ptr_) {
                cuMemFree(ptr_);
                ptr_ = 0;
                count_ = 0;
            }
        }

        void upload(const T* hostData, size_t count) {
            if (count > count_) allocate(count);
            if (count == 0 || !hostData) return;
            
            cuMemcpyHtoD(ptr_, hostData, count * sizeof(T));
        }

        void upload(const std::vector<T>& hostData) {
            upload(hostData.data(), hostData.size());
        }

        void download(T* hostData, size_t count) const {
            if (!ptr_ || count == 0 || !hostData) return;
            size_t downloadCount = (count < count_) ? count : count_;
            cuMemcpyDtoH(hostData, ptr_, downloadCount * sizeof(T));
        }

        void download(std::vector<T>& hostData) const {
            if (hostData.size() < count_) hostData.resize(count_);
            download(hostData.data(), count_);
        }

        CUdeviceptr get() const { return ptr_; }
        size_t count() const { return count_; }
        size_t sizeInBytes() const { return count_ * sizeof(T); }

        operator CUdeviceptr() const { return ptr_; }

    private:
        CUdeviceptr ptr_;
        size_t count_;
    };

} // namespace Umgebung::util
