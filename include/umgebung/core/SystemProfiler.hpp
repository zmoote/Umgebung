#pragma once

#include <string>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>

namespace umgebung::core {

    struct HardwareCapabilities {
        // Host
        uint32_t cpuCoreCount{0};
        uint64_t totalSystemRAM{0};

        // CUDA
        int cudaDeviceCount{0};
        int selectedCUDADevice{-1};
        std::string gpuName;
        size_t totalVRAM{0};
        size_t freeVRAM{0};
        int computeCapabilityMajor{0};
        int computeCapabilityMinor{0};
        cudaUUID_t gpuUUID{};
    };

    class SystemProfiler {
    public:
        static HardwareCapabilities ProfileSystem();
    };

} // namespace umgebung::core