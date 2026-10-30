#include "umgebung/core/SystemProfiler.hpp"
#include <thread>

namespace umgebung::core {

    HardwareCapabilities SystemProfiler::ProfileSystem() {
        HardwareCapabilities caps{};

        // Query CPU Cores
        caps.cpuCoreCount = std::thread::hardware_concurrency();

        // Query CUDA Devices
        cudaError_t cudaStatus = cudaGetDeviceCount(&caps.cudaDeviceCount);
        if (cudaStatus != cudaSuccess || caps.cudaDeviceCount == 0) {
            spdlog::warn("No CUDA-compatible GPUs detected or CUDA driver error: {}", cudaGetErrorString(cudaStatus));
            return caps;
        }

        // Default to device 0 for initial capability inspection
        caps.selectedCUDADevice = 0;
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, caps.selectedCUDADevice);

        caps.gpuName = prop.name;
        caps.computeCapabilityMajor = prop.major;
        caps.computeCapabilityMinor = prop.minor;
        caps.gpuUUID = prop.uuid;

        // Query current VRAM stats
        cudaMemGetInfo(&caps.freeVRAM, &caps.totalVRAM);

        spdlog::info("System Profiling Complete:");
        spdlog::info("  CPU Logical Cores: {}", caps.cpuCoreCount);
        spdlog::info("  GPU: {}", caps.gpuName);
        spdlog::info("  Compute Capability: {}.{}", caps.computeCapabilityMajor, caps.computeCapabilityMinor);
        spdlog::info("  VRAM: {:.2f} GB free / {:.2f} GB total",
                     static_cast<double>(caps.freeVRAM) / (1024.0 * 1024.0 * 1024.0),
                     static_cast<double>(caps.totalVRAM) / (1024.0 * 1024.0 * 1024.0));

        return caps;
    }

} // namespace umgebung::core