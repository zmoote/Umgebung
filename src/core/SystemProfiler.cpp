/**
 * @file SystemProfiler.cpp
 * @brief Implementation of cross-platform host CPU/RAM discovery, NVIDIA CUDA
 *        GPU inventory, and adaptive runtime budgeting for Umgebung.
 *
 * @details
 * Implements `umgebung::core::SystemProfiler` using:
 *   - **Linux:** `/proc/cpuinfo`, `/proc/meminfo`, and `<sys/sysinfo.h>` fallback
 *     for accurate physical/available RAM (accounting for OS page cache).
 *   - **Windows:** Win32 `GlobalMemoryStatusEx` and CPUID brand string probing.
 *   - **CUDA:** `cudaDriverGetVersion`, `cudaRuntimeGetVersion`,
 *     `cudaGetDeviceCount`, `cudaGetDeviceProperties`, and per-device
 *     `cudaMemGetInfo` (saving and restoring the active CUDA device ordinal).
 *
 * All byte-to-gibibyte conversions use IEC binary GiB (`1024^3 = 1,073,741,824`
 * bytes) so VRAM reports match `nvidia-smi` and NVIDIA hardware specifications.
 */

#include "umgebung/core/SystemProfiler.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#include <spdlog/spdlog.h>

#if defined(UMGEBUNG_PLATFORM_LINUX)
    #include <sys/sysinfo.h>
#elif defined(UMGEBUNG_PLATFORM_WINDOWS)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <intrin.h>
    #include <windows.h>
#endif

namespace umgebung::core {

    namespace {

        /** @brief Conversion factor from bytes to binary gibibytes (`1 GiB = 2^30 bytes`). */
        constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

        /** @brief Conversion factor from bytes to binary mebibytes (`1 MiB = 2^20 bytes`). */
        constexpr std::size_t kBytesPerMiB = 1024ULL * 1024ULL;

        /**
         * @brief Conservative Structure-of-Arrays (SoA) VRAM footprint per
         *        double-precision (`FP64`) N-body entity in bytes.
         *
         * @details
         * Accounts for:
         *   - Position `(x, y, z)`: 3 * 8 B = 24 B
         *   - Velocity `(vx, vy, vz)`: 3 * 8 B = 24 B
         *   - Acceleration `(ax, ay, az)`: 3 * 8 B = 24 B
         *   - Mass + softening/metadata `(m, flags)`: 16 B
         *   - Double-buffering / scratch headroom rounded up to 128 B/entity.
         */
        constexpr std::uint64_t kEstimatedBytesPerFP64Body = 128ULL;

        /**
         * @brief Trims leading and trailing ASCII whitespace from a string.
         *
         * @param value Input string to sanitize.
         * @return Trimmed copy of `value`.
         */
        [[nodiscard]] std::string TrimWhitespace(const std::string& value) {
            const std::size_t start = value.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) {
                return {};
            }
            const std::size_t end = value.find_last_not_of(" \t\r\n");
            return value.substr(start, end - start + 1);
        }

        /**
         * @brief Queries the host operating system for the CPU brand/model string.
         *
         * @return Human-readable CPU model name, or `"Unknown CPU"` if unavailable.
         */
        [[nodiscard]] std::string QueryCpuModelName() {
#if defined(UMGEBUNG_PLATFORM_LINUX)
            // Parse the first `model name : ...` entry from Linux's `/proc/cpuinfo`.
            std::ifstream cpuinfo("/proc/cpuinfo");
            std::string line;
            while (std::getline(cpuinfo, line)) {
                if (line.rfind("model name", 0) == 0) {
                    const std::size_t colonPos = line.find(':');
                    if (colonPos != std::string::npos) {
                        std::string model = TrimWhitespace(line.substr(colonPos + 1));
                        if (!model.empty()) {
                            return model;
                        }
                    }
                }
            }
            return "Unknown Linux CPU";
#elif defined(UMGEBUNG_PLATFORM_WINDOWS)
            // Query extended CPUID leaves 0x80000002..0x80000004 for the 48-byte brand string.
            int cpuInfo[4] = {0, 0, 0, 0};
            __cpuid(cpuInfo, static_cast<int>(0x80000000));
            const unsigned int maxExtendedLeaf = static_cast<unsigned int>(cpuInfo[0]);
            if (maxExtendedLeaf >= 0x80000004U) {
                char brand[49] = {};
                __cpuid(reinterpret_cast<int*>(brand + 0), static_cast<int>(0x80000002));
                __cpuid(reinterpret_cast<int*>(brand + 16), static_cast<int>(0x80000003));
                __cpuid(reinterpret_cast<int*>(brand + 32), static_cast<int>(0x80000004));
                std::string model = TrimWhitespace(std::string(brand));
                if (!model.empty()) {
                    return model;
                }
            }
            return "Unknown Windows CPU";
#else
            return "Unknown CPU";
#endif
        }

        /**
         * @brief Queries total physical host RAM and currently available RAM in bytes.
         *
         * @param[out] outTotalBytes     Populated with installed physical RAM in bytes.
         * @param[out] outAvailableBytes Populated with immediately allocatable RAM in bytes.
         */
        void QuerySystemMemory(std::uint64_t& outTotalBytes, std::uint64_t& outAvailableBytes) {
            outTotalBytes = 0;
            outAvailableBytes = 0;

#if defined(UMGEBUNG_PLATFORM_LINUX)
            // Prefer `/proc/meminfo` because `MemAvailable` accurately includes
            // reclaimable kernel buffers and page cache, whereas `sysinfo().freeram`
            // drastically under-reports usable RAM on active Linux workstations.
            std::ifstream meminfo("/proc/meminfo");
            std::string key;
            std::uint64_t valueKiB = 0;
            std::string unit;

            while (meminfo >> key >> valueKiB) {
                // Consume the remainder of the line (typically `"kB"`).
                std::getline(meminfo, unit);
                if (key == "MemTotal:") {
                    outTotalBytes = valueKiB * 1024ULL;
                } else if (key == "MemAvailable:") {
                    outAvailableBytes = valueKiB * 1024ULL;
                }
                if (outTotalBytes > 0 && outAvailableBytes > 0) {
                    break;
                }
            }

            // Fall back to `sysinfo(2)` if `/proc/meminfo` was inaccessible or incomplete.
            if (outTotalBytes == 0) {
                struct sysinfo info {};
                if (sysinfo(&info) == 0) {
                    const std::uint64_t memUnit = static_cast<std::uint64_t>(info.mem_unit);
                    outTotalBytes = static_cast<std::uint64_t>(info.totalram) * memUnit;
                    if (outAvailableBytes == 0) {
                        outAvailableBytes = static_cast<std::uint64_t>(info.freeram) * memUnit;
                    }
                }
            }
#elif defined(UMGEBUNG_PLATFORM_WINDOWS)
            MEMORYSTATUSEX memStatus{};
            memStatus.dwLength = sizeof(MEMORYSTATUSEX);
            if (GlobalMemoryStatusEx(&memStatus) != 0) {
                outTotalBytes = static_cast<std::uint64_t>(memStatus.ullTotalPhys);
                outAvailableBytes = static_cast<std::uint64_t>(memStatus.ullAvailPhys);
            }
#endif
        }

        /**
         * @brief Determines whether a given NVIDIA Compute Capability belongs to
         *        a datacenter SKU class with native 1:2 FP64:FP32 throughput.
         *
         * @details
         * On NVIDIA architectures:
         *   - `sm_70` (Volta V100), `sm_80` (Ampere A100), `sm_90` (Hopper H100),
         *     and `sm_100` (Blackwell B100/B200) provide full-rate datacenter FP64
         *     pipelines (1:2 FP64:FP32 ratio).
         *   - Consumer/workstation architectures such as `sm_75` (Turing TITAN RTX),
         *     `sm_86` (GA102), and `sm_89` (Ada Lovelace RTX 4070) execute FP64 at
         *     reduced throughput (typically 1:32 or 1:64), though FP64 remains fully
         *     supported and essential for symplectic energy conservation tests.
         *
         * @param major Compute capability major revision.
         * @param minor Compute capability minor revision.
         * @return True for high-throughput datacenter FP64 compute capabilities.
         */
        [[nodiscard]] bool IsHighThroughputFP64Arch(int major, int minor) noexcept {
            return (major == 7 && minor == 0) ||  // Volta (V100)
                   (major == 8 && minor == 0) ||  // Ampere Datacenter (A100)
                   (major == 9 && minor == 0) ||  // Hopper (H100)
                   (major >= 10 && minor == 0);   // Blackwell Datacenter (B100/B200)
        }

        /**
         * @brief Computes a heuristic suitability score for ranking multiple GPUs
         *        when no explicit `--device` index was requested by the user.
         *
         * @details
         * Prioritizes:
         *   1. Available VRAM capacity (primary bottleneck for large multi-scale scenes),
         *   2. Streaming Multiprocessor (SM) compute width,
         *   3. Bonus weight for datacenter high-throughput FP64 silicon (A100/V100).
         *
         * @param gpu Candidate GPU descriptor.
         * @return Monotonic score where higher is better.
         */
        [[nodiscard]] double ScoreGpuForSelection(const GPUDeviceInfo& gpu) noexcept {
            const double totalVramGiB = static_cast<double>(gpu.totalVRAMBytes) / kBytesPerGiB;
            const double freeVramGiB = static_cast<double>(gpu.freeVRAMBytes) / kBytesPerGiB;
            const double smWeight = static_cast<double>(gpu.multiProcessorCount) * 0.5;
            const double fp64Bonus = gpu.hasHighThroughputFP64 ? 50.0 : 0.0;
            return (totalVramGiB * 2.0) + (freeVramGiB * 1.5) + smWeight + fp64Bonus;
        }

    } // anonymous namespace

    std::string SystemProfiler::InferArchitectureName(int major, int minor) {
        // Map NVIDIA SM revisions (including every GPU in GEMINI.md's Hardware
        // Topology Matrix) to their corresponding microarchitecture names.
        if (major == 7 && minor == 0) {
            return "Volta";
        }
        if (major == 7 && minor == 5) {
            return "Turing";
        }
        if (major == 8 && minor == 0) {
            return "Ampere (Datacenter)";
        }
        if (major == 8 && (minor == 6 || minor == 7)) {
            return "Ampere";
        }
        if (major == 8 && minor == 9) {
            return "Ada Lovelace";
        }
        if (major == 9 && minor == 0) {
            return "Hopper";
        }
        if (major >= 10) {
            return "Blackwell+";
        }
        return "Legacy / Unknown NVIDIA Architecture";
    }

    const char* SystemProfiler::HardwareTierToString(HardwareTier tier) noexcept {
        switch (tier) {
            case HardwareTier::Unsupported:
                return "Unsupported";
            case HardwareTier::LaptopDiscrete:
                return "LaptopDiscrete";
            case HardwareTier::DesktopWorkstation:
                return "DesktopWorkstation";
            case HardwareTier::SingleGpuHpcNode:
                return "SingleGpuHpcNode";
            case HardwareTier::MultiGpuHpcNode:
                return "MultiGpuHpcNode";
        }
        return "Unknown";
    }

    std::string SystemProfiler::FormatCudaVersion(int encodedVersion) {
        // NVIDIA encodes CUDA versions as `1000 * major + 10 * minor` (e.g., 13040 -> "13.4").
        if (encodedVersion <= 0) {
            return "Unavailable";
        }
        const int major = encodedVersion / 1000;
        const int minor = (encodedVersion % 1000) / 10;
        return std::to_string(major) + "." + std::to_string(minor);
    }

    std::string SystemProfiler::FormatUuid(const std::array<std::uint8_t, 16>& uuid) {
        // Format 16 raw bytes as `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` (36 chars + NUL).
        char buffer[37] = {};
        std::snprintf(
            buffer,
            sizeof(buffer),
            "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            uuid[0], uuid[1], uuid[2], uuid[3],
            uuid[4], uuid[5],
            uuid[6], uuid[7],
            uuid[8], uuid[9],
            uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]
        );
        return std::string(buffer);
    }

    HardwareCapabilities SystemProfiler::ProfileSystem() {
        HardwareCapabilities caps{};

        // ---------------------------------------------------------------------
        // Step 1: Identify Host Operating System, CPU Topology, and Host RAM
        // ---------------------------------------------------------------------
#if defined(UMGEBUNG_PLATFORM_LINUX)
        caps.osName = "Linux";
#elif defined(UMGEBUNG_PLATFORM_WINDOWS)
        caps.osName = "Windows";
#else
        caps.osName = "Unknown";
#endif

        caps.cpuModelName = QueryCpuModelName();

        // `std::thread::hardware_concurrency()` returns 0 only if the platform
        // cannot determine thread count; normalize that edge case to 1.
        const std::uint32_t detectedThreads = std::thread::hardware_concurrency();
        caps.cpuLogicalCores = (detectedThreads > 0U) ? detectedThreads : 1U;

        QuerySystemMemory(caps.totalSystemRAMBytes, caps.availableSystemRAMBytes);

        // ---------------------------------------------------------------------
        // Step 2: Query CUDA Driver & Runtime Versions
        // ---------------------------------------------------------------------
        (void)cudaDriverGetVersion(&caps.cudaDriverVersion);
        (void)cudaRuntimeGetVersion(&caps.cudaRuntimeVersion);

        // ---------------------------------------------------------------------
        // Step 3: Enumerate All Visible NVIDIA CUDA Devices
        // ---------------------------------------------------------------------
        int deviceCount = 0;
        const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);
        if (countStatus != cudaSuccess || deviceCount <= 0) {
            caps.cudaDeviceCount = 0;
            if (countStatus != cudaSuccess) {
                caps.cudaErrorMessage = cudaGetErrorString(countStatus);
            } else {
                caps.cudaErrorMessage = "No CUDA-capable NVIDIA GPUs detected on this system.";
            }
            return caps;
        }

        caps.cudaDeviceCount = deviceCount;
        caps.gpus.reserve(static_cast<std::size_t>(deviceCount));

        // Record the previously active CUDA device so profiling leaves global
        // CUDA thread state clean for any subsequent callers.
        int previousDevice = 0;
        const bool hadPreviousDevice = (cudaGetDevice(&previousDevice) == cudaSuccess);

        for (int devIdx = 0; devIdx < deviceCount; ++devIdx) {
            cudaDeviceProp prop{};
            const cudaError_t propStatus = cudaGetDeviceProperties(&prop, devIdx);
            if (propStatus != cudaSuccess) {
                spdlog::warn(
                    "SystemProfiler: Failed to query properties for CUDA device {}: {}",
                    devIdx,
                    cudaGetErrorString(propStatus)
                );
                continue;
            }

            GPUDeviceInfo info{};
            info.deviceIndex = devIdx;
            info.name = prop.name;
            info.computeCapabilityMajor = prop.major;
            info.computeCapabilityMinor = prop.minor;
            info.architectureName = InferArchitectureName(prop.major, prop.minor);
            info.totalVRAMBytes = prop.totalGlobalMem;
            info.freeVRAMBytes = prop.totalGlobalMem; // Overwritten below by live `cudaMemGetInfo` if available.
            info.multiProcessorCount = prop.multiProcessorCount;
            info.warpSize = (prop.warpSize > 0) ? prop.warpSize : 32;
            info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
            info.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
            info.sharedMemPerBlockBytes = prop.sharedMemPerBlock;
            info.l2CacheSizeBytes = prop.l2CacheSize;
            info.supportsManagedMemory = (prop.managedMemory != 0);
            info.supportsConcurrentManagedAccess = (prop.concurrentManagedAccess != 0);
            info.hasHighThroughputFP64 = IsHighThroughputFP64Arch(prop.major, prop.minor);
            info.pciBusId = prop.pciBusID;
            info.pciDeviceId = prop.pciDeviceID;

            // Copy the 16-byte GPU UUID verbatim for future Vulkan physical device matching.
            static_assert(sizeof(prop.uuid.bytes) == 16, "Expected 16-byte CUDA GPU UUID.");
            std::memcpy(info.uuid.data(), prop.uuid.bytes, info.uuid.size());

            // Select this GPU momentarily to inspect its live free/total VRAM.
            if (cudaSetDevice(devIdx) == cudaSuccess) {
                std::size_t freeMem = 0;
                std::size_t totalMem = 0;
                if (cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess) {
                    info.freeVRAMBytes = freeMem;
                    info.totalVRAMBytes = totalMem;
                }
            }

            caps.gpus.push_back(std::move(info));
        }

        // Restore the prior CUDA device binding if one was active before profiling.
        if (hadPreviousDevice && previousDevice >= 0 && previousDevice < deviceCount) {
            (void)cudaSetDevice(previousDevice);
        }

        return caps;
    }

    HardwareAdaptationPlan SystemProfiler::BuildAdaptationPlan(
        const HardwareCapabilities& caps,
        const HardwareAdaptationConfig& config
    ) {
        HardwareAdaptationPlan plan{};
        plan.executionModeLabel = config.guiActive
            ? "Desktop Interactive (GUI)"
            : "Headless Compute (HPC/Batch)";

        // ---------------------------------------------------------------------
        // Step 1: Host CPU Worker Thread & Host RAM Budgets
        // ---------------------------------------------------------------------
        // In interactive GUI mode, reserve 2 logical threads (where available)
        // for the window/UI event loop and Vulkan submission so simulation I/O
        // or host preprocessing never causes frame hitches. In headless HPC
        // mode, reserve at most 1 thread for orchestration/telemetry.
        const std::uint32_t reservedHostThreads = config.guiActive ? 2U : 1U;
        if (caps.cpuLogicalCores > reservedHostThreads) {
            plan.recommendedHostWorkerThreads = caps.cpuLogicalCores - reservedHostThreads;
        } else {
            plan.recommendedHostWorkerThreads = 1U;
        }

        // Cap host staging/checkpoint RAM usage at 75% of currently available RAM
        // (or 50% of total RAM if available RAM could not be probed) to avoid
        // triggering the Linux OOM killer or swapping on desktop/HPC nodes.
        if (caps.availableSystemRAMBytes > 0) {
            plan.hostRAMBudgetBytes = (caps.availableSystemRAMBytes * 3ULL) / 4ULL;
        } else {
            plan.hostRAMBudgetBytes = caps.totalSystemRAMBytes / 2ULL;
        }

        // ---------------------------------------------------------------------
        // Step 2: Validate GPU Availability
        // ---------------------------------------------------------------------
        if (caps.gpus.empty()) {
            plan.computeReady = false;
            plan.tier = HardwareTier::Unsupported;
            plan.adaptationNotes.emplace_back(
                "No CUDA-capable NVIDIA GPUs detected; CUDA simulation execution is disabled."
            );
            return plan;
        }

        // ---------------------------------------------------------------------
        // Step 3: Select Primary GPU & Populate Multi-GPU Active List
        // ---------------------------------------------------------------------
        const GPUDeviceInfo* selectedGpu = nullptr;

        if (config.preferredDeviceIndex.has_value()) {
            const int requested = config.preferredDeviceIndex.value();
            for (const auto& gpu : caps.gpus) {
                if (gpu.deviceIndex == requested) {
                    selectedGpu = &gpu;
                    plan.adaptationNotes.push_back(
                        "Primary GPU explicitly pinned to CUDA device " + std::to_string(requested) +
                        " (" + gpu.name + ") via runtime configuration."
                    );
                    break;
                }
            }
            if (selectedGpu == nullptr) {
                plan.adaptationNotes.push_back(
                    "Requested CUDA device " + std::to_string(requested) +
                    " was not found; falling back to automatic highest-capability GPU selection."
                );
            }
        }

        if (selectedGpu == nullptr) {
            double bestScore = -1.0;
            for (const auto& gpu : caps.gpus) {
                const double score = ScoreGpuForSelection(gpu);
                if (score > bestScore) {
                    bestScore = score;
                    selectedGpu = &gpu;
                }
            }
        }

        plan.computeReady = true;
        plan.primaryGpuIndex = selectedGpu->deviceIndex;

        // Place the primary GPU first in `activeGpuIndices`, followed by any
        // additional GPUs for future multi-GPU domain decomposition on HPC nodes.
        plan.activeGpuIndices.push_back(selectedGpu->deviceIndex);
        for (const auto& gpu : caps.gpus) {
            if (gpu.deviceIndex != selectedGpu->deviceIndex) {
                plan.activeGpuIndices.push_back(gpu.deviceIndex);
            }
        }
        plan.enableMultiGpuDispatch = (plan.activeGpuIndices.size() > 1U);

        // ---------------------------------------------------------------------
        // Step 4: Classify Overall Hardware Tier
        // ---------------------------------------------------------------------
        const double primaryTotalGiB = static_cast<double>(selectedGpu->totalVRAMBytes) / kBytesPerGiB;
        if (caps.gpus.size() > 1U) {
            plan.tier = HardwareTier::MultiGpuHpcNode;
        } else if (primaryTotalGiB >= 30.0) {
            // Covers V100 32 GB, A100 40/80 GB, H100 80 GB, etc.
            plan.tier = HardwareTier::SingleGpuHpcNode;
        } else if (primaryTotalGiB > 12.0) {
            // Covers TITAN RTX 24 GB, RTX 3090/4090 24 GB, workstation RTX cards.
            plan.tier = HardwareTier::DesktopWorkstation;
        } else {
            // Covers RTX 4070 Laptop 8 GB and compact discrete mobile/desktop GPUs.
            plan.tier = HardwareTier::LaptopDiscrete;
        }

        // ---------------------------------------------------------------------
        // Step 5: Compute Safe VRAM Budget & Compositor/Driver Headroom
        // ---------------------------------------------------------------------
        // GUI mode shares the GPU with the OS desktop compositor, Vulkan
        // swapchain images, depth buffers, and Dear ImGui font/UI atlases.
        // Reserve max(15% of total VRAM, 1024 MiB) in GUI mode, or
        // max(8% of total VRAM, 512 MiB) in headless HPC batch mode.
        const std::size_t minHeadroomBytes = config.guiActive
            ? (1024ULL * kBytesPerMiB)
            : (512ULL * kBytesPerMiB);
        const std::size_t fractionalHeadroomBytes = config.guiActive
            ? (selectedGpu->totalVRAMBytes * 15ULL) / 100ULL
            : (selectedGpu->totalVRAMBytes * 8ULL) / 100ULL;

        plan.reservedVRAMHeadroomBytes = std::max(minHeadroomBytes, fractionalHeadroomBytes);

        if (selectedGpu->freeVRAMBytes > plan.reservedVRAMHeadroomBytes) {
            plan.simulationVRAMBudgetBytes = selectedGpu->freeVRAMBytes - plan.reservedVRAMHeadroomBytes;
        } else {
            // Even on a heavily loaded desktop, leave a modest safe slice (50% of
            // currently free VRAM) rather than reporting zero when free VRAM is tight.
            plan.simulationVRAMBudgetBytes = selectedGpu->freeVRAMBytes / 2ULL;
            plan.adaptationNotes.emplace_back(
                "Free VRAM on primary GPU is below nominal headroom target; "
                "throttling simulation VRAM budget to 50% of currently free VRAM."
            );
        }

        plan.estimatedMaxFP64Bodies =
            static_cast<std::uint64_t>(plan.simulationVRAMBudgetBytes) / kEstimatedBytesPerFP64Body;

        // ---------------------------------------------------------------------
        // Step 6: Tune CUDA Kernel Launch Geometry & Concurrency Parameters
        // ---------------------------------------------------------------------
        // Default to 256 threads/block (8 warps): achieves high occupancy across
        // Volta, Turing, Ampere, and Ada Lovelace while leaving ample register
        // budget per thread for FP64 N-body and field-equation kernels.
        plan.recommendedCudaBlockSize = std::min(256, selectedGpu->maxThreadsPerBlock);
        if (plan.recommendedCudaBlockSize < selectedGpu->warpSize) {
            plan.recommendedCudaBlockSize = selectedGpu->warpSize;
        }

        // Scale concurrent CUDA streams with SM width (2 streams for compact
        // laptop GPUs, 4 streams for workstation/datacenter GPUs with >= 48 SMs).
        plan.recommendedCudaStreams = (selectedGpu->multiProcessorCount >= 48) ? 4 : 2;

        // Physics verification always defaults to IEEE-754 double precision (FP64)
        // so symplectic integrators maintain conservation laws across long orbits.
        plan.useDoublePrecisionPhysics = true;
        if (!selectedGpu->hasHighThroughputFP64) {
            plan.adaptationNotes.push_back(
                "Primary GPU (" + selectedGpu->architectureName + ", SM " +
                std::to_string(selectedGpu->computeCapabilityMajor) + "." +
                std::to_string(selectedGpu->computeCapabilityMinor) +
                ") uses consumer/workstation FP64 pipelines; FP64 remains enabled for "
                "physical conservation rigor (expect higher throughput on HPC V100/A100 nodes)."
            );
        } else {
            plan.adaptationNotes.push_back(
                "Primary GPU provides native datacenter FP64 throughput (" +
                selectedGpu->architectureName + ")."
            );
        }

        if (plan.enableMultiGpuDispatch) {
            plan.adaptationNotes.push_back(
                "Multi-GPU topology detected (" + std::to_string(plan.activeGpuIndices.size()) +
                " CUDA devices); multi-GPU dispatch capability flagged as available."
            );
        }

        return plan;
    }

    void SystemProfiler::LogSummary(
        const HardwareCapabilities& caps,
        const HardwareAdaptationPlan& plan
    ) {
        spdlog::info("=================================================================");
        spdlog::info("  Umgebung Startup Hardware Profile & Adaptation Summary");
        spdlog::info("=================================================================");
        spdlog::info("  Execution Mode : {}", plan.executionModeLabel);
        spdlog::info("  Hardware Tier  : {}", HardwareTierToString(plan.tier));
        spdlog::info("  Host OS        : {}", caps.osName);
        spdlog::info("  CPU            : {} ({} logical threads)", caps.cpuModelName, caps.cpuLogicalCores);
        spdlog::info(
            "  Host RAM       : {:.2f} GiB available / {:.2f} GiB total",
            static_cast<double>(caps.availableSystemRAMBytes) / kBytesPerGiB,
            static_cast<double>(caps.totalSystemRAMBytes) / kBytesPerGiB
        );
        spdlog::info(
            "  CUDA Versions  : Driver {}, Runtime {}",
            FormatCudaVersion(caps.cudaDriverVersion),
            FormatCudaVersion(caps.cudaRuntimeVersion)
        );

        if (caps.gpus.empty()) {
            spdlog::warn("  CUDA GPUs      : None detected ({})", caps.cudaErrorMessage);
        } else {
            spdlog::info("  CUDA GPUs ({}):", caps.gpus.size());
            for (const auto& gpu : caps.gpus) {
                const bool isPrimary = (gpu.deviceIndex == plan.primaryGpuIndex);
                spdlog::info(
                    "    [{}] {}{} | {} (SM {}.{}) | {} SMs | {:.2f}/{:.2f} GiB VRAM free | UUID {}",
                    gpu.deviceIndex,
                    gpu.name,
                    isPrimary ? " [PRIMARY]" : "",
                    gpu.architectureName,
                    gpu.computeCapabilityMajor,
                    gpu.computeCapabilityMinor,
                    gpu.multiProcessorCount,
                    static_cast<double>(gpu.freeVRAMBytes) / kBytesPerGiB,
                    static_cast<double>(gpu.totalVRAMBytes) / kBytesPerGiB,
                    FormatUuid(gpu.uuid)
                );
            }
        }

        spdlog::info("-----------------------------------------------------------------");
        spdlog::info("  Adapted Runtime Budgets:");
        spdlog::info("    Compute Ready        : {}", plan.computeReady ? "YES" : "NO");
        spdlog::info("    Host Worker Threads  : {}", plan.recommendedHostWorkerThreads);
        spdlog::info(
            "    Host RAM Budget      : {:.2f} GiB",
            static_cast<double>(plan.hostRAMBudgetBytes) / kBytesPerGiB
        );
        spdlog::info(
            "    Sim VRAM Budget      : {:.2f} GiB (headroom reserved: {:.2f} GiB)",
            static_cast<double>(plan.simulationVRAMBudgetBytes) / kBytesPerGiB,
            static_cast<double>(plan.reservedVRAMHeadroomBytes) / kBytesPerGiB
        );
        spdlog::info(
            "    CUDA Launch Defaults : blockSize={}, streams={}, FP64={}, multiGPU={}",
            plan.recommendedCudaBlockSize,
            plan.recommendedCudaStreams,
            plan.useDoublePrecisionPhysics ? "ON" : "OFF",
            plan.enableMultiGpuDispatch ? "ENABLED" : "SINGLE-GPU"
        );
        spdlog::info("    Est. Max FP64 Bodies : ~{}", plan.estimatedMaxFP64Bodies);

        for (const auto& note : plan.adaptationNotes) {
            spdlog::info("    Note: {}", note);
        }
        spdlog::info("=================================================================");
    }

    nlohmann::json SystemProfiler::ToJson(
        const HardwareCapabilities& caps,
        const HardwareAdaptationPlan& plan
    ) {
        nlohmann::json gpuArray = nlohmann::json::array();
        for (const auto& gpu : caps.gpus) {
            gpuArray.push_back({
                {"device_index", gpu.deviceIndex},
                {"name", gpu.name},
                {"architecture", gpu.architectureName},
                {"compute_capability_major", gpu.computeCapabilityMajor},
                {"compute_capability_minor", gpu.computeCapabilityMinor},
                {"total_vram_bytes", gpu.totalVRAMBytes},
                {"free_vram_bytes", gpu.freeVRAMBytes},
                {"multiprocessor_count", gpu.multiProcessorCount},
                {"warp_size", gpu.warpSize},
                {"max_threads_per_block", gpu.maxThreadsPerBlock},
                {"max_threads_per_multiprocessor", gpu.maxThreadsPerMultiProcessor},
                {"shared_mem_per_block_bytes", gpu.sharedMemPerBlockBytes},
                {"l2_cache_size_bytes", gpu.l2CacheSizeBytes},
                {"supports_managed_memory", gpu.supportsManagedMemory},
                {"supports_concurrent_managed_access", gpu.supportsConcurrentManagedAccess},
                {"has_high_throughput_fp64", gpu.hasHighThroughputFP64},
                {"pci_bus_id", gpu.pciBusId},
                {"pci_device_id", gpu.pciDeviceId},
                {"uuid", FormatUuid(gpu.uuid)}
            });
        }

        return nlohmann::json{
            {"hardware", {
                {"os_name", caps.osName},
                {"cpu_model_name", caps.cpuModelName},
                {"cpu_logical_cores", caps.cpuLogicalCores},
                {"total_system_ram_bytes", caps.totalSystemRAMBytes},
                {"available_system_ram_bytes", caps.availableSystemRAMBytes},
                {"cuda_driver_version", FormatCudaVersion(caps.cudaDriverVersion)},
                {"cuda_runtime_version", FormatCudaVersion(caps.cudaRuntimeVersion)},
                {"cuda_device_count", caps.cudaDeviceCount},
                {"cuda_error_message", caps.cudaErrorMessage},
                {"gpus", std::move(gpuArray)}
            }},
            {"adaptation", {
                {"compute_ready", plan.computeReady},
                {"hardware_tier", HardwareTierToString(plan.tier)},
                {"execution_mode", plan.executionModeLabel},
                {"primary_gpu_index", plan.primaryGpuIndex},
                {"active_gpu_indices", plan.activeGpuIndices},
                {"recommended_host_worker_threads", plan.recommendedHostWorkerThreads},
                {"simulation_vram_budget_bytes", plan.simulationVRAMBudgetBytes},
                {"reserved_vram_headroom_bytes", plan.reservedVRAMHeadroomBytes},
                {"host_ram_budget_bytes", plan.hostRAMBudgetBytes},
                {"recommended_cuda_block_size", plan.recommendedCudaBlockSize},
                {"recommended_cuda_streams", plan.recommendedCudaStreams},
                {"use_double_precision_physics", plan.useDoublePrecisionPhysics},
                {"enable_multi_gpu_dispatch", plan.enableMultiGpuDispatch},
                {"estimated_max_fp64_bodies", plan.estimatedMaxFP64Bodies},
                {"notes", plan.adaptationNotes}
            }}
        };
    }

} // namespace umgebung::core
