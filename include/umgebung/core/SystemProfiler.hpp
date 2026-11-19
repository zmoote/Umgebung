/**
 * @file SystemProfiler.hpp
 * @brief Cross-platform host CPU/RAM and NVIDIA CUDA GPU discovery and runtime
 *        hardware adaptation engine for Umgebung.
 *
 * @details
 * Umgebung targets a wide hardware spectrum spanning personal workstations
 * (AMD Ryzen 9 3900XT + TITAN RTX, Intel Core i9-13900H + RTX 4070 Laptop) and
 * multi-GPU university HPC clusters (UMaine ARCSIM, UNH Premise, RIT SPORC).
 *
 * Rather than hardcoding thread counts, VRAM allocations, or GPU launch
 * parameters, this module performs two strictly separated responsibilities at
 * application startup:
 *   1. **Hardware Discovery (`SystemProfiler::ProfileSystem`):**
 *      Queries operating system APIs and the NVIDIA CUDA Runtime/Driver to
 *      populate a complete, immutable snapshot (`HardwareCapabilities`) of host
 *      CPU topology, physical memory, and every visible CUDA-capable GPU.
 *   2. **Hardware Adaptation (`SystemProfiler::BuildAdaptationPlan`):**
 *      Translates the raw hardware snapshot and execution mode (interactive GUI
 *      vs. headless HPC batch) into concrete, safe operating budgets
 *      (`HardwareAdaptationPlan`), including primary/secondary GPU selection,
 *      VRAM allocation ceilings with compositor headroom, host worker thread
 *      budgets, and CUDA kernel launch geometry heuristics.
 *
 * @note
 * This header has zero dependency on windowing or graphics libraries (GLFW,
 * Vulkan, NVRHI, Dear ImGui) so it compiles identically in both Desktop GUI
 * (`UMGEBUNG_HAS_GUI == 1`) and Headless HPC (`UMGEBUNG_HAS_GUI == 0`) builds.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

/**
 * @namespace umgebung::core
 * @brief Core engine lifecycle, hardware profiling, CLI configuration, and
 *        platform-independent runtime utilities shared across all execution modes.
 */
namespace umgebung::core {

    /**
     * @enum HardwareTier
     * @brief Coarse classification of the detected host + GPU hardware topology.
     *
     * @details
     * Used by higher-level simulation subsystems (such as scene defaults,
     * solver batch sizing, and GUI viewport detail levels) to automatically
     * scale initial workloads without manual user tuning.
     */
    enum class HardwareTier : std::uint8_t {
        /** @brief No usable CUDA GPU detected; simulation compute cannot launch. */
        Unsupported = 0,
        /** @brief Single discrete GPU with <= 12 GiB VRAM (e.g., RTX 4070 Laptop 8 GiB). */
        LaptopDiscrete = 1,
        /** @brief Single high-VRAM workstation GPU > 12 GiB (e.g., TITAN RTX 24 GiB). */
        DesktopWorkstation = 2,
        /** @brief Single datacenter-class GPU (>= 32 GiB VRAM, e.g., V100/A100 node). */
        SingleGpuHpcNode = 3,
        /** @brief Multiple CUDA GPUs detected (e.g., DGX A100 or multi-GPU cluster node). */
        MultiGpuHpcNode = 4
    };

    /**
     * @struct GPUDeviceInfo
     * @brief Complete hardware capability descriptor for a single NVIDIA CUDA GPU.
     *
     * @details
     * Populated via `cudaGetDeviceProperties` and `cudaMemGetInfo` for each
     * CUDA device index `[0, cudaDeviceCount - 1]`. Retains the 16-byte hardware
     * UUID so Desktop GUI builds can match the exact same physical GPU between
     * CUDA compute contexts and Vulkan/NVRHI rendering devices for future
     * CUDA-Vulkan zero-copy interoperability.
     */
    struct GPUDeviceInfo {
        /** @brief Zero-based CUDA runtime device ordinal (`0 .. N-1`). */
        int deviceIndex{-1};

        /** @brief Human-readable NVIDIA product name (e.g., `"NVIDIA TITAN RTX"`). */
        std::string name;

        /** @brief Derived NVIDIA microarchitecture label (e.g., `"Volta"`, `"Turing"`, `"Ampere"`, `"Ada Lovelace"`). */
        std::string architectureName;

        /** @brief CUDA Compute Capability major revision (e.g., `7` for Turing, `8` for Ampere/Ada). */
        int computeCapabilityMajor{0};

        /** @brief CUDA Compute Capability minor revision (e.g., `5` for Turing, `9` for Ada). */
        int computeCapabilityMinor{0};

        /** @brief Total global device VRAM in bytes as reported by `cudaMemGetInfo` / `cudaDeviceProp`. */
        std::size_t totalVRAMBytes{0};

        /** @brief Currently unallocated (free) device VRAM in bytes at startup snapshot time. */
        std::size_t freeVRAMBytes{0};

        /** @brief Number of Streaming Multiprocessors (SMs) on the GPU. */
        int multiProcessorCount{0};

        /** @brief Hardware SIMD warp width in threads (32 on all modern NVIDIA architectures). */
        int warpSize{32};

        /** @brief Maximum threads permitted in a single CUDA thread block. */
        int maxThreadsPerBlock{0};

        /** @brief Maximum resident threads per Streaming Multiprocessor (occupancy ceiling). */
        int maxThreadsPerMultiProcessor{0};

        /** @brief Maximum shared memory available per thread block in bytes. */
        std::size_t sharedMemPerBlockBytes{0};

        /** @brief L2 cache capacity in bytes (impacts N-body / field tiling efficiency). */
        int l2CacheSizeBytes{0};

        /** @brief True if the GPU supports CUDA Unified / Managed Memory allocation. */
        bool supportsManagedMemory{false};

        /** @brief True if host and GPU can access Managed Memory concurrently without page thrashing. */
        bool supportsConcurrentManagedAccess{false};

        /** @brief True if the GPU has high-throughput datacenter FP64 hardware (e.g., Volta V100, Ampere A100, Hopper H100). */
        bool hasHighThroughputFP64{false};

        /** @brief PCI bus identifier used to disambiguate identical GPUs on multi-GPU nodes. */
        int pciBusId{0};

        /** @brief PCI device identifier on `pciBusId`. */
        int pciDeviceId{0};

        /** @brief Raw 16-byte GPU hardware UUID for exact CUDA <-> Vulkan physical device matching. */
        std::array<std::uint8_t, 16> uuid{};
    };

    /**
     * @struct HardwareCapabilities
     * @brief Immutable snapshot of host CPU, host RAM, CUDA toolchain, and all
     *        detected NVIDIA GPUs captured during engine startup.
     */
    struct HardwareCapabilities {
        // ---------------------------------------------------------------------
        // Host Platform & CPU Information
        // ---------------------------------------------------------------------

        /** @brief Operating system label (`"Linux"`, `"Windows"`, or `"Unknown"`). */
        std::string osName;

        /** @brief Human-readable CPU model string (e.g., `"AMD Ryzen 9 3900XT 12-Core Processor"`). */
        std::string cpuModelName;

        /** @brief Number of hardware logical CPU threads reported by the OS / C++ runtime. */
        std::uint32_t cpuLogicalCores{0};

        /** @brief Total physical host RAM installed in the system, in bytes. */
        std::uint64_t totalSystemRAMBytes{0};

        /** @brief Available (immediately allocatable without swapping) host RAM, in bytes. */
        std::uint64_t availableSystemRAMBytes{0};

        // ---------------------------------------------------------------------
        // CUDA Driver, Runtime & Device Inventory
        // ---------------------------------------------------------------------

        /** @brief Encoded CUDA Driver version (e.g., `13040` for CUDA 13.4; `0` if unavailable). */
        int cudaDriverVersion{0};

        /** @brief Encoded CUDA Runtime version compiled into this binary (e.g., `13040` for CUDA 13.4). */
        int cudaRuntimeVersion{0};

        /** @brief Total count of CUDA-compatible NVIDIA GPUs detected on this host. */
        int cudaDeviceCount{0};

        /** @brief Detailed capability records for every detected GPU (`size() == cudaDeviceCount`). */
        std::vector<GPUDeviceInfo> gpus;

        /** @brief Non-empty diagnostic message if CUDA enumeration failed or reported an error. */
        std::string cudaErrorMessage;
    };

    /**
     * @struct HardwareAdaptationConfig
     * @brief User/runtime inputs that influence how `SystemProfiler` adapts to
     *        the detected hardware.
     */
    struct HardwareAdaptationConfig {
        /**
         * @brief True when the application intends to run with an active desktop
         *        GUI/Vulkan compositor on the primary GPU; false for headless HPC
         *        or `--headless` execution.
         *
         * @details
         * When `guiActive` is true, the adaptation planner reserves additional
         * VRAM headroom for the Vulkan swapchain, Dear ImGui textures, and OS
         * display compositor, and reserves host CPU threads so UI responsiveness
         * does not starve under compute-heavy host tasks.
         */
        bool guiActive{false};

        /**
         * @brief Optional explicit CUDA device ordinal requested via CLI (`--device <id>`).
         *        When `std::nullopt`, the planner automatically selects the highest-scoring GPU.
         */
        std::optional<int> preferredDeviceIndex{std::nullopt};
    };

    /**
     * @struct HardwareAdaptationPlan
     * @brief Concrete, hardware-adapted execution policy computed at startup.
     *
     * @details
     * Every simulation subsystem (CUDA allocator, solver kernel launchers, host
     * thread pools, and `.umgebung` scene capacity guards) reads its budgets
     * from this plan so the same binary scales cleanly from an 8 GiB RTX 4070
     * Laptop to a 24 GiB TITAN RTX workstation to multi-A100 HPC nodes.
     */
    struct HardwareAdaptationPlan {
        /** @brief True if at least one usable CUDA GPU was selected and compute execution is viable. */
        bool computeReady{false};

        /** @brief Classified hardware tier for the host machine. */
        HardwareTier tier{HardwareTier::Unsupported};

        /** @brief Active execution mode label (`"Desktop Interactive (GUI)"` or `"Headless Compute (HPC/Batch)"`). */
        std::string executionModeLabel;

        /** @brief Selected primary CUDA device ordinal (`-1` if no GPU is available). */
        int primaryGpuIndex{-1};

        /** @brief Ordered list of all CUDA device ordinals eligible for multi-GPU dispatch. */
        std::vector<int> activeGpuIndices;

        /** @brief Recommended number of host worker threads for CPU-side scene/I/O/telemetry tasks. */
        std::uint32_t recommendedHostWorkerThreads{1};

        /** @brief Safe device VRAM budget (bytes) allocated to simulation data on the primary GPU. */
        std::size_t simulationVRAMBudgetBytes{0};

        /** @brief Reserved VRAM headroom (bytes) kept free for CUDA driver, Vulkan swapchain, and OS compositor. */
        std::size_t reservedVRAMHeadroomBytes{0};

        /** @brief Safe host RAM budget (bytes) for CPU staging buffers, checkpoints, and scene graphs. */
        std::uint64_t hostRAMBudgetBytes{0};

        /** @brief Recommended default 1D CUDA block size (threads per block) tuned to SM warp geometry. */
        int recommendedCudaBlockSize{256};

        /** @brief Recommended concurrent CUDA streams per GPU for overlapping H2D/kernel/D2H work. */
        int recommendedCudaStreams{2};

        /** @brief True when physics solvers should default to IEEE-754 double precision (`FP64`) for conservation accuracy. */
        bool useDoublePrecisionPhysics{true};

        /** @brief True when multi-GPU domain decomposition can be enabled across `activeGpuIndices`. */
        bool enableMultiGpuDispatch{false};

        /** @brief Estimated upper bound of double-precision N-body entities fitting in `simulationVRAMBudgetBytes` (assuming 64 B/entity SoA footprint). */
        std::uint64_t estimatedMaxFP64Bodies{0};

        /** @brief Human-readable notes explaining key adaptation choices and any warnings (e.g., consumer FP64 ratio, fallback GPU selection). */
        std::vector<std::string> adaptationNotes;
    };

    /**
     * @class SystemProfiler
     * @brief Stateless utility class providing startup hardware discovery,
     *        runtime adaptation planning, structured logging, and JSON export.
     */
    class SystemProfiler final {
    public:
        /**
         * @brief Probes the host OS and NVIDIA CUDA runtime to assemble a full
         *        `HardwareCapabilities` snapshot.
         *
         * @return Populated `HardwareCapabilities` structure. Never throws; if
         *         no CUDA GPU is found or the driver returns an error,
         *         `cudaDeviceCount` is `0` and `cudaErrorMessage` describes why.
         */
        [[nodiscard]] static HardwareCapabilities ProfileSystem();

        /**
         * @brief Derives an adaptive runtime execution plan from the hardware
         *        snapshot and requested execution configuration.
         *
         * @param caps   Hardware snapshot returned by `ProfileSystem()`.
         * @param config Execution context (GUI vs. headless and optional `--device` override).
         * @return Populated `HardwareAdaptationPlan` with safe CPU, RAM, VRAM,
         *         and CUDA launch geometry budgets.
         */
        [[nodiscard]] static HardwareAdaptationPlan BuildAdaptationPlan(
            const HardwareCapabilities& caps,
            const HardwareAdaptationConfig& config
        );

        /**
         * @brief Logs a formatted, human-readable hardware inventory and
         *        adaptation summary through `spdlog`.
         *
         * @param caps Hardware snapshot to report.
         * @param plan Computed adaptation plan to report.
         */
        static void LogSummary(
            const HardwareCapabilities& caps,
            const HardwareAdaptationPlan& plan
        );

        /**
         * @brief Serializes both the raw hardware snapshot and the computed
         *        adaptation plan into a structured JSON document.
         *
         * @details
         * Useful for HPC job provenance logs, reproducible benchmark metadata,
         * `--json-profile` CLI output, and automated CTest assertions.
         *
         * @param caps Hardware snapshot to serialize.
         * @param plan Adaptation plan to serialize.
         * @return `nlohmann::json` object containing `"hardware"` and `"adaptation"` sections.
         */
        [[nodiscard]] static nlohmann::json ToJson(
            const HardwareCapabilities& caps,
            const HardwareAdaptationPlan& plan
        );

        /**
         * @brief Maps an NVIDIA CUDA Compute Capability `(major, minor)` pair
         *        to its marketing/microarchitecture family name.
         *
         * @param major Compute capability major version (e.g., `7`, `8`, `9`, `10`).
         * @param minor Compute capability minor version (e.g., `0`, `5`, `6`, `9`).
         * @return Human-readable architecture name (e.g., `"Turing"`, `"Ampere"`, `"Ada Lovelace"`).
         */
        [[nodiscard]] static std::string InferArchitectureName(int major, int minor);

        /**
         * @brief Converts a `HardwareTier` enum value to a human-readable string.
         *
         * @param tier Hardware tier classification.
         * @return Null-terminated string literal naming the tier.
         */
        [[nodiscard]] static const char* HardwareTierToString(HardwareTier tier) noexcept;

        /**
         * @brief Formats a CUDA encoded version integer (e.g., `13040`) into a
         *        `"major.minor"` string (e.g., `"13.4"`).
         *
         * @param encodedVersion Integer returned by `cudaDriverGetVersion` or `cudaRuntimeGetVersion`.
         * @return Formatted version string, or `"Unavailable"` when `encodedVersion <= 0`.
         */
        [[nodiscard]] static std::string FormatCudaVersion(int encodedVersion);

        /**
         * @brief Formats a 16-byte GPU hardware UUID as a standard hyphenated
         *        RFC-4122-style hexadecimal string (`8-4-4-4-12`).
         *
         * @param uuid 16-byte raw GPU UUID array.
         * @return Lowercase hexadecimal UUID string.
         */
        [[nodiscard]] static std::string FormatUuid(const std::array<std::uint8_t, 16>& uuid);
    };

} // namespace umgebung::core
