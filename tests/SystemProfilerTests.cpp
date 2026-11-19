/**
 * @file SystemProfilerTests.cpp
 * @brief Automated CTest verification suite for startup hardware discovery and
 *        cross-topology runtime adaptation (`umgebung::core::SystemProfiler`).
 *
 * @details
 * Exercises two complementary test categories:
 *   1. **Live Host Probing:** Verifies that `SystemProfiler::ProfileSystem()`
 *      accurately detects CPU cores, host RAM, and NVIDIA CUDA GPUs on the
 *      executing workstation/cluster node, and produces a valid adaptation plan.
 *   2. **Synthetic Hardware Topology Matrix Tests:** Validates
 *      `SystemProfiler::BuildAdaptationPlan()` deterministically against every
 *      system in `GEMINI.md`'s Target Hardware Topology Matrix:
 *        - Personal Laptop (`RTX 4070 Laptop`, 8 GiB, SM 8.9, GUI mode)
 *        - Personal Desktop (`TITAN RTX`, 24 GiB, SM 7.5, GUI vs. Headless)
 *        - Single-GPU HPC Node (`NVIDIA V100` 32 GiB SM 7.0 / `A100` 80 GiB SM 8.0)
 *        - Multi-GPU HPC Node (4x `NVIDIA A100`, SM 8.0, explicit `--device` override)
 *        - Unsupported Host (0 CUDA GPUs)
 */

#include "umgebung/core/SystemProfiler.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

    /** @brief Constant `1 GiB` in bytes for constructing synthetic hardware test fixtures. */
    constexpr std::uint64_t kGiB = 1024ULL * 1024ULL * 1024ULL;

    /** @brief Tracks total failed assertions across all test cases in this executable. */
    int gFailedAssertions = 0;

    /**
     * @brief Minimal zero-dependency assertion helper that prints rich failure
     *        diagnostics to `stderr` for CTest (`--output-on-failure`).
     *
     * @param condition Boolean invariant expected to hold.
     * @param exprText  Stringified expression for human-readable failure output.
     * @param testName  Name of the enclosing test case.
     */
    void ExpectTrue(bool condition, const char* exprText, const char* testName) {
        if (!condition) {
            std::cerr << "[FAIL] " << testName << ": assertion failed -> " << exprText << std::endl;
            ++gFailedAssertions;
        }
    }

    /**
     * @brief Helper to construct a synthetic `GPUDeviceInfo` record for
     *        deterministic adaptation-logic verification.
     */
    [[nodiscard]] umgebung::core::GPUDeviceInfo MakeSyntheticGpu(
        int index,
        const std::string& name,
        int major,
        int minor,
        std::size_t totalGiB,
        std::size_t freeGiB,
        int smCount,
        bool highThroughputFP64
    ) {
        umgebung::core::GPUDeviceInfo gpu{};
        gpu.deviceIndex = index;
        gpu.name = name;
        gpu.computeCapabilityMajor = major;
        gpu.computeCapabilityMinor = minor;
        gpu.architectureName = umgebung::core::SystemProfiler::InferArchitectureName(major, minor);
        gpu.totalVRAMBytes = totalGiB * static_cast<std::size_t>(kGiB);
        gpu.freeVRAMBytes = freeGiB * static_cast<std::size_t>(kGiB);
        gpu.multiProcessorCount = smCount;
        gpu.warpSize = 32;
        gpu.maxThreadsPerBlock = 1024;
        gpu.maxThreadsPerMultiProcessor = 2048;
        gpu.sharedMemPerBlockBytes = 49152;
        gpu.l2CacheSizeBytes = 6 * 1024 * 1024;
        gpu.supportsManagedMemory = true;
        gpu.supportsConcurrentManagedAccess = true;
        gpu.hasHighThroughputFP64 = highThroughputFP64;
        gpu.uuid = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                    0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, static_cast<std::uint8_t>(index)};
        return gpu;
    }

    /**
     * @brief Verifies live host hardware discovery and JSON export on the active machine.
     */
    void TestLiveHostProfiling() {
        const char* testName = "TestLiveHostProfiling";
        using umgebung::core::HardwareAdaptationConfig;
        using umgebung::core::HardwareAdaptationPlan;
        using umgebung::core::HardwareCapabilities;
        using umgebung::core::SystemProfiler;

        const HardwareCapabilities caps = SystemProfiler::ProfileSystem();
        ExpectTrue(!caps.osName.empty(), "!caps.osName.empty()", testName);
        ExpectTrue(caps.cpuLogicalCores >= 1U, "caps.cpuLogicalCores >= 1U", testName);
        ExpectTrue(caps.totalSystemRAMBytes > 0ULL, "caps.totalSystemRAMBytes > 0ULL", testName);
        ExpectTrue(
            caps.availableSystemRAMBytes <= caps.totalSystemRAMBytes,
            "caps.availableSystemRAMBytes <= caps.totalSystemRAMBytes",
            testName
        );

        // On an NVIDIA-equipped host, verify CUDA discovery & adaptation end-to-end.
        ExpectTrue(caps.cudaDeviceCount >= 1, "caps.cudaDeviceCount >= 1", testName);
        ExpectTrue(!caps.gpus.empty(), "!caps.gpus.empty()", testName);

        if (!caps.gpus.empty()) {
            const auto& gpu0 = caps.gpus.front();
            ExpectTrue(!gpu0.name.empty(), "!gpu0.name.empty()", testName);
            ExpectTrue(gpu0.totalVRAMBytes > 0U, "gpu0.totalVRAMBytes > 0U", testName);
            ExpectTrue(gpu0.multiProcessorCount > 0, "gpu0.multiProcessorCount > 0", testName);
        }

        HardwareAdaptationConfig cfg{};
        cfg.guiActive = false;
        const HardwareAdaptationPlan plan = SystemProfiler::BuildAdaptationPlan(caps, cfg);
        ExpectTrue(plan.computeReady, "plan.computeReady", testName);
        ExpectTrue(plan.primaryGpuIndex >= 0, "plan.primaryGpuIndex >= 0", testName);
        ExpectTrue(plan.simulationVRAMBudgetBytes > 0U, "plan.simulationVRAMBudgetBytes > 0U", testName);
        ExpectTrue(plan.estimatedMaxFP64Bodies > 0ULL, "plan.estimatedMaxFP64Bodies > 0ULL", testName);

        // Verify JSON serialization contains both top-level sections and matches counts.
        const nlohmann::json report = SystemProfiler::ToJson(caps, plan);
        ExpectTrue(report.contains("hardware"), "report.contains(\"hardware\")", testName);
        ExpectTrue(report.contains("adaptation"), "report.contains(\"adaptation\")", testName);
        ExpectTrue(
            report["hardware"]["gpus"].size() == caps.gpus.size(),
            "report[\"hardware\"][\"gpus\"].size() == caps.gpus.size()",
            testName
        );
    }

    /**
     * @brief Verifies adaptation across Laptop, Desktop, Single-GPU HPC, Multi-GPU HPC,
     *        and Unsupported (0 GPU) synthetic topologies.
     */
    void TestSyntheticTopologyMatrix() {
        const char* testName = "TestSyntheticTopologyMatrix";
        using umgebung::core::HardwareAdaptationConfig;
        using umgebung::core::HardwareAdaptationPlan;
        using umgebung::core::HardwareCapabilities;
        using umgebung::core::HardwareTier;
        using umgebung::core::SystemProfiler;

        // 1. Personal Laptop: Core i9-13900H (20t) + RTX 4070 Laptop (8 GiB, SM 8.9) in GUI mode.
        {
            HardwareCapabilities laptop{};
            laptop.osName = "Linux";
            laptop.cpuLogicalCores = 20U;
            laptop.totalSystemRAMBytes = 32ULL * kGiB;
            laptop.availableSystemRAMBytes = 24ULL * kGiB;
            laptop.cudaDeviceCount = 1;
            laptop.gpus.push_back(MakeSyntheticGpu(0, "NVIDIA GeForce RTX 4070 Laptop GPU", 8, 9, 8, 7, 36, false));

            const HardwareAdaptationPlan plan = SystemProfiler::BuildAdaptationPlan(laptop, {.guiActive = true});
            ExpectTrue(plan.computeReady, "laptop plan.computeReady", testName);
            ExpectTrue(plan.tier == HardwareTier::LaptopDiscrete, "laptop tier == LaptopDiscrete", testName);
            ExpectTrue(plan.recommendedHostWorkerThreads == 18U, "laptop GUI reserves 2 threads", testName);
            ExpectTrue(plan.recommendedCudaStreams == 2, "laptop uses 2 CUDA streams", testName);
            ExpectTrue(!plan.enableMultiGpuDispatch, "laptop single-GPU", testName);
        }

        // 2. Personal Desktop: Ryzen 9 3900XT (24t) + TITAN RTX (24 GiB, SM 7.5), GUI vs. Headless.
        {
            HardwareCapabilities desktop{};
            desktop.osName = "Linux";
            desktop.cpuLogicalCores = 24U;
            desktop.totalSystemRAMBytes = 32ULL * kGiB;
            desktop.availableSystemRAMBytes = 28ULL * kGiB;
            desktop.cudaDeviceCount = 1;
            desktop.gpus.push_back(MakeSyntheticGpu(0, "NVIDIA TITAN RTX", 7, 5, 24, 23, 72, false));

            const HardwareAdaptationPlan guiPlan = SystemProfiler::BuildAdaptationPlan(desktop, {.guiActive = true});
            const HardwareAdaptationPlan hpcPlan = SystemProfiler::BuildAdaptationPlan(desktop, {.guiActive = false});

            ExpectTrue(guiPlan.tier == HardwareTier::DesktopWorkstation, "desktop tier == DesktopWorkstation", testName);
            ExpectTrue(guiPlan.recommendedCudaStreams == 4, "TITAN RTX 72 SMs -> 4 streams", testName);
            ExpectTrue(
                guiPlan.reservedVRAMHeadroomBytes > hpcPlan.reservedVRAMHeadroomBytes,
                "GUI mode reserves more VRAM headroom than headless mode",
                testName
            );
            ExpectTrue(
                hpcPlan.recommendedHostWorkerThreads > guiPlan.recommendedHostWorkerThreads,
                "Headless mode allocates more host worker threads than GUI mode",
                testName
            );
        }

        // 3. Multi-GPU HPC Node (e.g., ARCSIM / SPORC): 4x A100 80 GiB (SM 8.0).
        {
            HardwareCapabilities hpcNode{};
            hpcNode.osName = "Linux";
            hpcNode.cpuLogicalCores = 128U;
            hpcNode.totalSystemRAMBytes = 512ULL * kGiB;
            hpcNode.availableSystemRAMBytes = 480ULL * kGiB;
            hpcNode.cudaDeviceCount = 4;
            hpcNode.gpus.push_back(MakeSyntheticGpu(0, "NVIDIA A100-SXM4-80GB", 8, 0, 80, 78, 108, true));
            hpcNode.gpus.push_back(MakeSyntheticGpu(1, "NVIDIA A100-SXM4-80GB", 8, 0, 80, 79, 108, true));
            hpcNode.gpus.push_back(MakeSyntheticGpu(2, "NVIDIA A100-SXM4-80GB", 8, 0, 80, 79, 108, true));
            hpcNode.gpus.push_back(MakeSyntheticGpu(3, "NVIDIA A100-SXM4-80GB", 8, 0, 80, 79, 108, true));

            // Explicit CLI `--device 2` pin should select GPU 2 as primary.
            HardwareAdaptationConfig pinnedCfg{};
            pinnedCfg.guiActive = false;
            pinnedCfg.preferredDeviceIndex = 2;

            const HardwareAdaptationPlan plan = SystemProfiler::BuildAdaptationPlan(hpcNode, pinnedCfg);
            ExpectTrue(plan.tier == HardwareTier::MultiGpuHpcNode, "multi-GPU tier == MultiGpuHpcNode", testName);
            ExpectTrue(plan.enableMultiGpuDispatch, "multi-GPU dispatch enabled", testName);
            ExpectTrue(plan.primaryGpuIndex == 2, "pinned primaryGpuIndex == 2", testName);
            ExpectTrue(plan.activeGpuIndices.size() == 4U, "all 4 GPUs listed in activeGpuIndices", testName);
            ExpectTrue(plan.activeGpuIndices.front() == 2, "primary GPU ordered first", testName);
        }

        // 4. Unsupported Host (0 CUDA GPUs).
        {
            HardwareCapabilities noGpu{};
            noGpu.osName = "Linux";
            noGpu.cpuLogicalCores = 8U;
            noGpu.totalSystemRAMBytes = 16ULL * kGiB;
            noGpu.availableSystemRAMBytes = 12ULL * kGiB;
            noGpu.cudaDeviceCount = 0;

            const HardwareAdaptationPlan plan = SystemProfiler::BuildAdaptationPlan(noGpu, {.guiActive = false});
            ExpectTrue(!plan.computeReady, "no-GPU plan.computeReady == false", testName);
            ExpectTrue(plan.tier == HardwareTier::Unsupported, "no-GPU tier == Unsupported", testName);
            ExpectTrue(plan.primaryGpuIndex == -1, "no-GPU primaryGpuIndex == -1", testName);
        }
    }

    /**
     * @brief Verifies formatting helpers (CUDA version, architecture naming, UUID).
     */
    void TestFormattingHelpers() {
        const char* testName = "TestFormattingHelpers";
        using umgebung::core::SystemProfiler;

        ExpectTrue(SystemProfiler::FormatCudaVersion(13040) == "13.4", "FormatCudaVersion(13040) == 13.4", testName);
        ExpectTrue(SystemProfiler::FormatCudaVersion(0) == "Unavailable", "FormatCudaVersion(0) == Unavailable", testName);
        ExpectTrue(SystemProfiler::InferArchitectureName(7, 0) == "Volta", "SM 7.0 == Volta", testName);
        ExpectTrue(SystemProfiler::InferArchitectureName(7, 5) == "Turing", "SM 7.5 == Turing", testName);
        ExpectTrue(SystemProfiler::InferArchitectureName(8, 0) == "Ampere (Datacenter)", "SM 8.0 == Ampere (Datacenter)", testName);
        ExpectTrue(SystemProfiler::InferArchitectureName(8, 9) == "Ada Lovelace", "SM 8.9 == Ada Lovelace", testName);

        const std::array<std::uint8_t, 16> sampleUuid{
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
            0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
        };
        ExpectTrue(
            SystemProfiler::FormatUuid(sampleUuid) == "01234567-89ab-cdef-fedc-ba9876543210",
            "FormatUuid matches RFC-4122 layout",
            testName
        );
    }

} // anonymous namespace

int main() {
    TestLiveHostProfiling();
    TestSyntheticTopologyMatrix();
    TestFormattingHelpers();

    if (gFailedAssertions > 0) {
        std::cerr << "SystemProfilerTests failed with " << gFailedAssertions << " error(s)." << std::endl;
        return 1;
    }

    std::cout << "All SystemProfilerTests passed." << std::endl;
    return 0;
}
