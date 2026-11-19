/**
 * @file ComputeTests.cpp
 * @brief Automated CTest verification suite for `umgebung::compute`
 *        (`CudaUtils`, `CUDAContext`, `DeviceBuffer<T>`, and atomic VRAM budget enforcement).
 *
 * @details
 * Validates live NVIDIA GPU execution on the host device:
 *   1. `CUDAContext` initialization from `SystemProfiler`'s `HardwareAdaptationPlan`,
 *      stream pool creation, and live `cudaMemGetInfo` queries.
 *   2. `DeviceBuffer<double>` allocation, synchronous and stream-ordered H2D/D2H
 *      bit-exact data round-trips, and `ZeroFill`.
 *   3. Move semantics (`DeviceBuffer&&`) preserving exact VRAM budget accounting
 *      without double-counting or leaking reserved bytes.
 *   4. Strict VRAM budget rejection: attempting to allocate beyond
 *      `simulationVRAMBudgetBytes` fails cleanly and leaves tracked VRAM unchanged.
 */

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/compute/DeviceBuffer.hpp"
#include "umgebung/core/SystemProfiler.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

namespace {

    int gFailedAssertions = 0;

    void ExpectTrue(bool condition, const char* exprText, const char* testName) {
        if (!condition) {
            std::cerr << "[FAIL] " << testName << ": assertion failed -> " << exprText << std::endl;
            ++gFailedAssertions;
        }
    }

    /**
     * @brief Verifies `ComputeGridSize1D` edge cases and ceiling division math.
     */
    void TestLaunchGeometryHelpers() {
        const char* testName = "TestLaunchGeometryHelpers";
        using umgebung::compute::ComputeGridSize1D;

        ExpectTrue(ComputeGridSize1D(0U, 256) == 0U, "ComputeGridSize1D(0, 256) == 0", testName);
        ExpectTrue(ComputeGridSize1D(1U, 256) == 1U, "ComputeGridSize1D(1, 256) == 1", testName);
        ExpectTrue(ComputeGridSize1D(256U, 256) == 1U, "ComputeGridSize1D(256, 256) == 1", testName);
        ExpectTrue(ComputeGridSize1D(257U, 256) == 2U, "ComputeGridSize1D(257, 256) == 2", testName);
        ExpectTrue(ComputeGridSize1D(1024U, 256) == 4U, "ComputeGridSize1D(1024, 256) == 4", testName);
        ExpectTrue(ComputeGridSize1D(100U, 0) == 0U, "ComputeGridSize1D(100, 0) == 0", testName);
    }

    /**
     * @brief Verifies live `CUDAContext`, `DeviceBuffer<double>` H2D/D2H round-trip,
     *        move semantics, and exact VRAM budget accounting.
     */
    void TestCudaContextAndDeviceBuffer() {
        const char* testName = "TestCudaContextAndDeviceBuffer";
        using umgebung::compute::CUDAContext;
        using umgebung::compute::DeviceBuffer;
        using umgebung::core::HardwareAdaptationConfig;
        using umgebung::core::HardwareAdaptationPlan;
        using umgebung::core::HardwareCapabilities;
        using umgebung::core::SystemProfiler;

        const HardwareCapabilities caps = SystemProfiler::ProfileSystem();
        const HardwareAdaptationPlan plan = SystemProfiler::BuildAdaptationPlan(caps, HardwareAdaptationConfig{.guiActive = false});
        ExpectTrue(plan.computeReady, "plan.computeReady", testName);

        CUDAContext ctx;
        ExpectTrue(ctx.Initialize(caps, plan), "ctx.Initialize(caps, plan)", testName);
        ExpectTrue(ctx.IsInitialized(), "ctx.IsInitialized()", testName);
        ExpectTrue(ctx.GetStreamCount() >= 1U, "ctx.GetStreamCount() >= 1U", testName);
        ExpectTrue(ctx.GetStream(0U) != nullptr, "ctx.GetStream(0) != nullptr", testName);
        ExpectTrue(ctx.GetTrackedAllocatedBytes() == 0U, "initial tracked VRAM == 0", testName);

        std::size_t liveFree = 0U;
        std::size_t liveTotal = 0U;
        ExpectTrue(ctx.QueryLiveDeviceVRAM(liveFree, liveTotal), "ctx.QueryLiveDeviceVRAM", testName);
        ExpectTrue(liveTotal > 0U && liveFree <= liveTotal, "liveFree <= liveTotal && liveTotal > 0", testName);

        // Allocate 4096 FP64 values (32 KiB) and verify H2D -> D2H bit-exact round trip on stream 0.
        constexpr std::size_t kCount = 4096U;
        std::vector<double> hostIn(kCount);
        for (std::size_t i = 0U; i < kCount; ++i) {
            hostIn[i] = static_cast<double>(i) * 1.125 - 42.5;
        }

        {
            DeviceBuffer<double> buf;
            ExpectTrue(buf.Allocate(ctx, kCount), "buf.Allocate(ctx, 4096)", testName);
            ExpectTrue(buf.IsAllocated(), "buf.IsAllocated()", testName);
            ExpectTrue(buf.Count() == kCount, "buf.Count() == kCount", testName);
            ExpectTrue(buf.SizeBytes() == kCount * sizeof(double), "buf.SizeBytes() == 32768", testName);
            ExpectTrue(
                ctx.GetTrackedAllocatedBytes() == kCount * sizeof(double),
                "ctx tracked VRAM == 32768 after Allocate",
                testName
            );

            cudaStream_t stream = ctx.GetStream(0U);
            ExpectTrue(buf.CopyFromHost(hostIn, stream), "buf.CopyFromHost async", testName);

            std::vector<double> hostOut;
            ExpectTrue(buf.CopyToHost(hostOut, stream), "buf.CopyToHost async", testName);
            ExpectTrue(ctx.SynchronizeStream(0U), "ctx.SynchronizeStream(0)", testName);
            ExpectTrue(hostOut.size() == kCount, "hostOut.size() == kCount", testName);

            bool allExact = true;
            for (std::size_t i = 0U; i < kCount; ++i) {
                if (hostOut[i] != hostIn[i]) {
                    allExact = false;
                    break;
                }
            }
            ExpectTrue(allExact, "H2D -> D2H FP64 values match bit-for-bit", testName);

            // Verify ZeroFill clears all bytes to 0.0.
            ExpectTrue(buf.ZeroFill(stream), "buf.ZeroFill", testName);
            ExpectTrue(buf.CopyToHost(hostOut, stream), "buf.CopyToHost after ZeroFill", testName);
            ExpectTrue(ctx.SynchronizeStream(0U), "ctx.SynchronizeStream after ZeroFill", testName);
            bool allZero = true;
            for (double v : hostOut) {
                if (v != 0.0) {
                    allZero = false;
                    break;
                }
            }
            ExpectTrue(allZero, "ZeroFill cleared all FP64 entries to 0.0", testName);

            // Verify move construction transfers ownership cleanly without double-counting VRAM.
            DeviceBuffer<double> movedBuf(std::move(buf));
            ExpectTrue(!buf.IsAllocated() && buf.Count() == 0U, "moved-from buf is empty", testName);
            ExpectTrue(movedBuf.IsAllocated() && movedBuf.Count() == kCount, "movedBuf owns allocation", testName);
            ExpectTrue(
                ctx.GetTrackedAllocatedBytes() == kCount * sizeof(double),
                "VRAM accounting unchanged after move",
                testName
            );
        }

        // After `movedBuf` goes out of scope, tracked VRAM must return strictly to 0.
        ExpectTrue(ctx.GetTrackedAllocatedBytes() == 0U, "tracked VRAM returns to 0 after RAII destroy", testName);

        ctx.Shutdown();
    }

    /**
     * @brief Verifies that `CUDAContext` strictly rejects allocations exceeding
     *        its adapted `simulationVRAMBudgetBytes` ceiling.
     */
    void TestVramBudgetEnforcement() {
        const char* testName = "TestVramBudgetEnforcement";
        using umgebung::compute::CUDAContext;
        using umgebung::compute::DeviceBuffer;
        using umgebung::core::HardwareAdaptationConfig;
        using umgebung::core::HardwareAdaptationPlan;
        using umgebung::core::HardwareCapabilities;
        using umgebung::core::SystemProfiler;

        const HardwareCapabilities caps = SystemProfiler::ProfileSystem();
        HardwareAdaptationPlan tightPlan =
            SystemProfiler::BuildAdaptationPlan(caps, HardwareAdaptationConfig{.guiActive = false});

        // Artificially constrain the simulation VRAM budget to exactly 64 KiB.
        tightPlan.simulationVRAMBudgetBytes = 64ULL * 1024ULL;

        CUDAContext ctx;
        ExpectTrue(ctx.Initialize(caps, tightPlan), "tight budget ctx.Initialize", testName);

        DeviceBuffer<double> fitsBuf;
        // 4096 doubles = 32 KiB (fits within 64 KiB).
        ExpectTrue(fitsBuf.Allocate(ctx, 4096U), "32 KiB allocation fits within 64 KiB budget", testName);
        ExpectTrue(ctx.GetTrackedAllocatedBytes() == 32768U, "tracked VRAM == 32768 B", testName);
        ExpectTrue(ctx.GetRemainingVRAMBudgetBytes() == 32768U, "remaining budget == 32768 B", testName);

        DeviceBuffer<double> exceedsBuf;
        // 8192 doubles = 64 KiB (32 + 64 = 96 KiB > 64 KiB -> MUST fail cleanly).
        ExpectTrue(!exceedsBuf.Allocate(ctx, 8192U), "64 KiB second allocation rejected cleanly", testName);
        ExpectTrue(!exceedsBuf.IsAllocated(), "rejected buffer remains unallocated", testName);
        ExpectTrue(ctx.GetTrackedAllocatedBytes() == 32768U, "tracked VRAM unchanged after rejection", testName);

        fitsBuf.Release();
        ExpectTrue(ctx.GetTrackedAllocatedBytes() == 0U, "tracked VRAM == 0 after Release()", testName);

        ctx.Shutdown();
    }

} // anonymous namespace

int main() {
    TestLaunchGeometryHelpers();
    TestCudaContextAndDeviceBuffer();
    TestVramBudgetEnforcement();

    if (gFailedAssertions > 0) {
        std::cerr << "ComputeTests failed with " << gFailedAssertions << " error(s)." << std::endl;
        return 1;
    }

    std::cout << "All ComputeTests passed." << std::endl;
    return 0;
}
