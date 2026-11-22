/**
 * @file StatisticsPanel.cpp
 * @brief Implementation of the floating non-docking `"Statistics"` panel
 *        combining legacy Umgebung's FPS/frame-time overlay with current
 *        Umgebung's Hardware & Adaptation + CUDA VRAM telemetry.
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#include "umgebung/ui/panels/StatisticsPanel.hpp"
#include "umgebung/compute/CudaUtils.hpp"
#include "umgebung/ui/UIContext.hpp"

#include <algorithm>

#include <imgui.h>

namespace umgebung::ui {

    StatisticsPanel::StatisticsPanel()
        : Panel(
              "Statistics",
              false,
              ImGuiWindowFlags_NoResize |
                  ImGuiWindowFlags_NoCollapse |
                  ImGuiWindowFlags_NoDocking |
                  ImGuiWindowFlags_NoScrollbar |
                  ImGuiWindowFlags_AlwaysAutoResize
          ) {}

    void StatisticsPanel::OnUIRender(UIContext& ctx) {
        if (!m_isOpen) {
            return;
        }

        constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;

        if (ImGui::Begin(m_name.c_str(), &m_isOpen, m_flags)) {
            m_focused = ImGui::IsWindowFocused();
            m_hovered = ImGui::IsWindowHovered();

            const ImGuiIO& io = ImGui::GetIO();
            const float fps = std::max(0.001f, io.Framerate);

            // Legacy StatisticsPanel live frame-rate readout.
            ImGui::Text("Frame Rate     : %.1f FPS (%.3f ms/frame)", fps, 1000.0f / fps);
            ImGui::Text("Playback State : %s", SimulationPlaybackStateToString(ctx.playbackState));
            if (ctx.latestDiagnostics != nullptr) {
                ImGui::Text(
                    "Sim Progress   : Step %llu | t = %.5f | Sub-Steps/Frame = %d",
                    static_cast<unsigned long long>(ctx.latestDiagnostics->stepCount),
                    ctx.latestDiagnostics->simulationTime,
                    ctx.lastFrameExecutedSteps
                );
            }

            // Current Umgebung Hardware & Adaptation diagnostics.
            if (ctx.hardwareCaps != nullptr && ctx.adaptationPlan != nullptr) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.45f, 0.85f, 1.0f, 1.0f), "Hardware & Adaptation");
                ImGui::Text(
                    "Engine Tier    : v%s (%s)",
                    ctx.engineVersion ? ctx.engineVersion : "0.0.1",
                    core::SystemProfiler::HardwareTierToString(ctx.adaptationPlan->tier)
                );
                ImGui::Text(
                    "CPU            : %s (%u threads)",
                    ctx.hardwareCaps->cpuModelName.c_str(),
                    ctx.hardwareCaps->cpuLogicalCores
                );
                const double totalRamGiB = static_cast<double>(ctx.hardwareCaps->totalSystemRAMBytes) / kBytesPerGiB;
                const double availRamGiB = static_cast<double>(ctx.hardwareCaps->availableSystemRAMBytes) / kBytesPerGiB;
                ImGui::Text(
                    "Host RAM       : %.2f / %.2f GiB used",
                    std::max(0.0, totalRamGiB - availRamGiB),
                    totalRamGiB
                );
                ImGui::Text(
                    "Vulkan GPU     : %s (CUDA-UUID Match: %s)",
                    ctx.selectedVulkanGpuName.empty() ? "Unknown" : ctx.selectedVulkanGpuName.c_str(),
                    ctx.vulkanMatchedCudaUuid ? "YES" : "NO"
                );
                if (ctx.cudaContext != nullptr && ctx.cudaContext->IsInitialized()) {
                    ImGui::Text(
                        "CUDA Compute   : GPU [%d] | %zu streams | VRAM: %s / %s",
                        ctx.cudaContext->GetDeviceIndex(),
                        ctx.cudaContext->GetStreamCount(),
                        compute::FormatBytes(ctx.cudaContext->GetTrackedAllocatedBytes()).c_str(),
                        compute::FormatBytes(ctx.cudaContext->GetVRAMBudgetBytes()).c_str()
                    );
                }
            }
        }
        ImGui::End();
    }

} // namespace umgebung::ui
