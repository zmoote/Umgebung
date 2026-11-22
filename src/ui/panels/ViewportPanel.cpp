/**
 * @file ViewportPanel.cpp
 * @brief Implementation of the center-docked `"Viewport"` panel adapting legacy
 *        Umgebung's zero-padding viewport + playback-state indicator to current
 *        Umgebung's four domain-adaptive visualizers (`FlowerOfLifeMultiverse`,
 *        `NBodyDynamics`, `FieldGrid2D`, and `RadialAndScaleProfile`).
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#include "umgebung/ui/panels/ViewportPanel.hpp"
#include "umgebung/core/Application.hpp"
#include "umgebung/ui/UIContext.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>
#include <vector>

#include <imgui.h>

namespace umgebung::ui {

    namespace {

        /**
         * @brief Smooth perceptual inferno / cool-cyan RGB ramp for 2D GPU field
         *        heatmaps (`t in [0, 1]`).
         */
        [[nodiscard]] ImU32 EvaluateHeatmapColor(float t, bool isDiffMap) noexcept {
            const float u = std::clamp(t, 0.0f, 1.0f);
            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;
            if (isDiffMap) {
                // Cool-indigo -> cyan -> bright white-gold for |A - B| difference maps.
                if (u < 0.5f) {
                    const float s = u * 2.0f;
                    r = 0.06f + (0.10f * s);
                    g = 0.08f + (0.68f * s);
                    b = 0.18f + (0.78f * s);
                } else {
                    const float s = (u - 0.5f) * 2.0f;
                    r = 0.16f + (0.84f * s);
                    g = 0.76f + (0.22f * s);
                    b = 0.96f - (0.35f * s);
                }
            } else {
                // Inferno ramp: deep navy -> magenta -> fiery orange -> bright yellow.
                if (u < 0.33f) {
                    const float s = u / 0.33f;
                    r = 0.05f + (0.50f * s);
                    g = 0.05f + (0.08f * s);
                    b = 0.15f + (0.45f * s);
                } else if (u < 0.67f) {
                    const float s = (u - 0.33f) / 0.34f;
                    r = 0.55f + (0.42f * s);
                    g = 0.13f + (0.37f * s);
                    b = 0.60f - (0.48f * s);
                } else {
                    const float s = (u - 0.67f) / 0.33f;
                    r = 0.97f + (0.03f * s);
                    g = 0.50f + (0.48f * s);
                    b = 0.12f + (0.68f * s);
                }
            }
            return IM_COL32(
                static_cast<int>(std::clamp(r, 0.0f, 1.0f) * 255.0f),
                static_cast<int>(std::clamp(g, 0.0f, 1.0f) * 255.0f),
                static_cast<int>(std::clamp(b, 0.0f, 1.0f) * 255.0f),
                255
            );
        }

        /**
         * @brief Returns a subtle accent border color corresponding to the active
         *        playback state (adapted cleanly from legacy `vulkan-backend`'s
         *        green=Simulating / yellow=Paused Viewport feedback).
         */
        [[nodiscard]] ImU32 PlaybackStateBorderColor(SimulationPlaybackState state) noexcept {
            switch (state) {
                case SimulationPlaybackState::Playing: return IM_COL32(70, 215, 130, 185);
                case SimulationPlaybackState::Paused:  return IM_COL32(245, 195, 70, 185);
                case SimulationPlaybackState::Stopped: return IM_COL32(95, 120, 160, 150);
            }
            return IM_COL32(95, 120, 160, 150);
        }

    } // anonymous namespace

    ViewportPanel::ViewportPanel()
        : Panel("Viewport", true) {}

    void ViewportPanel::RenderViewportToolbar(UIContext& ctx) {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6.0f, 4.0f));
        ImGui::SetCursorPos(ImVec2(8.0f, ImGui::GetCursorPosY() + 4.0f));

        // Playback transport controls (Play / Pause / Stop / Step / Reset).
        const bool isRadial =
            (ctx.hasActiveScene && ctx.activeScene != nullptr &&
             ctx.activeScene->domain == scene::SimulationDomain::RadialAndScaleProfile);

        ImGui::BeginDisabled(isRadial || !ctx.hasActiveScene);
        const bool isPlaying = (ctx.playbackState == SimulationPlaybackState::Playing);
        if (isPlaying) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.50f, 0.30f, 1.0f));
        }
        if (ImGui::SmallButton("Play")) {
            if (ctx.onPlay) { ctx.onPlay(); }
        }
        if (isPlaying) {
            ImGui::PopStyleColor();
        }
        ImGui::SameLine();

        const bool isPaused = (ctx.playbackState == SimulationPlaybackState::Paused);
        if (isPaused) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.62f, 0.46f, 0.14f, 1.0f));
        }
        if (ImGui::SmallButton("Pause")) {
            if (ctx.onPause) { ctx.onPause(); }
        }
        if (isPaused) {
            ImGui::PopStyleColor();
        }
        ImGui::SameLine();

        if (ImGui::SmallButton("Stop")) {
            if (ctx.onStop) { ctx.onStop(); }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Step")) {
            if (ctx.onSingleStep) { ctx.onSingleStep(); }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset")) {
            if (ctx.onResetSimulation) { ctx.onResetSimulation(); }
        }
        ImGui::EndDisabled();

        // Compact domain-specific inline controls.
        if (ctx.hasActiveScene && ctx.activeScene != nullptr) {
            const bool latticeMode =
                (ctx.showFlowerOfLifeLatticeView != nullptr && *ctx.showFlowerOfLifeLatticeView);
            if (!latticeMode && ctx.activeScene->domain == scene::SimulationDomain::NBodyDynamics) {
                ImGui::SameLine();
                ImGui::TextDisabled("|");
                ImGui::SameLine();
                if (ctx.viewportAutoFit != nullptr) {
                    ImGui::Checkbox("Auto-Fit", ctx.viewportAutoFit);
                }
                if (ctx.viewportHalfExtent != nullptr) {
                    ImGui::SameLine();
                    ImGui::BeginDisabled(ctx.viewportAutoFit != nullptr && *ctx.viewportAutoFit);
                    ImGui::SetNextItemWidth(115.0f);
                    ImGui::SliderFloat("##ViewRadius", ctx.viewportHalfExtent, 0.25f, 100.0f, "R=%.2f", ImGuiSliderFlags_Logarithmic);
                    ImGui::EndDisabled();
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Center")) {
                    if (ctx.viewportPanX != nullptr) { *ctx.viewportPanX = 0.0f; }
                    if (ctx.viewportPanY != nullptr) { *ctx.viewportPanY = 0.0f; }
                }
            } else if (!latticeMode && ctx.activeScene->domain == scene::SimulationDomain::FieldGrid2D &&
                       ctx.fieldHeatmapMode != nullptr) {
                ImGui::SameLine();
                ImGui::TextDisabled("|");
                ImGui::SameLine();
                ImGui::RadioButton("Model A", ctx.fieldHeatmapMode, 0);
                if (ctx.compareOverlayEnabled &&
                    ctx.compareFieldSnap != nullptr &&
                    !ctx.compareFieldSnap->scalarField.empty()) {
                    ImGui::SameLine();
                    ImGui::RadioButton("Model B", ctx.fieldHeatmapMode, 1);
                    ImGui::SameLine();
                    ImGui::RadioButton("|A - B|", ctx.fieldHeatmapMode, 2);
                } else {
                    *ctx.fieldHeatmapMode = 0;
                }
            }
        }

        // State pill readout.
        ImGui::SameLine();
        ImVec4 stateCol(0.65f, 0.72f, 0.85f, 1.0f);
        if (ctx.playbackState == SimulationPlaybackState::Playing) {
            stateCol = ImVec4(0.45f, 0.92f, 0.62f, 1.0f);
        } else if (ctx.playbackState == SimulationPlaybackState::Paused) {
            stateCol = ImVec4(1.0f, 0.82f, 0.36f, 1.0f);
        }
        ImGui::TextColored(stateCol, "[%s]", SimulationPlaybackStateToString(ctx.playbackState));

        ImGui::PopStyleVar();
        ImGui::Separator();
    }

    void ViewportPanel::RenderFlowerOfLifeCanvas(UIContext& ctx) {
        if (ctx.flowerMultiverse == nullptr) {
            return;
        }
        const ImVec2 canvasMin = ImGui::GetCursorScreenPos();
        const ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        if (canvasSize.x <= 32.0f || canvasSize.y <= 32.0f) {
            return;
        }

        const ImVec2 canvasMax(canvasMin.x + canvasSize.x, canvasMin.y + canvasSize.y);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->PushClipRect(canvasMin, canvasMax, true);

        drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(8, 10, 22, 255));
        drawList->AddRect(canvasMin, canvasMax, IM_COL32(95, 80, 165, 230), 0.0f, 0, 2.0f);

        const ImVec2 center(canvasMin.x + 0.5f * canvasSize.x, canvasMin.y + 0.5f * canvasSize.y);
        const float minDim = std::min(canvasSize.x, canvasSize.y);
        const double bubbleR = std::max(0.1, ctx.flowerMultiverse->GetBubbleRadius());
        const float pixelsPerUnit = (0.145f * minDim) / static_cast<float>(bubbleR);

        drawList->AddCircle(center, 3.05f * static_cast<float>(bubbleR) * pixelsPerUnit, IM_COL32(120, 95, 210, 120), 96, 2.0f);

        // Allow direct click-to-enter interaction on populated Flower of Life Bubble Universes.
        ImGui::InvisibleButton("##FlowerLatticeCanvas", canvasSize);
        const bool clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
        const ImVec2 mousePos = ImGui::GetMousePos();

        const auto& nodes = ctx.flowerMultiverse->GetNodes();
        for (std::size_t idx = 0U; idx < nodes.size(); ++idx) {
            const auto& node = nodes[idx];
            const ImVec2 pos(
                center.x + static_cast<float>(node.latticePosition.x) * pixelsPerUnit,
                center.y - static_cast<float>(node.latticePosition.y) * pixelsPerUnit
            );
            const float radiusPx = static_cast<float>(node.bubbleRadius) * pixelsPerUnit;
            const bool populated = node.loadedUniverse.has_value();
            const bool isSelected = (idx == ctx.flowerMultiverse->GetActiveNodeIndex());

            if (populated) {
                drawList->AddCircleFilled(
                    pos, radiusPx,
                    isSelected ? IM_COL32(75, 185, 255, 48) : IM_COL32(150, 115, 255, 36),
                    64
                );
                if (clicked) {
                    const float dx = mousePos.x - pos.x;
                    const float dy = mousePos.y - pos.y;
                    if ((dx * dx + dy * dy) <= (radiusPx * radiusPx * 0.55f)) {
                        (void)ctx.flowerMultiverse->SetActiveNodeIndex(idx);
                        if (ctx.showFlowerOfLifeLatticeView != nullptr) {
                            *ctx.showFlowerOfLifeLatticeView = false;
                        }
                        if (ctx.onActivateScene) {
                            (void)ctx.onActivateScene(
                                node.loadedUniverse.value(),
                                scene::SimulationScope::FlowerOfLifeMultiverse,
                                std::nullopt,
                                std::nullopt
                            );
                        }
                    }
                }
            }
            const ImU32 ringCol = isSelected
                ? IM_COL32(110, 235, 255, 240)
                : (populated ? IM_COL32(255, 210, 115, 210) : IM_COL32(125, 115, 175, 110));
            drawList->AddCircle(pos, radiusPx, ringCol, 64, populated ? 2.2f : 1.1f);

            if (populated) {
                drawList->AddCircleFilled(pos, 5.0f, ringCol);
                drawList->AddText(ImVec2(pos.x + 8.0f, pos.y - 8.0f), IM_COL32(240, 245, 255, 245), node.label.c_str());
            }
        }
        drawList->AddText(
            ImVec2(canvasMin.x + 12.0f, canvasMin.y + 10.0f),
            IM_COL32(195, 180, 255, 230),
            "Elena Danaan Flower of Life Multiverse (Click a Populated Bubble Universe to Enter)"
        );
        drawList->PopClipRect();
    }

    void ViewportPanel::RenderOrbitalCanvas(UIContext& ctx) {
        if (ctx.liveBodies == nullptr || ctx.viewportHalfExtent == nullptr || ctx.viewportAutoFit == nullptr) {
            return;
        }

        const auto& liveBodies = *ctx.liveBodies;
        float panX = (ctx.viewportPanX != nullptr) ? *ctx.viewportPanX : 0.0f;
        float panY = (ctx.viewportPanY != nullptr) ? *ctx.viewportPanY : 0.0f;

        double activeHalfExtent = static_cast<double>(*ctx.viewportHalfExtent);
        if (*ctx.viewportAutoFit) {
            activeHalfExtent = 1.35;
            for (const auto& b : liveBodies) {
                activeHalfExtent = std::max(activeHalfExtent, std::max(std::abs(b.position.x - panX), std::abs(b.position.y - panY)) * 1.25);
            }
            if (ctx.compareOverlayEnabled && ctx.compareBodies != nullptr) {
                for (const auto& b : *ctx.compareBodies) {
                    activeHalfExtent = std::max(activeHalfExtent, std::max(std::abs(b.position.x - panX), std::abs(b.position.y - panY)) * 1.25);
                }
            }
            *ctx.viewportHalfExtent = static_cast<float>(activeHalfExtent);
        }

        const ImVec2 canvasMin = ImGui::GetCursorScreenPos();
        const ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        if (canvasSize.x <= 32.0f || canvasSize.y <= 32.0f) {
            return;
        }

        const ImVec2 canvasMax(canvasMin.x + canvasSize.x, canvasMin.y + canvasSize.y);
        const ImVec2 center(canvasMin.x + 0.5f * canvasSize.x, canvasMin.y + 0.5f * canvasSize.y);
        const float minCanvasDim = std::min(canvasSize.x, canvasSize.y);
        const float pixelsPerUnit = (0.45f * minCanvasDim) / static_cast<float>(std::max(1.0e-4, activeHalfExtent));

        // Interactive canvas hit-target: mouse wheel zoom + right/middle mouse drag pan.
        ImGui::InvisibleButton("##OrbitalViewportCanvas", canvasSize, ImGuiButtonFlags_MouseButtonRight | ImGuiButtonFlags_MouseButtonMiddle);
        if (ImGui::IsItemHovered()) {
            const float wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.0f) {
                *ctx.viewportAutoFit = false;
                const float zoomFactor = std::pow(0.88f, wheel);
                *ctx.viewportHalfExtent = std::clamp(*ctx.viewportHalfExtent * zoomFactor, 0.25f, 200.0f);
            }
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Right) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                const ImVec2 delta = ImGui::GetIO().MouseDelta;
                if (pixelsPerUnit > 1.0e-4f) {
                    panX -= delta.x / pixelsPerUnit;
                    panY += delta.y / pixelsPerUnit;
                    if (ctx.viewportPanX != nullptr) { *ctx.viewportPanX = panX; }
                    if (ctx.viewportPanY != nullptr) { *ctx.viewportPanY = panY; }
                }
            }
        }

        auto worldToScreen = [&](double wx, double wy) -> ImVec2 {
            return ImVec2(
                center.x + static_cast<float>(wx - static_cast<double>(panX)) * pixelsPerUnit,
                center.y - static_cast<float>(wy - static_cast<double>(panY)) * pixelsPerUnit
            );
        };

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->PushClipRect(canvasMin, canvasMax, true);

        drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(10, 14, 24, 255));

        const ImVec2 originPx = worldToScreen(0.0, 0.0);
        drawList->AddLine(ImVec2(canvasMin.x, originPx.y), ImVec2(canvasMax.x, originPx.y), IM_COL32(40, 55, 80, 160));
        drawList->AddLine(ImVec2(originPx.x, canvasMin.y), ImVec2(originPx.x, canvasMax.y), IM_COL32(40, 55, 80, 160));
        drawList->AddCircle(originPx, 0.225f * minCanvasDim, IM_COL32(40, 58, 86, 110), 64, 1.0f);
        drawList->AddCircle(originPx, 0.450f * minCanvasDim, IM_COL32(40, 58, 86, 110), 64, 1.0f);

        static constexpr ImU32 kPalette[6] = {
            IM_COL32(255, 210, 90, 255), IM_COL32(95, 190, 255, 255),
            IM_COL32(255, 115, 145, 255), IM_COL32(130, 245, 165, 255),
            IM_COL32(195, 150, 255, 255), IM_COL32(255, 165, 85, 255)
        };

        if (ctx.compareOverlayEnabled && ctx.compareTrails != nullptr) {
            const auto& cTrails = *ctx.compareTrails;
            for (std::size_t i = 0U; i < cTrails.size(); ++i) {
                const auto& trail = cTrails[i];
                if (trail.size() < 2U) { continue; }
                const ImU32 ghostColor = (kPalette[i % 6U] & 0x00FFFFFFU) | (150U << 24U);
                for (std::size_t k = 1U; k < trail.size(); k += 2U) {
                    drawList->AddLine(
                        worldToScreen(trail[k - 1U].x, trail[k - 1U].y),
                        worldToScreen(trail[k].x, trail[k].y),
                        ghostColor, 1.3f
                    );
                }
            }
        }

        if (ctx.bodyTrails != nullptr) {
            const auto& trails = *ctx.bodyTrails;
            for (std::size_t i = 0U; i < trails.size(); ++i) {
                const auto& trail = trails[i];
                if (trail.size() < 2U) { continue; }
                const ImU32 trailColor = (kPalette[i % 6U] & 0x00FFFFFFU) | (175U << 24U);
                for (std::size_t k = 1U; k < trail.size(); ++k) {
                    drawList->AddLine(
                        worldToScreen(trail[k - 1U].x, trail[k - 1U].y),
                        worldToScreen(trail[k].x, trail[k].y),
                        trailColor, 1.8f
                    );
                }
            }
        }

        for (std::size_t i = 0U; i < liveBodies.size(); ++i) {
            const auto& bA = liveBodies[i];
            const ImVec2 posA = worldToScreen(bA.position.x, bA.position.y);
            const ImU32 color = kPalette[i % 6U];
            const float markerRadius = (bA.category == "Star" || bA.category == "GalacticCore") ? 7.0f : 5.0f;

            const bool isSelected =
                (ctx.selectedBodyIndex != nullptr &&
                 ctx.selectedBodyIndex->has_value() &&
                 ctx.selectedBodyIndex->value() == i);

            if (ctx.compareOverlayEnabled && ctx.compareBodies != nullptr && i < ctx.compareBodies->size()) {
                const auto& bB = (*ctx.compareBodies)[i];
                const ImVec2 posB = worldToScreen(bB.position.x, bB.position.y);
                drawList->AddLine(posA, posB, IM_COL32(255, 255, 255, 85), 1.0f);
                drawList->AddCircle(posB, markerRadius + 2.5f, color, 24, 2.0f);
            }
            if (isSelected) {
                drawList->AddCircle(posA, markerRadius + 5.5f, IM_COL32(120, 240, 255, 240), 32, 2.0f);
            }
            drawList->AddCircleFilled(posA, markerRadius, color);
            drawList->AddText(ImVec2(posA.x + markerRadius + 4.0f, posA.y - 7.0f), IM_COL32(225, 232, 245, 230), bA.name.c_str());
        }

        const std::string legendA = std::string("Model A (Solid): ") + scene::ModelIdToString(ctx.primaryModelId);
        drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 8.0f), IM_COL32(160, 230, 255, 240), legendA.c_str());
        if (ctx.compareOverlayEnabled) {
            const std::string legendB = std::string("Model B (Dashed/Ring): ") + scene::ModelIdToString(ctx.compareModelId);
            drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 24.0f), IM_COL32(255, 210, 130, 240), legendB.c_str());
        }
        drawList->AddText(
            ImVec2(canvasMin.x + 10.0f, canvasMax.y - 20.0f),
            IM_COL32(150, 175, 210, 195),
            "Wheel: Zoom | Right-Drag: Pan | Double-Click Entity in Hierarchy: Focus"
        );

        drawList->AddRect(canvasMin, canvasMax, PlaybackStateBorderColor(ctx.playbackState), 0.0f, 0, 2.0f);
        drawList->PopClipRect();
    }

    void ViewportPanel::RenderFieldGrid2DCanvas(UIContext& ctx) {
        if (ctx.primaryFieldSnap == nullptr || ctx.activeScene == nullptr || ctx.latestDiagnostics == nullptr) {
            return;
        }
        const auto& snapA = *ctx.primaryFieldSnap;
        if (snapA.width == 0U || snapA.height == 0U || snapA.scalarField.empty()) {
            ImGui::SetCursorPos(ImVec2(12.0f, ImGui::GetCursorPosY() + 8.0f));
            ImGui::TextDisabled("Waiting for 2D GPU FieldGrid2D snapshot...");
            return;
        }

        const auto& diag = *ctx.latestDiagnostics;
        ImGui::SetCursorPosX(8.0f);
        ImGui::TextDisabled(
            "Domain: FieldGrid2D (%ux%u) | %s | Peak=%.4g | CFL=%.3f",
            snapA.width, snapA.height, snapA.fieldLabel.c_str(), snapA.peakAbsVal, diag.effectiveCflNumber
        );

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        const float waveformHeight = 76.0f;
        const float mapHeight = std::max(120.0f, avail.y - waveformHeight - 14.0f);
        const float mapSide = std::min(avail.x - 16.0f, mapHeight);
        if (mapSide <= 32.0f) {
            return;
        }

        const float padLeft = std::max(8.0f, 0.5f * (avail.x - mapSide));
        ImGui::SetCursorPosX(padLeft);
        const ImVec2 mapMin = ImGui::GetCursorScreenPos();
        const ImVec2 mapMax(mapMin.x + mapSide, mapMin.y + mapSide);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->PushClipRect(mapMin, mapMax, true);

        const int heatmapMode = (ctx.fieldHeatmapMode != nullptr) ? *ctx.fieldHeatmapMode : 0;
        const bool hasSnapB =
            ctx.compareOverlayEnabled && ctx.compareFieldSnap != nullptr &&
            ctx.compareFieldSnap->scalarField.size() == snapA.scalarField.size();
        const bool isDiff = (heatmapMode == 2 && hasSnapB);
        const bool showB = (heatmapMode == 1 && hasSnapB);

        double lo = 0.0;
        double hi = 1.0e-12;
        if (isDiff && ctx.abMetrics != nullptr) {
            hi = std::max(1.0e-12, ctx.abMetrics->fieldPeakResidual);
        } else if (!showB && ctx.primaryModelId == scene::ModelId::ThinkerTitorC204) {
            lo = 0.0;
            hi = 2.8;
        } else if (showB && ctx.compareModelId == scene::ModelId::ThinkerTitorC204) {
            lo = 0.0;
            hi = 2.8;
        } else {
            const auto& ref = (showB && ctx.compareFieldSnap != nullptr) ? *ctx.compareFieldSnap : snapA;
            lo = ref.minVal;
            hi = std::max(lo + 1.0e-12, ref.maxVal);
        }

        const std::uint32_t stepX = std::max(1U, snapA.width / 64U);
        const std::uint32_t stepY = std::max(1U, snapA.height / 64U);
        const float cellPxW = mapSide / static_cast<float>((snapA.width + stepX - 1U) / stepX);
        const float cellPxH = mapSide / static_cast<float>((snapA.height + stepY - 1U) / stepY);

        std::uint32_t outY = 0U;
        for (std::uint32_t iy = 0U; iy < snapA.height; iy += stepY, ++outY) {
            std::uint32_t outX = 0U;
            for (std::uint32_t ix = 0U; ix < snapA.width; ix += stepX, ++outX) {
                const std::size_t idx = static_cast<std::size_t>(iy) * snapA.width + ix;
                const double val = isDiff
                    ? std::abs(ctx.compareFieldSnap->scalarField[idx] - snapA.scalarField[idx])
                    : (showB ? ctx.compareFieldSnap->scalarField[idx] : snapA.scalarField[idx]);
                const float norm = static_cast<float>((val - lo) / (hi - lo));
                const ImU32 col = EvaluateHeatmapColor(norm, isDiff);
                const float x0 = mapMin.x + static_cast<float>(outX) * cellPxW;
                const float y0 = mapMax.y - static_cast<float>(outY + 1U) * cellPxH;
                drawList->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + cellPxW + 0.5f, y0 + cellPxH + 0.5f), col);
            }
        }

        const ImVec2 center(mapMin.x + 0.5f * mapSide, mapMin.y + 0.5f * mapSide);
        const float pxPerWorld = (0.5f * mapSide) / static_cast<float>(std::max(1.0e-6, snapA.domainHalfExtent));

        // GE C204 specialized physical overlays (170-deg X-ray cone, 22m x 6m envelope, Lense-Thirring streamlines, VGL pulse).
        if (ctx.primaryModelId == scene::ModelId::ThinkerTitorC204) {
            const auto& tc = ctx.activeScene->physicsProfile.unifiedField.titorC204;
            const float coneLenPx = static_cast<float>(tc.envelopeHalfLengthA * 0.95) * pxPerWorld;
            const float halfConeRad = static_cast<float>(0.5 * tc.xrayVentConeDeg * (3.14159265358979323846 / 180.0));
            const ImVec2 coneApex = center;
            const ImVec2 coneTop(coneApex.x - coneLenPx * std::cos(halfConeRad), coneApex.y - coneLenPx * std::sin(halfConeRad));
            const ImVec2 coneBot(coneApex.x - coneLenPx * std::cos(halfConeRad), coneApex.y + coneLenPx * std::sin(halfConeRad));
            const float xrayDrive = static_cast<float>(std::clamp(diag.c204CabinGForce / 1.85, 0.0, 1.0));
            if (xrayDrive > 0.02f) {
                const int fillAlpha = static_cast<int>(18.0f + 55.0f * xrayDrive);
                drawList->AddTriangleFilled(coneApex, coneTop, coneBot, IM_COL32(255, 70, 60, fillAlpha));
            }
            drawList->AddLine(coneApex, coneTop, IM_COL32(255, 95, 85, 200), 1.6f);
            drawList->AddLine(coneApex, coneBot, IM_COL32(255, 95, 85, 200), 1.6f);
            drawList->AddText(
                ImVec2(mapMin.x + 8.0f, center.y - 8.0f),
                IM_COL32(255, 140, 130, 235),
                (xrayDrive > 0.05f) ? "170 deg X-Ray Vent (@ c)" : "170 deg X-Ray Cone"
            );
            drawList->AddText(
                ImVec2(mapMin.x + 8.0f, mapMin.y + 6.0f),
                IM_COL32(190, 225, 255, 225),
                "Quasi-Stationary ADM Metric (GW & X-Ray Transit @ c ~ 37 ns / 11m)"
            );

            const float envCx = center.x + static_cast<float>(tc.envelopeOffsetXM) * pxPerWorld;
            const float envCy = center.y;
            const float rx = static_cast<float>(tc.envelopeHalfLengthA) * pxPerWorld;
            const float ry = static_cast<float>(tc.envelopeHalfWidthB) * pxPerWorld;
            constexpr int kEllSegs = 64;
            for (int seg = 0; seg < kEllSegs; ++seg) {
                const float th0 = (2.0f * 3.14159265f * static_cast<float>(seg)) / static_cast<float>(kEllSegs);
                const float th1 = (2.0f * 3.14159265f * static_cast<float>(seg + 1)) / static_cast<float>(kEllSegs);
                drawList->AddLine(
                    ImVec2(envCx + rx * std::cos(th0), envCy - ry * std::sin(th0)),
                    ImVec2(envCx + rx * std::cos(th1), envCy - ry * std::sin(th1)),
                    IM_COL32(130, 240, 255, 220), 2.0f
                );
            }
            drawList->AddText(ImVec2(envCx - 64.0f, envCy - ry - 16.0f), IM_COL32(150, 245, 255, 235), "22m x 6m Envelope");

            if (xrayDrive > 0.02f) {
                const float spinDir = diag.c204IsForwardTravel ? -1.0f : 1.0f;
                const float innerAngle = spinDir * static_cast<float>(diag.simulationTime * 2.8);
                const float outerAngle = spinDir * static_cast<float>(diag.simulationTime * 1.15);
                const float innerRx = 0.48f * rx;
                const float innerRy = 0.48f * ry;
                const float outerRx = 0.82f * rx;
                const float outerRy = 0.82f * ry;
                const ImU32 dragColInner = diag.c204IsForwardTravel
                    ? IM_COL32(110, 255, 185, static_cast<int>(195.0f * xrayDrive))
                    : IM_COL32(255, 225, 120, static_cast<int>(215.0f * xrayDrive));
                const ImU32 dragColOuter = diag.c204IsForwardTravel
                    ? IM_COL32(110, 255, 185, static_cast<int>(125.0f * xrayDrive))
                    : IM_COL32(255, 215, 120, static_cast<int>(135.0f * xrayDrive));
                for (int arc = 0; arc < 4; ++arc) {
                    const float offset = static_cast<float>(arc) * (2.0f * 3.14159265f / 4.0f);
                    const float a0 = innerAngle + offset;
                    const float b0 = outerAngle + offset + 0.4f;
                    constexpr float kArcSpan = 0.72f;
                    constexpr int kArcSteps = 10;
                    for (int s = 0; s < kArcSteps; ++s) {
                        const float f0 = static_cast<float>(s) / static_cast<float>(kArcSteps);
                        const float f1 = static_cast<float>(s + 1) / static_cast<float>(kArcSteps);
                        const float ta0 = a0 + kArcSpan * f0;
                        const float ta1 = a0 + kArcSpan * f1;
                        drawList->AddLine(
                            ImVec2(envCx + innerRx * std::cos(ta0), envCy - innerRy * std::sin(ta0)),
                            ImVec2(envCx + innerRx * std::cos(ta1), envCy - innerRy * std::sin(ta1)),
                            dragColInner, 1.8f
                        );
                        const float tb0 = b0 + kArcSpan * f0;
                        const float tb1 = b0 + kArcSpan * f1;
                        drawList->AddLine(
                            ImVec2(envCx + outerRx * std::cos(tb0), envCy - outerRy * std::sin(tb0)),
                            ImVec2(envCx + outerRx * std::cos(tb1), envCy - outerRy * std::sin(tb1)),
                            dragColOuter, 1.3f
                        );
                    }
                }
            }

            if (diag.c204OperationalState.find("VGL") != std::string::npos) {
                const float pulseFrac = static_cast<float>(std::fmod(diag.simulationTime * 0.65, 1.0));
                const float vglRx = (0.12f + 0.92f * pulseFrac) * rx;
                const float vglRy = (0.12f + 0.92f * pulseFrac) * ry;
                const int vglAlpha = static_cast<int>((1.0f - pulseFrac) * 230.0f);
                constexpr int kVglSegs = 48;
                for (int seg = 0; seg < kVglSegs; ++seg) {
                    const float th0 = (2.0f * 3.14159265f * static_cast<float>(seg)) / static_cast<float>(kVglSegs);
                    const float th1 = (2.0f * 3.14159265f * static_cast<float>(seg + 1)) / static_cast<float>(kVglSegs);
                    drawList->AddLine(
                        ImVec2(envCx + vglRx * std::cos(th0), envCy - vglRy * std::sin(th0)),
                        ImVec2(envCx + vglRx * std::cos(th1), envCy - vglRy * std::sin(th1)),
                        IM_COL32(110, 235, 255, vglAlpha), 2.0f
                    );
                }
            }
            drawList->AddText(
                ImVec2(mapMin.x + 8.0f, mapMax.y - 18.0f),
                IM_COL32(200, 220, 245, 215),
                "Fixed Scale |1+g_00|: 0.0 Minkowski | 1.0 Null Horizon | >1.0 Neg CTC"
            );
        }

        // Overlay stationary device anchors.
        for (const auto& anchor : ctx.activeScene->bodies) {
            const ImVec2 p(
                center.x + static_cast<float>(anchor.position.x) * pxPerWorld,
                center.y - static_cast<float>(anchor.position.y) * pxPerWorld
            );
            const float rPx = static_cast<float>(anchor.radius) * pxPerWorld;
            if (anchor.category == "PhryllCrystal") {
                const float hw = rPx * 0.65f;
                drawList->AddQuad(
                    ImVec2(p.x, p.y - rPx * 1.15f),
                    ImVec2(p.x + hw, p.y),
                    ImVec2(p.x, p.y + rPx * 0.85f),
                    ImVec2(p.x - hw, p.y),
                    IM_COL32(220, 250, 255, 220), 2.0f
                );
            } else if (anchor.category == "HEEMFG_Cavity") {
                drawList->AddCircle(p, rPx, IM_COL32(220, 250, 255, 210), 64, 2.0f);
            } else if (anchor.category == "C204_Envelope") {
                continue;
            } else if (anchor.category == "C204_Singularity") {
                drawList->AddCircleFilled(p, std::max(4.0f, rPx), IM_COL32(255, 245, 140, 245));
                drawList->AddCircle(p, std::max(6.0f, rPx + 2.0f), IM_COL32(255, 160, 70, 230), 24, 1.5f);
            } else if (anchor.category == "C204_Cabin") {
                const float hw = std::max(8.0f, rPx * 1.4f);
                const float hh = std::max(5.0f, rPx * 0.9f);
                drawList->AddRect(ImVec2(p.x - hw, p.y - hh), ImVec2(p.x + hw, p.y + hh), IM_COL32(115, 255, 160, 235), 2.0f, 0, 2.0f);
                drawList->AddText(ImVec2(p.x + hw + 4.0f, p.y - 7.0f), IM_COL32(150, 255, 185, 235), "Cabin (1.8m)");
            } else {
                drawList->AddCircle(p, std::max(5.0f, rPx), IM_COL32(255, 230, 120, 220), 24, 1.8f);
            }
        }

        drawList->AddRect(mapMin, mapMax, PlaybackStateBorderColor(ctx.playbackState), 0.0f, 0, 2.0f);
        drawList->PopClipRect();

        ImGui::Dummy(ImVec2(mapSide, mapSide + 4.0f));

        if (ctx.probeWaveformA != nullptr && !ctx.probeWaveformA->empty()) {
            ImGui::SetCursorPosX(8.0f);
            std::vector<float> waveA(ctx.probeWaveformA->begin(), ctx.probeWaveformA->end());
            const char* waveLabel = (ctx.primaryModelId == scene::ModelId::ThinkerTitorC204)
                ? "Cabin Metric g_00(t) [A]"
                : "Center Probe Ez(t) [A]";
            ImGui::PlotLines(
                waveLabel,
                waveA.data(),
                static_cast<int>(waveA.size()),
                0,
                nullptr,
                FLT_MAX,
                FLT_MAX,
                ImVec2(std::max(120.0f, avail.x * 0.72f), 56.0f)
            );
        }
    }

    void ViewportPanel::RenderRadialScaleCanvas(UIContext& ctx) {
        if (ctx.primaryRadialSnap == nullptr || ctx.activeScene == nullptr) {
            return;
        }
        const auto& snapA = *ctx.primaryRadialSnap;
        if (snapA.xValues.size() < 2U) {
            ImGui::SetCursorPos(ImVec2(12.0f, ImGui::GetCursorPosY() + 8.0f));
            ImGui::TextDisabled("No RadialAndScaleProfile data available.");
            return;
        }

        ImGui::SetCursorPosX(8.0f);
        ImGui::TextDisabled(
            "Domain: RadialAndScaleProfile (%zu samples) | X: %s | Y: %s",
            snapA.xValues.size(), snapA.xLabel.c_str(), snapA.primaryYLabel.c_str()
        );

        const ImVec2 canvasMin = ImGui::GetCursorScreenPos();
        const ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        if (canvasSize.x <= 60.0f || canvasSize.y <= 60.0f) {
            return;
        }

        const ImVec2 canvasMax(canvasMin.x + canvasSize.x, canvasMin.y + canvasSize.y);
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->PushClipRect(canvasMin, canvasMax, true);

        drawList->AddRectFilled(canvasMin, canvasMax, IM_COL32(10, 14, 24, 255));
        drawList->AddRect(canvasMin, canvasMax, IM_COL32(55, 80, 120, 255));

        const float padL = 52.0f;
        const float padR = 20.0f;
        const float padT = 38.0f;
        const float padB = 34.0f;
        const ImVec2 pMin(canvasMin.x + padL, canvasMin.y + padT);
        const ImVec2 pMax(canvasMax.x - padR, canvasMax.y - padB);
        const float plotW = std::max(10.0f, pMax.x - pMin.x);
        const float plotH = std::max(10.0f, pMax.y - pMin.y);

        auto mapX = [&](double x) -> double {
            return snapA.logarithmicX ? std::log10(std::max(x, 1.0e-300)) : x;
        };
        auto mapY = [&](double y) -> double {
            return snapA.logarithmicY ? std::log10(std::max(y, 1.0e-300)) : y;
        };

        double xMin = mapX(snapA.xValues.front());
        double xMax = mapX(snapA.xValues.back());
        if (xMax <= xMin) { xMax = xMin + 1.0; }

        double yMin = 1.0e300;
        double yMax = -1.0e300;
        auto includeVec = [&](const std::vector<double>& v) {
            for (double val : v) {
                if (!std::isfinite(val)) { continue; }
                const double mv = mapY(val);
                if (std::isfinite(mv)) {
                    yMin = std::min(yMin, mv);
                    yMax = std::max(yMax, mv);
                }
            }
        };
        includeVec(snapA.primaryYValues);
        includeVec(snapA.secondaryYValues);
        if (ctx.compareOverlayEnabled && ctx.compareRadialSnap != nullptr) {
            includeVec(ctx.compareRadialSnap->primaryYValues);
            includeVec(ctx.compareRadialSnap->secondaryYValues);
        }
        if (yMax <= yMin) { yMin = 0.0; yMax = 1.0; }
        const double yPad = 0.06 * (yMax - yMin);
        yMin -= yPad;
        yMax += yPad;

        auto toScreen = [&](double x, double y) -> ImVec2 {
            const float u = static_cast<float>((mapX(x) - xMin) / (xMax - xMin));
            const float v = static_cast<float>((mapY(y) - yMin) / (yMax - yMin));
            return ImVec2(pMin.x + u * plotW, pMax.y - v * plotH);
        };

        for (int g = 0; g <= 4; ++g) {
            const float frac = static_cast<float>(g) / 4.0f;
            const float gx = pMin.x + frac * plotW;
            const float gy = pMax.y - frac * plotH;
            drawList->AddLine(ImVec2(gx, pMin.y), ImVec2(gx, pMax.y), IM_COL32(36, 50, 74, 150));
            drawList->AddLine(ImVec2(pMin.x, gy), ImVec2(pMax.x, gy), IM_COL32(36, 50, 74, 150));
        }

        auto drawLandmark = [&](double xVal, const char* label, ImU32 col) {
            const double mx = mapX(xVal);
            if (mx >= xMin && mx <= xMax) {
                const float sx = pMin.x + static_cast<float>((mx - xMin) / (xMax - xMin)) * plotW;
                drawList->AddLine(ImVec2(sx, pMin.y), ImVec2(sx, pMax.y), col, 1.2f);
                drawList->AddText(ImVec2(sx + 3.0f, pMin.y + 4.0f), col, label);
            }
        };
        if (ctx.activeScene->physicsProfile.radialScale.sweepKind == "HolographicMassScale") {
            drawLandmark(0.84184e-15, "Proton r_p", IM_COL32(150, 255, 175, 190));
            drawLandmark(2.5e4, "Cygnus X-1", IM_COL32(255, 220, 120, 190));
        } else if (ctx.activeScene->physicsProfile.radialScale.sweepKind == "C204MetricProfile") {
            const auto& tc = ctx.activeScene->physicsProfile.unifiedField.titorC204;
            drawLandmark(tc.cabinProbeDistanceM, "Cabin 1.8m", IM_COL32(150, 255, 175, 210));
            drawLandmark(tc.massOffsetM, "Mass Offset 5m", IM_COL32(255, 215, 120, 210));
            drawLandmark(tc.envelopeHalfLengthA + tc.envelopeOffsetXM, "Fwd Boundary", IM_COL32(130, 240, 255, 210));
        } else {
            const double Rs = ctx.activeScene->physicsProfile.radialScale.referenceSchwarzschildRadius;
            drawLandmark(Rs, "GR Rs", IM_COL32(255, 130, 130, 190));
            drawLandmark(1.5 * Rs, "R=1.5 Rs", IM_COL32(255, 215, 120, 190));
            drawLandmark(2.0 * Rs, "Essonne 2 Rs", IM_COL32(130, 240, 255, 190));
        }

        auto drawCurve = [&](const std::vector<double>& xs, const std::vector<double>& ys, ImU32 col, bool dashed, float thick) {
            const std::size_t n = std::min(xs.size(), ys.size());
            for (std::size_t k = 1U; k < n; k += (dashed ? 2U : 1U)) {
                drawList->AddLine(toScreen(xs[k - 1U], ys[k - 1U]), toScreen(xs[k], ys[k]), col, thick);
            }
        };

        drawCurve(snapA.xValues, snapA.primaryYValues, IM_COL32(95, 215, 255, 255), false, 2.2f);
        drawCurve(snapA.xValues, snapA.secondaryYValues, IM_COL32(255, 210, 90, 240), false, 1.7f);
        if (ctx.compareOverlayEnabled && ctx.compareRadialSnap != nullptr && !ctx.compareRadialSnap->xValues.empty()) {
            drawCurve(ctx.compareRadialSnap->xValues, ctx.compareRadialSnap->primaryYValues, IM_COL32(255, 120, 145, 240), true, 2.0f);
            drawCurve(ctx.compareRadialSnap->xValues, ctx.compareRadialSnap->secondaryYValues, IM_COL32(180, 145, 255, 220), true, 1.5f);
        }

        drawList->AddRect(pMin, pMax, IM_COL32(95, 125, 175, 255));
        const std::string legA = std::string("A (Solid): ") + snapA.primaryYLabel + " / " + snapA.secondaryYLabel;
        drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 6.0f), IM_COL32(130, 225, 255, 245), legA.c_str());
        if (ctx.compareOverlayEnabled && ctx.compareRadialSnap != nullptr && !ctx.compareRadialSnap->primaryYLabel.empty()) {
            const std::string legB = std::string("B (Dashed): ") + ctx.compareRadialSnap->primaryYLabel + " / " + ctx.compareRadialSnap->secondaryYLabel;
            drawList->AddText(ImVec2(canvasMin.x + 10.0f, canvasMin.y + 20.0f), IM_COL32(255, 150, 170, 245), legB.c_str());
        }
        drawList->PopClipRect();
    }

    void ViewportPanel::OnUIRender(UIContext& ctx) {
        if (!m_isOpen) {
            return;
        }

        // Legacy ViewportPanel signature zero-padding canvas framing.
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        if (ImGui::Begin(m_name.c_str(), &m_isOpen, m_flags)) {
            m_focused = ImGui::IsWindowFocused();
            m_hovered = ImGui::IsWindowHovered();
            m_viewportSize = ImGui::GetContentRegionAvail();

            RenderViewportToolbar(ctx);

            const bool showLattice =
                (ctx.showFlowerOfLifeLatticeView != nullptr && *ctx.showFlowerOfLifeLatticeView &&
                 ctx.flowerMultiverse != nullptr && !ctx.flowerMultiverse->IsEmpty());

            if (showLattice) {
                RenderFlowerOfLifeCanvas(ctx);
            } else if (!ctx.hasActiveScene || ctx.activeScene == nullptr) {
                ImGui::SetCursorPos(ImVec2(14.0f, ImGui::GetCursorPosY() + 12.0f));
                ImGui::TextDisabled("No active scene loaded. Choose a Scenario or Bubble Universe from Hierarchy, or File -> Open Scene...");
            } else if (ctx.activeScene->domain == scene::SimulationDomain::FieldGrid2D) {
                RenderFieldGrid2DCanvas(ctx);
            } else if (ctx.activeScene->domain == scene::SimulationDomain::RadialAndScaleProfile) {
                RenderRadialScaleCanvas(ctx);
            } else {
                RenderOrbitalCanvas(ctx);
            }
        }
        ImGui::End();
        ImGui::PopStyleVar();
    }

} // namespace umgebung::ui
