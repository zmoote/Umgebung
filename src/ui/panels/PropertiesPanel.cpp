/**
 * @file PropertiesPanel.cpp
 * @brief Implementation of the right-docked `"Properties"` inspector panel
 *        combining legacy Umgebung's `CollapsingHeader` entity inspector with
 *        current Umgebung's domain-filtered Model Registry, live Thinker/Device
 *        controls, playback pacing, and Simultaneous A/B telemetry.
 *
 * @note Compiled exclusively when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#include "umgebung/ui/panels/PropertiesPanel.hpp"
#include "umgebung/core/Application.hpp"
#include "umgebung/models/ModelRegistry.hpp"
#include "umgebung/ui/UIContext.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#include <imgui.h>

namespace umgebung::ui {

    namespace {

        /**
         * @brief Maps a vibrational density scalar (`1.0 .. 13.0`) to the exact
         *        resonance plane descriptor used in `Umgebung-legacy`'s `PropertiesPanel`.
         */
        [[nodiscard]] const char* VibrationalDensityPlaneLabel(double density) noexcept {
            if (density < 2.0)  { return "1D-2D: Microscopic / Physical Foundation"; }
            if (density < 4.0)  { return "3D: Physical Plane (Earth Standard)"; }
            if (density < 5.0)  { return "4D: Astral / Time-Fluid Matrix"; }
            if (density < 9.0)  { return "5D-8D: Crystalline Light (Nataru Shift)"; }
            if (density < 12.0) { return "9D-11D: Galactic Seeders / Oona / The Nine"; }
            if (density < 13.0) { return "12D: Prime Creator Transition"; }
            return "13D: Source Singularity";
        }

    } // anonymous namespace

    PropertiesPanel::PropertiesPanel()
        : Panel("Properties", true) {}

    void PropertiesPanel::RenderEntityInspector(UIContext& ctx, std::size_t bodyIndex) {
        if (ctx.activeScene == nullptr || bodyIndex >= ctx.activeScene->bodies.size()) {
            ImGui::TextDisabled("No entity selected.");
            return;
        }

        auto& sceneBody = ctx.activeScene->bodies[bodyIndex];
        const scene::BodyEntity* liveBody =
            (ctx.liveBodies != nullptr && bodyIndex < ctx.liveBodies->size())
                ? &(*ctx.liveBodies)[bodyIndex]
                : &sceneBody;

        bool mutated = false;

        // ---------------------------------------------------------------------
        // 1. Name & Identity Header (Legacy Style)
        // ---------------------------------------------------------------------
        if (ImGui::CollapsingHeader("Name", ImGuiTreeNodeFlags_DefaultOpen)) {
            std::array<char, 256> nameBuf{};
            std::snprintf(nameBuf.data(), nameBuf.size(), "%s", sceneBody.name.c_str());
            if (ImGui::InputText("Entity Name", nameBuf.data(), nameBuf.size())) {
                sceneBody.name = std::string(nameBuf.data());
                mutated = true;
            }

            std::array<char, 128> catBuf{};
            std::snprintf(catBuf.data(), catBuf.size(), "%s", sceneBody.category.c_str());
            if (ImGui::InputText("Category", catBuf.data(), catBuf.size())) {
                sceneBody.category = std::string(catBuf.data());
                mutated = true;
            }

            ImGui::TextDisabled("ID: %s", sceneBody.id.c_str());
            if (!sceneBody.parentId.empty()) {
                ImGui::TextDisabled("Parent ID: %s", sceneBody.parentId.c_str());
            }
        }

        // ---------------------------------------------------------------------
        // 2. Transform & Kinematics Header (Legacy Style)
        // ---------------------------------------------------------------------
        if (ImGui::CollapsingHeader("Transform & Kinematics", ImGuiTreeNodeFlags_DefaultOpen)) {
            float pos[3] = {
                static_cast<float>(sceneBody.position.x),
                static_cast<float>(sceneBody.position.y),
                static_cast<float>(sceneBody.position.z)
            };
            if (ImGui::DragFloat3("Initial Pos", pos, 0.05f, 0.0f, 0.0f, "%.4f")) {
                sceneBody.position = {static_cast<double>(pos[0]), static_cast<double>(pos[1]), static_cast<double>(pos[2])};
                mutated = true;
            }

            if (ctx.activeScene->domain == scene::SimulationDomain::NBodyDynamics) {
                float vel[3] = {
                    static_cast<float>(sceneBody.velocity.x),
                    static_cast<float>(sceneBody.velocity.y),
                    static_cast<float>(sceneBody.velocity.z)
                };
                if (ImGui::DragFloat3("Initial Vel", vel, 0.02f, 0.0f, 0.0f, "%.4f")) {
                    sceneBody.velocity = {static_cast<double>(vel[0]), static_cast<double>(vel[1]), static_cast<double>(vel[2])};
                    mutated = true;
                }
            }

            float rad = static_cast<float>(sceneBody.radius);
            if (ImGui::DragFloat("Radius", &rad, 0.01f, 0.001f, 100.0f, "%.4f")) {
                sceneBody.radius = std::max(0.001, static_cast<double>(rad));
                mutated = true;
            }

            if (liveBody != nullptr) {
                ImGui::Separator();
                ImGui::TextDisabled(
                    "Live Pos : (%.4f, %.4f, %.4f)",
                    liveBody->position.x, liveBody->position.y, liveBody->position.z
                );
                if (ctx.activeScene->domain == scene::SimulationDomain::NBodyDynamics) {
                    ImGui::TextDisabled(
                        "Live Vel : (%.4f, %.4f, %.4f) | |v|=%.4f",
                        liveBody->velocity.x, liveBody->velocity.y, liveBody->velocity.z,
                        liveBody->velocity.Norm()
                    );
                }
            }
        }

        // ---------------------------------------------------------------------
        // 3. Scale Tier Header (Legacy ScaleComponent Combo)
        // ---------------------------------------------------------------------
        if (ImGui::CollapsingHeader("Scale", ImGuiTreeNodeFlags_DefaultOpen)) {
            static constexpr const char* kScaleNames[] = {
                "Multiverse", "BubbleUniverse", "Galaxy", "StarSystem",
                "Planet", "EntityOrDevice", "Atom", "SubatomicPSU"
            };
            int curScale = static_cast<int>(sceneBody.scaleTier);
            if (ImGui::Combo("Scale Tier", &curScale, kScaleNames, IM_ARRAYSIZE(kScaleNames))) {
                sceneBody.scaleTier = static_cast<scene::ScaleTier>(curScale);
                mutated = true;
            }
        }

        // ---------------------------------------------------------------------
        // 4. Physical & Relativistic / Metaphysical Properties (Legacy Style)
        // ---------------------------------------------------------------------
        if (ImGui::CollapsingHeader("Physical Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Intrinsic Properties");
            ImGui::Separator();
            double massVal = sceneBody.mass;
            if (ImGui::InputScalar("Rest Mass", ImGuiDataType_Double, &massVal, nullptr, nullptr, "%.6g")) {
                if (std::isfinite(massVal) && massVal >= 0.0) {
                    sceneBody.mass = massVal;
                    mutated = true;
                }
            }

            if (ctx.primaryProperTimes != nullptr && bodyIndex < ctx.primaryProperTimes->size()) {
                ImGui::TextDisabled("Proper Time (A): %.6f", (*ctx.primaryProperTimes)[bodyIndex]);
            }
            if (ctx.compareOverlayEnabled &&
                ctx.compareProperTimes != nullptr &&
                bodyIndex < ctx.compareProperTimes->size()) {
                ImGui::TextDisabled("Proper Time (B): %.6f", (*ctx.compareProperTimes)[bodyIndex]);
            }

            if (ctx.abMetrics != nullptr && ctx.abMetrics->active && bodyIndex < ctx.abMetrics->perBody.size()) {
                const auto& rec = ctx.abMetrics->perBody[bodyIndex];
                ImGui::TextColored(
                    ImVec4(1.0f, 0.82f, 0.40f, 1.0f),
                    "A/B Delta: dR=%.3e | dV=%.3e | dTau=%.3e",
                    rec.deltaR, rec.deltaV, rec.deltaProperTime
                );
            }

            ImGui::Spacing();
            ImGui::Text("Relativistic & Metaphysical State");
            ImGui::Separator();

            float densityF = static_cast<float>(sceneBody.vibrationalDensity);
            if (ImGui::SliderFloat("Vibrational Density", &densityF, 1.0f, 13.0f, "%.2f D")) {
                sceneBody.vibrationalDensity = static_cast<double>(densityF);
                mutated = true;
            }
            ImGui::TextColored(
                ImVec4(0.42f, 0.84f, 1.0f, 1.0f),
                "Resonance: %s",
                VibrationalDensityPlaneLabel(sceneBody.vibrationalDensity)
            );

            float coherenceF = static_cast<float>(sceneBody.consciousnessCoherence);
            if (ImGui::SliderFloat("Consciousness Coherence", &coherenceF, 0.0f, 10.0f, "%.2f")) {
                sceneBody.consciousnessCoherence = static_cast<double>(coherenceF);
                mutated = true;
            }

            double phryllVal = sceneBody.phryllCharge;
            if (ImGui::InputScalar("Phryll Charge", ImGuiDataType_Double, &phryllVal, nullptr, nullptr, "%.4g")) {
                if (std::isfinite(phryllVal)) {
                    sceneBody.phryllCharge = phryllVal;
                    mutated = true;
                }
            }
        }

        if (mutated && ctx.onSceneParametersChanged) {
            ctx.onSceneParametersChanged();
        }

        // ---------------------------------------------------------------------
        // 5. Entity Actions (Focus & Delete — Legacy PropertiesPanel Footer)
        // ---------------------------------------------------------------------
        ImGui::Separator();
        if (ImGui::Button("Focus in Viewport", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
            if (ctx.onFocusBodyEntity) {
                ctx.onFocusBodyEntity(bodyIndex);
            }
        }
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.55f, 0.16f, 0.18f, 0.90f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.72f, 0.22f, 0.24f, 1.00f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.84f, 0.26f, 0.28f, 1.00f));
        if (ImGui::Button("Delete Entity", ImVec2(ImGui::GetContentRegionAvail().x, 0.0f))) {
            if (ctx.onDeleteBodyEntity) {
                ctx.onDeleteBodyEntity(bodyIndex);
            }
        }
        ImGui::PopStyleColor(3);
    }

    void PropertiesPanel::RenderSimulationInspector(UIContext& ctx) {
        if (!ctx.hasActiveScene || ctx.activeScene == nullptr || !ctx.hasActiveSolver) {
            ImGui::TextDisabled("Select a Scenario or Bubble Universe in Hierarchy.");
            return;
        }

        auto& activeScene = *ctx.activeScene;
        const scene::SimulationDomain domain = activeScene.domain;

        // ---------------------------------------------------------------------
        // 1. Scene & Physics Model Registry Header
        // ---------------------------------------------------------------------
        if (ImGui::CollapsingHeader("Physics Model & A/B Comparison", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextWrapped("Scene  : %s", activeScene.name.c_str());
            ImGui::TextColored(
                ImVec4(0.52f, 0.90f, 0.70f, 1.0f),
                "Domain : %s",
                scene::SimulationDomainToString(domain)
            );
            ImGui::TextDisabled("Scope  : %s", scene::SimulationScopeToString(ctx.activeScope));
            ImGui::Separator();

            const auto compatibleModels = models::ModelRegistry::GetModelsForDomain(domain);
            const auto* curDescA = models::ModelRegistry::FindById(ctx.primaryModelId);

            if (ImGui::BeginCombo("Primary Model A", curDescA ? curDescA->displayName : "Unknown")) {
                for (const auto& desc : compatibleModels) {
                    const bool selected = (desc.id == ctx.primaryModelId);
                    const std::string itemLabel =
                        std::string("[") + models::ModelProvenanceToString(desc.provenance) + "] " + desc.displayName;
                    if (ImGui::Selectable(itemLabel.c_str(), selected) && desc.id != ctx.primaryModelId) {
                        if (ctx.onSelectPrimaryModel) {
                            ctx.onSelectPrimaryModel(desc.id);
                        }
                    }
                    if (selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            if (curDescA != nullptr) {
                ImVec4 badgeCol(0.55f, 0.90f, 0.65f, 1.0f);
                if (curDescA->provenance == models::ModelProvenance::VerbatimEquation) {
                    badgeCol = ImVec4(0.45f, 0.85f, 1.0f, 1.0f);
                } else if (curDescA->provenance == models::ModelProvenance::HybridVerbatimAndAnsatz) {
                    badgeCol = ImVec4(1.0f, 0.80f, 0.35f, 1.0f);
                } else if (curDescA->provenance == models::ModelProvenance::OperationalizedSpec) {
                    badgeCol = ImVec4(0.85f, 0.60f, 1.0f, 1.0f);
                }
                ImGui::TextColored(badgeCol, "Provenance [A]: [%s]", models::ModelProvenanceToString(curDescA->provenance));
                ImGui::TextWrapped("%s", curDescA->provenanceDetails);
            }

            bool overlayToggled = ctx.compareOverlayEnabled;
            if (ImGui::Checkbox("Simultaneous Overlay (Model B)", &overlayToggled)) {
                if (ctx.onSetCompareOverlay) {
                    ctx.onSetCompareOverlay(overlayToggled, ctx.compareModelId);
                }
            }

            if (ctx.compareOverlayEnabled) {
                const auto* curDescB = models::ModelRegistry::FindById(ctx.compareModelId);
                if (ImGui::BeginCombo("Overlay Model B", curDescB ? curDescB->displayName : "Unknown")) {
                    for (const auto& desc : compatibleModels) {
                        const bool selected = (desc.id == ctx.compareModelId);
                        const std::string itemLabel =
                            std::string("[") + models::ModelProvenanceToString(desc.provenance) + "] " + desc.displayName;
                        if (ImGui::Selectable(itemLabel.c_str(), selected) && desc.id != ctx.compareModelId) {
                            if (ctx.onSetCompareOverlay) {
                                ctx.onSetCompareOverlay(true, desc.id);
                            }
                        }
                        if (selected) {
                            ImGui::SetItemDefaultFocus();
                        }
                    }
                    ImGui::EndCombo();
                }
                if (curDescB != nullptr) {
                    ImGui::TextDisabled("Provenance [B]: [%s]", models::ModelProvenanceToString(curDescB->provenance));
                }
            }
        }

        // ---------------------------------------------------------------------
        // 2. Domain / Thinker Interactive Controls (When Applicable)
        // ---------------------------------------------------------------------
        const bool hasSpecialControls =
            (domain == scene::SimulationDomain::FieldGrid2D &&
             (ctx.primaryModelId == scene::ModelId::ThinkerVogelWillis ||
              ctx.primaryModelId == scene::ModelId::ThinkerPais ||
              ctx.primaryModelId == scene::ModelId::ThinkerTitorC204)) ||
            (domain == scene::SimulationDomain::NBodyDynamics &&
             ctx.primaryModelId == scene::ModelId::ThinkerDanaan);

        if (hasSpecialControls && ImGui::CollapsingHeader("Device & Ansatz Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (domain == scene::SimulationDomain::FieldGrid2D &&
                ctx.primaryModelId == scene::ModelId::ThinkerVogelWillis) {
                float driveV = static_cast<float>(activeScene.physicsProfile.unifiedField.vogelWillis.driveVoltageV);
                if (ImGui::SliderFloat("Coil Drive (V)", &driveV, 5.0f, 80.0f, "%.1f V")) {
                    activeScene.physicsProfile.unifiedField.vogelWillis.driveVoltageV = static_cast<double>(driveV);
                    if (ctx.onSceneParametersChanged) {
                        ctx.onSceneParametersChanged();
                    }
                }
                const bool inSafe = (driveV >= 20.0f && driveV <= 60.0f);
                ImGui::TextColored(
                    inSafe ? ImVec4(0.45f, 0.90f, 0.60f, 1.0f) : ImVec4(1.0f, 0.40f, 0.35f, 1.0f),
                    "PhryllGen Spec [20-60 V]: %s",
                    inSafe ? "SAFE" : "OUT OF SPEC!"
                );
            } else if (domain == scene::SimulationDomain::FieldGrid2D &&
                       ctx.primaryModelId == scene::ModelId::ThinkerPais) {
                const double curKappa = activeScene.physicsProfile.unifiedField.pais.inertialReductionCoupling;
                float logKappa = (curKappa > 0.0) ? static_cast<float>(std::log10(curKappa)) : -12.0f;
                if (ImGui::SliderFloat("Pais log10(kappa)", &logKappa, -13.0f, -8.0f, "10^%.2f")) {
                    activeScene.physicsProfile.unifiedField.pais.inertialReductionCoupling =
                        std::pow(10.0, static_cast<double>(logKappa));
                    if (ctx.onSceneParametersChanged) {
                        ctx.onSceneParametersChanged();
                    }
                }
            } else if (domain == scene::SimulationDomain::FieldGrid2D &&
                       ctx.primaryModelId == scene::ModelId::ThinkerTitorC204) {
                ImGui::TextColored(ImVec4(0.50f, 0.90f, 1.0f, 1.0f), "GE C204 Pilot Controls (TMIM-200S/4005)");
                auto& tc = activeScene.physicsProfile.unifiedField.titorC204;
                bool c204Changed = false;
                if (ImGui::SmallButton("50% (3.4 yr/hr)"))  { tc.powerLevelPercent = 50.0;  tc.injectedChargeCoulombs = 0.0; c204Changed = true; }
                ImGui::SameLine();
                if (ImGui::SmallButton("75% (6.2 yr/hr)"))  { tc.powerLevelPercent = 75.0;  tc.injectedChargeCoulombs = 0.0; c204Changed = true; }
                ImGui::SameLine();
                if (ImGui::SmallButton("100% (10 yr/hr)"))  { tc.powerLevelPercent = 100.0; tc.injectedChargeCoulombs = 0.0; c204Changed = true; }

                float pPct = static_cast<float>(tc.powerLevelPercent);
                if (ImGui::SliderFloat("Matrix Power", &pPct, 10.0f, 100.0f, "%.0f%%")) {
                    tc.powerLevelPercent = static_cast<double>(pPct);
                    tc.injectedChargeCoulombs = 0.0;
                    c204Changed = true;
                }
                float targetYr = static_cast<float>(tc.targetDisplacementYears);
                if (ImGui::SliderFloat("Target Jump", &targetYr, -100.0f, 100.0f, "%+.1f yrs")) {
                    tc.targetDisplacementYears = static_cast<double>(targetYr);
                    c204Changed = true;
                }
                float payloadKg = static_cast<float>(tc.payloadMassKg);
                if (ImGui::SliderFloat("Payload Mass", &payloadKg, 225.0f, 2800.0f, "%.0f kg")) {
                    tc.payloadMassKg = static_cast<double>(payloadKg);
                    c204Changed = true;
                }
                float vglFault = static_cast<float>(tc.simulatedVglAnomaly);
                if (ImGui::SliderFloat("VGL Anomaly", &vglFault, 0.0f, 0.30f, "%.3f")) {
                    tc.simulatedVglAnomaly = static_cast<double>(vglFault);
                    c204Changed = true;
                }
                if (c204Changed && ctx.onSceneParametersChanged) {
                    ctx.onSceneParametersChanged();
                }
            } else if (domain == scene::SimulationDomain::NBodyDynamics &&
                       ctx.primaryModelId == scene::ModelId::ThinkerDanaan) {
                float loveK = static_cast<float>(activeScene.physicsProfile.unifiedField.loveBindingCoupling);
                if (ImGui::SliderFloat("Love Ansatz (kappa)", &loveK, 0.0f, 0.01f, "%.5f")) {
                    activeScene.physicsProfile.unifiedField.loveBindingCoupling = static_cast<double>(loveK);
                    if (ctx.onSceneParametersChanged) {
                        ctx.onSceneParametersChanged();
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // 3. Playback & Wall-Clock Pacing Controls
        // ---------------------------------------------------------------------
        if (domain != scene::SimulationDomain::RadialAndScaleProfile &&
            ImGui::CollapsingHeader("Playback & Stepping", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Play")) {
                if (ctx.onPlay) { ctx.onPlay(); }
            }
            ImGui::SameLine();
            if (ImGui::Button("Pause")) {
                if (ctx.onPause) { ctx.onPause(); }
            }
            ImGui::SameLine();
            if (ImGui::Button("Stop")) {
                if (ctx.onStop) { ctx.onStop(); }
            }
            ImGui::SameLine();
            if (ImGui::Button("Step")) {
                if (ctx.onSingleStep) { ctx.onSingleStep(); }
            }
            ImGui::SameLine();
            if (ImGui::Button("Reset")) {
                if (ctx.onResetSimulation) { ctx.onResetSimulation(); }
            }

            if (ctx.wallClockPaced != nullptr && ctx.simTimeScale != nullptr) {
                if (ImGui::Checkbox("Wall-Clock Paced (FPS-Independent)", ctx.wallClockPaced)) {
                    if (ctx.simTimeAccumulator != nullptr) {
                        *ctx.simTimeAccumulator = 0.0;
                    }
                }
                if (*ctx.wallClockPaced) {
                    ImGui::SliderFloat(
                        "Time Scale",
                        ctx.simTimeScale,
                        0.05f,
                        600.0f,
                        "%.2fx",
                        ImGuiSliderFlags_Logarithmic
                    );
                    if (ImGui::SmallButton("0.25x")) { *ctx.simTimeScale = 0.25f; }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("1x"))    { *ctx.simTimeScale = 1.0f; }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("10x"))   { *ctx.simTimeScale = 10.0f; }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("60x"))   { *ctx.simTimeScale = 60.0f; }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("300x"))  { *ctx.simTimeScale = 300.0f; }

                    const double dtVal = (ctx.activeTimestep != nullptr) ? *ctx.activeTimestep : 0.0;
                    ImGui::TextDisabled(
                        "dt=%.4g | Sub-Steps/Frame=%d%s",
                        dtVal,
                        ctx.lastFrameExecutedSteps,
                        (ctx.lastFrameExecutedSteps >= ctx.maxSubStepsPerFrame) ? " [CAPPED]" : ""
                    );
                } else if (ctx.stepsPerFrame != nullptr) {
                    ImGui::SliderInt("Steps / Frame", ctx.stepsPerFrame, 1, 60);
                }
            }
        }

        // ---------------------------------------------------------------------
        // 4. Domain-Adaptive Live Telemetry
        // ---------------------------------------------------------------------
        if (ctx.latestDiagnostics != nullptr &&
            ImGui::CollapsingHeader("Live Telemetry", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto& diag = *ctx.latestDiagnostics;
            if (domain == scene::SimulationDomain::NBodyDynamics) {
                ImGui::Text("Step : %llu | t = %.5f", static_cast<unsigned long long>(diag.stepCount), diag.simulationTime);
                ImGui::Text("E    : %.8f (drift=%.2e)", diag.totalEnergy, diag.relativeEnergyDrift);
                ImGui::Text("|P|  : %.2e | |L| = %.8f", diag.totalLinearMomentum.Norm(), diag.totalAngularMomentum.Norm());
            } else if (domain == scene::SimulationDomain::FieldGrid2D) {
                ImGui::Text("Step : %llu | t = %.4f | CFL = %.3f", static_cast<unsigned long long>(diag.stepCount), diag.simulationTime, diag.effectiveCflNumber);
                ImGui::Text("Field E : %.5g | L2 : %.5g", diag.fieldEnergy, diag.fieldL2Norm);
                ImGui::Text("Peak    : %.5g | Probe : %.5g", diag.fieldPeakValue, diag.fieldProbeValue);
            } else {
                const std::size_t sampleCount =
                    (ctx.primaryRadialSnap != nullptr) ? ctx.primaryRadialSnap->xValues.size() : 0U;
                ImGui::Text("Sweep Samples : %zu", sampleCount);
                ImGui::Text("<P/P0>        : %.6f", diag.meanVacuumPressureRatio);
                ImGui::Text("<c'/c>        : %.6f", diag.meanLocalLightSpeedRatio);
                ImGui::Text("<Haramein phi>: %.3e", diag.meanHolographicPhi);
            }

            if (diag.hasUnifiedTelemetry) {
                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.45f, 0.85f, 1.0f, 1.0f), "Thinker Telemetry (Model A)");
                if (diag.paisMaxPoyntingFlux > 0.0) {
                    ImGui::Text("Pais S_max  : %.4e W/m^2", diag.paisMaxPoyntingFlux);
                    ImGui::Text("Pais m_eff  : %.6f m0", diag.paisInertialMassFactor);
                }
                if (diag.totalPhryllFlux > 0.0 || diag.vogelCrystalSaturationRatio > 0.0) {
                    ImGui::Text("Phryll Flux : %.4g | Sat: %.4f", diag.totalPhryllFlux, diag.vogelCrystalSaturationRatio);
                }
                if (diag.vogelCrystalSinging) {
                    ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "WARNING: CRYSTAL SINGING THRESHOLD!");
                }
                if (diag.hasC204Telemetry) {
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.55f, 0.92f, 1.0f, 1.0f), "C204 Flight Computer & Metric");
                    ImGui::Text("State      : %s", diag.c204OperationalState.c_str());
                    ImGui::TextColored(
                        diag.c204IsForwardTravel
                            ? ImVec4(0.45f, 0.95f, 0.65f, 1.0f)
                            : ImVec4(1.0f, 0.72f, 0.40f, 1.0f),
                        "Direction  : %s",
                        diag.c204IsForwardTravel ? "FORWARD (-25% E)" : "BACKWARD (CTC)"
                    );
                    ImGui::Text(
                        "Worldline  : %.3f (dT=%+.3f yr) | Div %.3f%%",
                        diag.c204CurrentWorldlineYear, diag.c204ElapsedDisplacedYears, diag.c204WorldlineDivergencePct
                    );
                    ImGui::Text(
                        "Drive      : %.0f%% | %.0f C (%.2f yr/hr)",
                        diag.c204PowerLevelPercent, diag.c204InjectedChargeCoulombs, diag.c204PenetrationRateYrsPerHr
                    );
                    ImGui::Text(
                        "Cabin      : g00=%.4f | %.2f G | %.1f F",
                        diag.c204MetricG00AtCabin, diag.c204CabinGForce, diag.c204ExteriorTempF
                    );
                    ImGui::Text(
                        "O2 / Optics: %.1f%% | %s",
                        diag.c204CabinOxygenPct, diag.c204OpticalPhase.c_str()
                    );
                    ImGui::Text(
                        "VGL / Borda: #%llu | Next=%.0fs | Spr=%.1e",
                        static_cast<unsigned long long>(diag.c204VglCheckCount),
                        diag.c204VglNextCheckCountdownSec,
                        diag.c204BordaConsensusSpread
                    );
                    ImGui::Text(
                        "Horizons   : -:%llu | 0:%llu | +:%llu",
                        static_cast<unsigned long long>(diag.c204NegativeHorizonCells),
                        static_cast<unsigned long long>(diag.c204NullHorizonCells),
                        static_cast<unsigned long long>(diag.c204PositiveHorizonCells)
                    );
                    ImGui::Text(
                        "Kerr r_ph  : Pro %.2fM | Retro %.2fM",
                        diag.c204KerrProgradePhotonR, diag.c204KerrRetrogradePhotonR
                    );
                    if (diag.c204UvBurstActive) {
                        ImGui::TextColored(ImVec4(0.85f, 0.55f, 1.0f, 1.0f), "CAUTION: 1s UV BURST - EYE PROTECTION!");
                    }
                    if (diag.c204CarbonSensorAlert) {
                        ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.20f, 1.0f), "WARNING: CARBON SENSOR - REBREATHER!");
                    }
                    if (diag.c204OriginReturnTriggered) {
                        ImGui::TextColored(ImVec4(1.0f, 0.25f, 0.25f, 1.0f), "ABORT: %s", diag.c204AbortReason.c_str());
                    }
                }
            }
        }

        // ---------------------------------------------------------------------
        // 5. Simultaneous A/B Divergence Section
        // ---------------------------------------------------------------------
        if (ctx.abMetrics != nullptr && ctx.abMetrics->active &&
            ImGui::CollapsingHeader("Live A/B Divergence", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto& ab = *ctx.abMetrics;
            ImGui::TextColored(
                ImVec4(1.0f, 0.82f, 0.40f, 1.0f),
                "%s vs %s",
                scene::ModelIdToString(ctx.primaryModelId),
                scene::ModelIdToString(ctx.compareModelId)
            );
            if (domain == scene::SimulationDomain::NBodyDynamics) {
                ImGui::Text("Max dR   : %.4e | RMS dR : %.4e", ab.maxDeltaR, ab.rmsDeltaR);
                ImGui::Text("Max dV   : %.4e | dE     : %.4e", ab.maxDeltaV, ab.deltaTotalEnergy);
                ImGui::Text("Max dTau : %.4e", ab.maxDeltaProperTime);
            } else if (domain == scene::SimulationDomain::FieldGrid2D) {
                ImGui::Text("Grid L2 Residual : %.4e", ab.fieldL2Residual);
                ImGui::Text("Peak |A - B|     : %.4e", ab.fieldPeakResidual);
                ImGui::Text("Peak A / B       : %.3e / %.3e", ab.fieldPeakA, ab.fieldPeakB);
            } else {
                ImGui::Text("Max |A - B|      : %.4e", ab.curveMaxAbsDiff);
                ImGui::Text("RMS Diff         : %.4e", ab.curveRmsDiff);
                ImGui::Text("Max Rel Diff     : %.4e", ab.curveMaxRelDiff);
            }
        }
    }

    void PropertiesPanel::OnUIRender(UIContext& ctx) {
        if (!m_isOpen) {
            return;
        }

        if (ImGui::Begin(m_name.c_str(), &m_isOpen, m_flags)) {
            m_focused = ImGui::IsWindowFocused();
            m_hovered = ImGui::IsWindowHovered();

            const bool hasSelectedEntity =
                ctx.hasActiveScene && ctx.activeScene != nullptr &&
                ctx.selectedBodyIndex != nullptr &&
                ctx.selectedBodyIndex->has_value() &&
                ctx.selectedBodyIndex->value() < ctx.activeScene->bodies.size();

            if (hasSelectedEntity) {
                const bool selectEntityTabNow =
                    (ctx.entitySelectionJustChanged != nullptr && *ctx.entitySelectionJustChanged);
                if (ctx.entitySelectionJustChanged != nullptr) {
                    *ctx.entitySelectionJustChanged = false;
                }

                if (ImGui::BeginTabBar("PropertiesInspectorTabs")) {
                    const ImGuiTabItemFlags entityFlags =
                        selectEntityTabNow ? ImGuiTabItemFlags_SetSelected : ImGuiTabItemFlags_None;

                    if (ImGui::BeginTabItem("Entity", nullptr, entityFlags)) {
                        RenderEntityInspector(ctx, ctx.selectedBodyIndex->value());
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem("Simulation & A/B")) {
                        RenderSimulationInspector(ctx);
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            } else {
                if (ctx.entitySelectionJustChanged != nullptr) {
                    *ctx.entitySelectionJustChanged = false;
                }
                RenderSimulationInspector(ctx);
            }
        }
        ImGui::End();
    }

} // namespace umgebung::ui
