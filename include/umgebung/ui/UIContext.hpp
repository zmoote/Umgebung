/**
 * @file UIContext.hpp
 * @brief Shared state-and-command bridge connecting `umgebung::core::Application`
 *        with `umgebung::ui::UIManager` and every modular Dear ImGui panel.
 *
 * @details
 * Keeps `umgebung::ui` cleanly decoupled from `Application`'s private internals
 * while giving every docked and floating panel zero-overhead access to:
 *   - Hardware capability/adaptation snapshots, CUDA context telemetry, and
 *     selected Vulkan GPU info (`StatisticsPanel`, `AboutPanel`),
 *   - Three-scope scene workspace (`Scenarios`, `Bubble Universes`,
 *     `FlowerOfLifeMultiverse`) and active `UniverseScene` (`HierarchyPanel`,
 *     `PropertiesPanel`, `ViewportPanel`),
 *   - Domain-filtered Primary Model A + Simultaneous Overlay Model B state,
 *     live GPU snapshots (`NBodyDynamics` bodies & trails, `FieldGrid2D` scalar
 *     maps & center-probe waveforms, `RadialAndScaleProfile` analytical curves),
 *     diagnostics, and A/B divergence metrics,
 *   - Explicit legacy-aligned playback state (`Stopped`, `Playing`, `Paused`),
 *     wall-clock pacing parameters, entity selection/focus state, and strongly
 *     typed action callbacks back into `Application`.
 *
 * @warning Compiled strictly when `UMGEBUNG_BUILD_GUI=ON` (`UMGEBUNG_HAS_GUI == 1`).
 */

#pragma once

#if !defined(UMGEBUNG_HAS_GUI) || (UMGEBUNG_HAS_GUI == 0)
    #error "umgebung/ui/UIContext.hpp is exclusive to GUI builds (UMGEBUNG_BUILD_GUI=ON)."
#endif

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/core/SystemProfiler.hpp"
#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/FlowerOfLifeMultiverse.hpp"
#include "umgebung/scene/Multiverse.hpp"
#include "umgebung/scene/PhysicsProfile.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace umgebung::core {
    struct ABComparisonMetrics;
}

namespace umgebung::ui {

    /**
     * @enum SimulationPlaybackState
     * @brief Explicit interactive playback state mirroring legacy Umgebung's
     *        `Play / Pause / Stop` simulation controls.
     *
     * @details
     *   - `Stopped`: Simulation is halted at its initial/reset state (`t = 0`),
     *     ideal for inspecting or editing scene entities before launching a run.
     *   - `Playing`: Active GPU solver(s) advance continuously each frame using
     *     the configured fixed-timestep pacing or manual steps-per-frame.
     *   - `Paused`: Active GPU solver(s) are frozen at the current simulation
     *     time `t` without discarding trajectories, fields, or diagnostics.
     */
    enum class SimulationPlaybackState : std::uint8_t {
        Stopped = 0,
        Playing = 1,
        Paused = 2
    };

    /**
     * @brief Converts a `SimulationPlaybackState` value to a human-readable label.
     *
     * @param state Playback state enumerator.
     * @return Static string literal (`"Stopped"`, `"Playing"`, or `"Paused"`).
     */
    [[nodiscard]] inline const char* SimulationPlaybackStateToString(
        SimulationPlaybackState state
    ) noexcept {
        switch (state) {
            case SimulationPlaybackState::Stopped: return "Stopped";
            case SimulationPlaybackState::Playing: return "Playing";
            case SimulationPlaybackState::Paused:  return "Paused";
        }
        return "Stopped";
    }

    /**
     * @struct UIContext
     * @brief Non-owning view of live engine data and command callbacks passed
     *        into `UIManager::Render()` and each modular `Panel::OnUIRender()`.
     *
     * @pre All non-null pointers referenced by panels MUST remain valid for the
     *      duration of `UIManager::Render()`.
     */
    struct UIContext {
        // ---------------------------------------------------------------------
        // Engine & Hardware Metadata (Read-Only Telemetry)
        // ---------------------------------------------------------------------
        const char* engineVersion{"0.0.1"};
        const core::HardwareCapabilities* hardwareCaps{nullptr};
        const core::HardwareAdaptationPlan* adaptationPlan{nullptr};
        const compute::CUDAContext* cudaContext{nullptr};
        std::string selectedVulkanGpuName;
        bool vulkanMatchedCudaUuid{false};

        // ---------------------------------------------------------------------
        // Three-Scope Workspace & Active Scene
        // ---------------------------------------------------------------------
        scene::Multiverse* scenarios{nullptr};
        scene::Multiverse* universes{nullptr};
        scene::FlowerOfLifeMultiverse* flowerMultiverse{nullptr};

        scene::SimulationScope activeScope{scene::SimulationScope::Scenario};
        bool hasActiveScene{false};
        scene::UniverseScene* activeScene{nullptr};
        std::filesystem::path currentScenePath;

        // ---------------------------------------------------------------------
        // Active Solver Selection, Models & Playback State
        // ---------------------------------------------------------------------
        bool hasActiveSolver{false};
        scene::ModelId primaryModelId{scene::ModelId::MainstreamNewtonian};
        bool compareOverlayEnabled{false};
        scene::ModelId compareModelId{scene::ModelId::ThinkerEssonne};

        SimulationPlaybackState playbackState{SimulationPlaybackState::Playing};
        double* activeTimestep{nullptr};
        bool* wallClockPaced{nullptr};
        float* simTimeScale{nullptr};
        double* simTimeAccumulator{nullptr};
        int maxSubStepsPerFrame{256};
        int lastFrameExecutedSteps{0};
        int* stepsPerFrame{nullptr};

        // ---------------------------------------------------------------------
        // Domain Snapshots, Live Trails & Quantitative Diagnostics
        // ---------------------------------------------------------------------
        const std::vector<scene::BodyEntity>* liveBodies{nullptr};
        const std::vector<scene::BodyEntity>* compareBodies{nullptr};
        const std::vector<double>* primaryProperTimes{nullptr};
        const std::vector<double>* compareProperTimes{nullptr};
        const std::vector<std::deque<scene::Vec3d>>* bodyTrails{nullptr};
        const std::vector<std::deque<scene::Vec3d>>* compareTrails{nullptr};

        const models::FieldGrid2DSnapshot* primaryFieldSnap{nullptr};
        const models::FieldGrid2DSnapshot* compareFieldSnap{nullptr};
        const std::deque<float>* probeWaveformA{nullptr};
        const std::deque<float>* probeWaveformB{nullptr};

        const models::RadialProfileSnapshot* primaryRadialSnap{nullptr};
        const models::RadialProfileSnapshot* compareRadialSnap{nullptr};

        const models::SolverDiagnostics* latestDiagnostics{nullptr};
        const models::SolverDiagnostics* compareDiagnostics{nullptr};
        const core::ABComparisonMetrics* abMetrics{nullptr};

        // ---------------------------------------------------------------------
        // Interactive Selection & Viewport Camera State
        // ---------------------------------------------------------------------
        /** @brief Currently selected body index in `activeScene->bodies` (`std::nullopt` = none). */
        std::optional<std::size_t>* selectedBodyIndex{nullptr};

        /**
         * @brief Sticky notification flag set by `HierarchyPanel` when a user
         *        selects an entity so `PropertiesPanel` can switch focus to the
         *        Entity Inspector tab automatically.
         */
        bool* entitySelectionJustChanged{nullptr};

        /** @brief Displays the Flower of Life Multiverse lattice in `ViewportPanel`. */
        bool* showFlowerOfLifeLatticeView{nullptr};

        /** @brief FieldGrid2D heatmap channel selector (`0` = Model A, `1` = Model B, `2` = |A - B|). */
        int* fieldHeatmapMode{nullptr};

        /** @brief Automatic bounding-box camera scaling for `NBodyDynamics` 2D viewport. */
        bool* viewportAutoFit{nullptr};

        /** @brief Current half-extent (world units) of the `NBodyDynamics` 2D viewport. */
        float* viewportHalfExtent{nullptr};

        /** @brief Current 2D viewport camera pan center `(x, y)` in scene world units. */
        float* viewportPanX{nullptr};
        float* viewportPanY{nullptr};

        // ---------------------------------------------------------------------
        // Engine Command Callbacks (Routed Back to `core::Application`)
        // ---------------------------------------------------------------------
        std::function<void()> onRequestExit;
        std::function<void()> onPlay;
        std::function<void()> onPause;
        std::function<void()> onStop;
        std::function<void()> onSingleStep;
        std::function<void()> onResetSimulation;

        std::function<void()> onNewScene;
        std::function<bool(const std::filesystem::path&)> onOpenSceneFile;
        std::function<bool(const std::filesystem::path&)> onSaveSceneFile;

        std::function<bool(
            const scene::UniverseScene&,
            scene::SimulationScope,
            std::optional<scene::ModelId>,
            std::optional<scene::ModelId>
        )> onActivateScene;

        std::function<void(scene::ModelId)> onSelectPrimaryModel;
        std::function<void(bool, std::optional<scene::ModelId>)> onSetCompareOverlay;
        std::function<void()> onSceneParametersChanged;

        std::function<void(const scene::BodyEntity&)> onCreateBodyEntity;
        std::function<void(std::size_t)> onDeleteBodyEntity;
        std::function<void(std::size_t)> onFocusBodyEntity;
    };

} // namespace umgebung::ui
