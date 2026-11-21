/**
 * @file TitorC204Solver.hpp
 * @brief Tier 2 Standalone Thinker Module (`Thinker::TitorC204`): FP64 CUDA
 *        `SimulationDomain::FieldGrid2D` solver simulating the **GE Army Model C204
 *        Gravity Distortion Time Displacement Unit** (`C204_Operators_Manual.pdf`,
 *        `CompleteTitor.pdf`, Pohlman `US 2006/0073976 A1`, & Tipler 1974).
 *
 * @details
 * Simultaneously executes two tightly coupled subsystems on every simulation step
 * $\Delta t$ (measured in pilot proper seconds):
 *   1. **GPU FP64 2D Effective Phenomenological Metric Evaluation:**
 *      Launches `ComputeTitorC204MetricGrid2DKernel` across an $N_x \times N_y$ spatial
 *      slice spanning the $22\,\text{m} \times 6\,\text{m} \times 6\,\text{m}$ Van Den
 *      Broeck / Alcubierre ellipsoidal envelope, evaluating:
 *        - Damped offset Tipler sinusoid lapse $\alpha(x, y, t)$,
 *        - Rotating dual-singularity Lense-Thirring frame-dragging shift $\|\boldsymbol{\beta}(x, y, t)\|$,
 *        - Effective 3+1 ADM time-time metric $g_{00}^{\text{eff}} = -(\alpha^2 - \|\boldsymbol{\beta}\|^2)$
 *          capturing the concentric Negative, Null, and Positive Time Event Horizons,
 *        - Screened cabin proper gravitational acceleration $a(x, y, t) / g_0$ (`1.5–2.0 G`
 *          at the pilot seat, dropping to `0 G` outside the `22 m × 6 m` envelope),
 *        - Rearward $170^\circ$ X-ray radiation venting cone classification.
 *   2. **C204 Onboard Flight Computer & Operational State Machine:**
 *      Advances the discrete checklist and safety failsafes verbatim from `TMIM-200S/4005`:
 *        - Pre-flight mass limit enforcement ($\le 2500\,\text{kg}$),
 *        - `VglBaselineLock` $\to$ `RampUp` (with $1\,\text{s}$ rainbow lensing & UV burst)
 *          $\to$ `TemporalTransit` (total optical blackout at $3.4 / 6.2 / 10.0\,\text{yrs/hr}$)
 *          $\to$ `VglPulseCheck` (every $240\,\text{s}$, $54{,}000\,\text{C}$ budget, Borda
 *          triple-computer consensus voting) $\to$ `ArrivalCooldown`,
 *        - Automatic Origin Return (`OriginReturnAbort`) on VGL gravity anomaly,
 *          payload overload, or hazardous trapped-air-pocket carbon sensor reading.
 *
 * @note Zero GUI dependencies; compiles identically in Desktop GUI and Headless HPC modes.
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/compute/DeviceBuffer.hpp"
#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/models/thinkers/TitorC204Model.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace umgebung::models::thinkers {

    /**
     * @class TitorC204Solver
     * @brief Standalone FP64 CUDA solver for `ModelId::ThinkerTitorC204` in
     *        `SimulationDomain::FieldGrid2D`.
     */
    class TitorC204Solver final : public IPhysicsSolver {
    public:
        TitorC204Solver();
        ~TitorC204Solver() override;

        TitorC204Solver(const TitorC204Solver&) = delete;
        TitorC204Solver& operator=(const TitorC204Solver&) = delete;
        TitorC204Solver(TitorC204Solver&&) = delete;
        TitorC204Solver& operator=(TitorC204Solver&&) = delete;

        /** @brief Binds this solver instance to CUDA stream `streamIndex` (`0` = Primary, `1` = Overlay). */
        void SetStreamIndex(std::size_t streamIndex) noexcept override { m_streamIndex = streamIndex; }

        /**
         * @brief Allocates FP64 2D metric GPU buffers, validates C204 payload & geometry
         *        parameters, evaluates initial VGL baseline lock, and computes the `t = 0` slice.
         *
         * @param context Active initialized `CUDAContext`.
         * @param scene Validated `UniverseScene` carrying `fieldGrid2D` and `titorC204` config.
         * @return `true` on successful GPU initialization.
         */
        [[nodiscard]] bool Initialize(
            compute::CUDAContext& context,
            const scene::UniverseScene& scene
        ) override;

        /** @brief Synchronizes the CUDA stream and releases all allocated device buffers. */
        void Shutdown() noexcept override;

        /**
         * @brief Advances the C204 operational state machine by `dt` pilot proper seconds
         *        and re-evaluates the 2D FP64 CUDA Effective Phenomenological Metric grid.
         *
         * @param dt Proper time step $\Delta \tau > 0$ in seconds.
         * @return `true` if the CUDA kernel launch and state update succeeded.
         */
        [[nodiscard]] bool Step(double dt) override;

        /** @brief Executes `stepCount` consecutive steps and synchronizes the CUDA stream. */
        [[nodiscard]] bool StepMultiple(std::uint64_t stepCount, double dt) override;

        /** @brief Synchronizes the solver's CUDA stream. */
        [[nodiscard]] bool Synchronize() const noexcept override;

        /** @brief Copies stationary C204 hardware anchors (singularities, cabin, VGL) to `outBodies`. */
        [[nodiscard]] bool SyncToHost(std::vector<scene::BodyEntity>& outBodies) override;

        /** @brief Downloads GPU metric grids and populates complete C204 operational & metric telemetry. */
        [[nodiscard]] bool ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) override;

        /** @brief Downloads the 2D spatial spacetime distortion heatmap $|1 + g_{00}^{\text{eff}}(x,y,t)|$. */
        [[nodiscard]] bool FetchFieldGrid2DSnapshot(FieldGrid2DSnapshot& outSnapshot) override;

        /** @brief Returns `SimulationDomain::FieldGrid2D`. */
        [[nodiscard]] scene::SimulationDomain GetDomain() const noexcept override {
            return scene::SimulationDomain::FieldGrid2D;
        }

        [[nodiscard]] bool IsInitialized() const noexcept override { return m_initialized; }

        [[nodiscard]] scene::ModelId GetModelId() const noexcept override {
            return scene::ModelId::ThinkerTitorC204;
        }

        [[nodiscard]] const char* GetSolverName() const noexcept override {
            return "Thinker::TitorC204 (GE C204 Gravity Distortion Time Displacement Unit, FP64 CUDA)";
        }

        [[nodiscard]] scene::PhysicsProfileType GetProfileType() const noexcept override {
            return scene::PhysicsProfileType::Custom;
        }

        [[nodiscard]] scene::IntegratorType GetIntegratorType() const noexcept override {
            return m_integrator;
        }

        [[nodiscard]] std::size_t GetBodyCount() const noexcept override { return m_stationaryAnchors.size(); }
        [[nodiscard]] std::uint64_t GetStepCount() const noexcept override { return m_stepCount; }
        [[nodiscard]] double GetSimulationTime() const noexcept override { return m_simulationTime; }

    private:
        [[nodiscard]] bool LaunchMetricGridKernel();
        [[nodiscard]] bool DownloadGridBuffers();
        void AdvanceOperationalStateMachine(double dt);

        compute::CUDAContext* m_context{nullptr};
        std::size_t m_streamIndex{0U};
        cudaStream_t m_stream{nullptr};
        bool m_initialized{false};

        scene::FieldGrid2DConfig m_gridConfig{};
        scene::TitorC204Params m_params{};
        scene::IntegratorType m_integrator{scene::IntegratorType::VelocityVerlet};
        std::vector<scene::BodyEntity> m_stationaryAnchors;

        std::uint32_t m_width{128U};
        std::uint32_t m_height{128U};
        std::size_t m_cellCount{0U};
        double m_dx{0.25};

        // Flight computer & operational state variables.
        C204OperationalState m_state{C204OperationalState::Standby};
        double m_rampFraction{0.0};
        double m_stateTimerSec{0.0};
        double m_vglTimerSec{0.0};
        std::uint64_t m_vglCheckCount{0ULL};
        double m_bordaConsensusSpread{0.0};
        double m_effectiveChargeCoulombs{c204::kCharge100PctCoulombs};
        double m_expendedChargeCoulombs{0.0};
        double m_penetrationRateYrsPerHr{c204::kRate100PctYrsPerHr};
        double m_timeVelocitySecPerSec{87660.0};
        double m_elapsedDisplacedYears{0.0};
        double m_currentWorldlineYear{2036.0};
        double m_worldlineDivergencePct{0.0};
        double m_cabinOxygenPct{100.0};
        bool m_carbonSensorAlert{false};
        bool m_uvBurstActive{false};
        bool m_opticalBlackoutActive{false};
        bool m_originReturnTriggered{false};
        std::string m_abortReason;

        std::uint64_t m_stepCount{0ULL};
        double m_simulationTime{0.0};

        // FP64 CUDA 2D spatial metric buffers.
        compute::DeviceBuffer<double> m_visualDistortion;
        compute::DeviceBuffer<double> m_g00Grid;
        compute::DeviceBuffer<double> m_accelGGrid;
        compute::DeviceBuffer<int> m_horizonZoneGrid;

        // Reusable host staging buffers.
        std::vector<double> m_hostVisual;
        std::vector<double> m_hostG00;
        std::vector<double> m_hostAccelG;
        std::vector<int> m_hostHorizonZone;
    };

} // namespace umgebung::models::thinkers
