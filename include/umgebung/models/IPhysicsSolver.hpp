/**
 * @file IPhysicsSolver.hpp
 * @brief Pluggable polymorphism boundary decoupling `core::Application`,
 *        `ModelRegistry`, and CLI/GUI runners from concrete physics models
 *        across all three architectural tiers (`models/mainstream/`,
 *        `models/thinkers/`, and `models/unified/`).
 *
 * @details
 * Enables reproducible live Primary Model switching and simultaneous GPU
 * Model A + Model B overlaid execution on identical `.umgebung` /
 * `.scenario.umgebung` initial conditions:
 *   - `ModelRegistry::CreateSolver(ModelId)` instantiates any registered solver.
 *   - `SetStreamIndex(idx)` binds Model A to CUDA stream 0 and Model B to CUDA
 *     stream 1 so both solvers step concurrently on the GPU without serializing.
 *   - `SolverDiagnostics` carries both universal Hamiltonian invariants and
 *     per-Thinker / NFW telemetry fields cleanly.
 *
 * @note Zero GUI dependencies; compiles identically in Desktop GUI and Headless HPC modes.
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/scene/PhysicsProfile.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * @namespace umgebung::models
 * @brief Pluggable physics solver abstractions, three-tier Model Registry, and
 *        concrete GPU solvers spanning Pillar 1 (`mainstream/`) and Pillar 2
 *        (`thinkers/` + `unified/`).
 */
namespace umgebung::models {

    /**
     * @struct SolverDiagnostics
     * @brief Unified telemetry record reporting universal conservation
     *        invariants (Pillar 1 & 2), Mainstream NFW halo energy, and
     *        verbatim per-Thinker / Unified field diagnostics.
     */
    struct SolverDiagnostics {
        // ---------------------------------------------------------------------
        // Solver Identity & Universal Dynamical Conservation Invariants
        // ---------------------------------------------------------------------

        /** @brief Canonical three-tier `ModelId` of the solver that produced this record. */
        scene::ModelId modelId{scene::ModelId::MainstreamNewtonian};

        /** @brief Total number of integration steps completed since initialization. */
        std::uint64_t stepCount{0ULL};

        /** @brief Accumulated global coordinate simulation time $t$. */
        double simulationTime{0.0};

        /** @brief Total kinetic energy $T = \frac{1}{2} \sum_i m_i \|\mathbf{v}_i\|^2$. */
        double kineticEnergy{0.0};

        /** @brief Total gravitational/interaction potential energy $U$. */
        double potentialEnergy{0.0};

        /** @brief Total mechanical energy $E = T + U$. */
        double totalEnergy{0.0};

        /** @brief Baseline total energy $E_0$ captured at `t = 0`. */
        double initialTotalEnergy{0.0};

        /** @brief Dimensionless relative energy drift $|E - E_0| / \max(|E_0|, 10^{-30})$. */
        double relativeEnergyDrift{0.0};

        /** @brief Total linear momentum vector $\mathbf{P} = \sum_i m_i \mathbf{v}_i$. */
        scene::Vec3d totalLinearMomentum{};

        /** @brief Total angular momentum vector $\mathbf{L} = \sum_i m_i (\mathbf{r}_i \times \mathbf{v}_i)$. */
        scene::Vec3d totalAngularMomentum{};

        /** @brief System center-of-mass position $\mathbf{R}_{\text{cm}}$. */
        scene::Vec3d centerOfMass{};

        // ---------------------------------------------------------------------
        // Tier 1 Extension: Mainstream Dark Matter NFW Halo Telemetry
        // ---------------------------------------------------------------------

        /** @brief Total NFW dark matter halo potential energy $U_{\text{NFW}}$ (`Mainstream::DarkMatterNFW`). */
        double nfwHaloPotentialEnergy{0.0};

        // ---------------------------------------------------------------------
        // Tier 2 & Tier 3 Thinker / Unified Field Telemetry
        // ---------------------------------------------------------------------

        /** @brief True when the active solver populates Thinker/Unified field telemetry below. */
        bool hasUnifiedTelemetry{false};

        /** @brief Mean Essonne vacuum pressure ratio $\langle P_i / P_0 \rangle$ (`8052.pdf` Eq. 21). */
        double meanVacuumPressureRatio{1.0};

        /** @brief Mean Essonne local speed-of-light ratio $\langle c'_i / c \rangle = \langle \sqrt{P_i/P_0} \rangle$ (`8052.pdf` Eq. 27). */
        double meanLocalLightSpeedRatio{1.0};

        /** @brief Mean non-linear local proper time rate $\langle d\tau_i / dt \rangle$ (Danaan / Essonne coupling). */
        double meanLocalTimeRate{1.0};

        /** @brief Mean accumulated local proper time $\langle \tau_i \rangle$ across bodies. */
        double meanLocalProperTime{0.0};

        /** @brief Mean Haramein holographic surface-to-volume ratio $\langle \phi_i \rangle$ (`Haramein342013` Eq. 22–26). */
        double meanHolographicPhi{0.0};

        /** @brief Active Haramein-Rauscher spacetime-torque angular rate $\Omega_z$. */
        double harameinSpacetimeTorqueOmegaZ{0.0};

        /** @brief Salvatore Pais HEEMFG maximum Poynting energy flux $S_{\max}$ (`US10144532.pdf` Eq. 1). */
        double paisMaxPoyntingFlux{0.0};

        /** @brief Salvatore Pais effective inertial mass factor $m_{\text{eff}} / m_0 = 1 / (1 + \kappa S_{\max}) \in (0, 1]$. */
        double paisInertialMassFactor{1.0};

        /** @brief Total active Phryll field flux summed across all bodies and ambient generators. */
        double totalPhryllFlux{0.0};

        /** @brief Vogel/Willis cumulative crystal saturation ratio $E_{\text{cum}} / E_{\text{sat}}$ (`PhryllGenInstructions.pdf`). */
        double vogelCrystalSaturationRatio{0.0};

        /** @brief True when the Vogel/Willis crystal reached or exceeded its singing/implosion saturation ceiling. */
        bool vogelCrystalSinging{false};

        /** @brief Concise human-readable active solver description for logs and UI. */
        std::string solverSummary;
    };

    /**
     * @class IPhysicsSolver
     * @brief Abstract runtime interface implemented by every GPU simulation
     *        solver across Mainstream, standalone Thinker, and Unified Synthesis tiers.
     */
    class IPhysicsSolver {
    public:
        virtual ~IPhysicsSolver() = default;

        /**
         * @brief Selects the CUDA stream pool index borrowed from `CUDAContext`
         *        on subsequent `Initialize()` calls (e.g., `0` for Primary Model A,
         *        `1` for simultaneous Comparison Overlay Model B).
         *
         * @param streamIndex Zero-based index passed to `CUDAContext::GetStream()`.
         */
        virtual void SetStreamIndex(std::size_t streamIndex) noexcept {
            (void)streamIndex;
        }

        /**
         * @brief Allocates GPU resources under `context`'s VRAM budget and
         *        initializes internal state from `scene`.
         */
        [[nodiscard]] virtual bool Initialize(
            compute::CUDAContext& context,
            const scene::UniverseScene& scene
        ) = 0;

        /**
         * @brief Synchronizes pending GPU work and releases all owned GPU buffers.
         */
        virtual void Shutdown() noexcept = 0;

        /**
         * @brief Enqueues one integration step of size `dt` on the solver's CUDA stream.
         */
        [[nodiscard]] virtual bool Step(double dt) = 0;

        /**
         * @brief Enqueues `stepCount` steps of size `dt` and synchronizes once on completion.
         */
        [[nodiscard]] virtual bool StepMultiple(std::uint64_t stepCount, double dt) = 0;

        /**
         * @brief Blocks until all queued GPU work for this solver finishes.
         */
        [[nodiscard]] virtual bool Synchronize() const noexcept = 0;

        /**
         * @brief Downloads current body positions and velocities into `outBodies`.
         */
        [[nodiscard]] virtual bool SyncToHost(std::vector<scene::BodyEntity>& outBodies) = 0;

        /**
         * @brief Computes and returns the current conservation & field telemetry snapshot.
         */
        [[nodiscard]] virtual bool ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) = 0;

        /**
         * @brief Downloads per-body accumulated local proper times $\tau_i$.
         *        Linear-time solvers default to returning coordinate time $t$ for every body.
         */
        [[nodiscard]] virtual bool GetLocalProperTimes(std::vector<double>& outProperTimes) {
            outProperTimes.assign(GetBodyCount(), GetSimulationTime());
            return true;
        }

        /** @brief Returns `true` if `Initialize()` succeeded and `Shutdown()` has not run. */
        [[nodiscard]] virtual bool IsInitialized() const noexcept = 0;

        /** @brief Returns the canonical `ModelId` implemented by this solver. */
        [[nodiscard]] virtual scene::ModelId GetModelId() const noexcept = 0;

        /** @brief Returns the human-readable solver name. */
        [[nodiscard]] virtual const char* GetSolverName() const noexcept = 0;

        /** @brief Returns the broad `PhysicsProfileType` category handled by this solver. */
        [[nodiscard]] virtual scene::PhysicsProfileType GetProfileType() const noexcept = 0;

        /** @brief Returns the active numerical integrator (`VelocityVerlet` or `EulerExplicit`). */
        [[nodiscard]] virtual scene::IntegratorType GetIntegratorType() const noexcept = 0;

        /** @brief Returns the number of simulated bodies $N$. */
        [[nodiscard]] virtual std::size_t GetBodyCount() const noexcept = 0;

        /** @brief Returns the completed integration step count. */
        [[nodiscard]] virtual std::uint64_t GetStepCount() const noexcept = 0;

        /** @brief Returns the accumulated global coordinate simulation time $t$. */
        [[nodiscard]] virtual double GetSimulationTime() const noexcept = 0;
    };

} // namespace umgebung::models
