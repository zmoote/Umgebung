/**
 * @file IPhysicsSolver.hpp
 * @brief Pluggable polymorphism boundary decoupling `core::Application` and
 *        UI/CLI runners from individual physics implementations (`Mainstream`
 *        vs. `Unified` vs. `Custom`).
 *
 * @details
 * Enables reproducible side-by-side A/B comparisons between Pillar 1
 * (`models/mainstream/`) and Pillar 2 (`models/unified/`) on identical
 * `.umgebung` initial conditions: `Application` constructs the solver via
 * `CreateSolverForScene(scene)` and interacts solely through `IPhysicsSolver`
 * and `SolverDiagnostics`.
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
 * @brief Pluggable physics solver abstractions and factory dispatch spanning
 *        Pillar 1 (`mainstream/`) and Pillar 2 (`unified/`).
 */
namespace umgebung::models {

    /**
     * @struct SolverDiagnostics
     * @brief Unified telemetry record reporting both classical conservation
     *        invariants (Pillar 1 & 2) and optional vacuum/holographic/Phryll
     *        field diagnostics (Pillar 2).
     */
    struct SolverDiagnostics {
        // ---------------------------------------------------------------------
        // Universal Dynamical & Conservation Invariants
        // ---------------------------------------------------------------------

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
        // Pillar 2 Unified Research Field Telemetry (Populated by Unified Solvers)
        // ---------------------------------------------------------------------

        /** @brief True when the active solver populates Pillar 2 unified field telemetry below. */
        bool hasUnifiedTelemetry{false};

        /** @brief Mean Essonne vacuum pressure ratio $\langle P_i / P_0 \rangle$ across bodies (`8052.pdf` Eq. 21). */
        double meanVacuumPressureRatio{1.0};

        /** @brief Mean Essonne local speed-of-light ratio $\langle c'_i / c \rangle = \langle \sqrt{P_i/P_0} \rangle$ (`8052.pdf` Eq. 27). */
        double meanLocalLightSpeedRatio{1.0};

        /** @brief Mean non-linear local proper time rate $\langle d\tau_i / dt \rangle$ (Danaan / Essonne coupling). */
        double meanLocalTimeRate{1.0};

        /** @brief Mean accumulated local proper time $\langle \tau_i \rangle$ across bodies. */
        double meanLocalProperTime{0.0};

        /** @brief Mean Haramein holographic surface-to-volume ratio $\langle \phi_i \rangle$ (`Haramein342013` Eq. 22–26). */
        double meanHolographicPhi{0.0};

        /** @brief Total active Phryll field flux summed across all bodies in the universe. */
        double totalPhryllFlux{0.0};

        /** @brief Concise human-readable active solver description for logs and UI. */
        std::string solverSummary;
    };

    /**
     * @class IPhysicsSolver
     * @brief Abstract runtime interface implemented by every GPU/CPU simulation
     *        solver in Umgebung.
     */
    class IPhysicsSolver {
    public:
        virtual ~IPhysicsSolver() = default;

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
         * @brief Enqueues one integration step of size `dt` on the GPU stream.
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

        /** @brief Returns `true` if `Initialize()` succeeded and `Shutdown()` has not run. */
        [[nodiscard]] virtual bool IsInitialized() const noexcept = 0;

        /** @brief Returns the human-readable solver name (e.g., `"Mainstream Newtonian N-Body (FP64 CUDA)"`). */
        [[nodiscard]] virtual const char* GetSolverName() const noexcept = 0;

        /** @brief Returns the `PhysicsProfileType` handled by this solver instance. */
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
