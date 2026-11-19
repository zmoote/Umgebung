/**
 * @file UnifiedSynthesisSolver.hpp
 * @brief Tier 3 Combined Unified Synthesis (`Unified::Synthesis`): FP64 CUDA
 *        solver composing all five standalone verbatim Thinker modules
 *        (`Essonne`, `Haramein`, `Pais`, `Danaan`, `VogelWillis`) into a single
 *        multi-scale GPU reality solver without duplicating any equations.
 *
 * @details
 * Imports every equation directly from `models/thinkers/<Thinker>Model.hpp` via the
 * `UnifiedPhysicsEquations.hpp` umbrella header:
 *   1. **Chris Essonne (`EssonneModel.hpp`):** 2D galactic aether-vortex field
 *      (`10100.pdf` Eq. 7 & 11), vacuum pressure ratio $P_i/P_0$ (`8052.pdf` Eq. 21),
 *      and local speed of light $c'_i/c$ (`8052.pdf` Eq. 27).
 *   2. **Nassim Haramein (`HarameinModel.hpp`):** Holographic PSU ratio $\phi_i$
 *      (`Haramein342013PRRI3363.pdf`) and symplectic Strang-split Haramein-Rauscher
 *      spacetime-torque Coriolis dynamics (`-2 Omega x v`).
 *   3. **Salvatore Pais (`PaisModel.hpp`):** HEEMFG Eq. (1) maximum Poynting flux
 *      $S_{\max}$ (`US10144532.pdf`) and vacuum-polarized inertial mass reduction
 *      $m_{\text{eff}} = m_0 / (1 + \kappa_{\text{Pais}} S_{\max})$ on HEEMFG bodies.
 *   4. **Elena Danaan (`DanaanModel.hpp`):** Antisymmetric Love binding coherence
 *      force and vibrational-density coupled non-linear local proper time $\tau_i$.
 *   5. **Marcel Vogel / Dan Willis / Eredyon (`VogelWillisModel.hpp`):** 4096 Hz
 *      torsion-coil Phryll resonant coupling and cumulative crystal saturation /
 *      "singing" threshold monitor (`PhryllGenInstructions.pdf`).
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/compute/DeviceBuffer.hpp"
#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace umgebung::models::unified {

    /**
     * @class UnifiedSynthesisSolver
     * @brief Tier 3 FP64 CUDA solver composing all standalone Thinker models
     *        under `ModelId::UnifiedSynthesis`.
     */
    class UnifiedSynthesisSolver final : public IPhysicsSolver {
    public:
        static constexpr int kFieldTileSize = 256;

        UnifiedSynthesisSolver() = default;
        ~UnifiedSynthesisSolver() override;

        UnifiedSynthesisSolver(const UnifiedSynthesisSolver&) = delete;
        UnifiedSynthesisSolver& operator=(const UnifiedSynthesisSolver&) = delete;
        UnifiedSynthesisSolver(UnifiedSynthesisSolver&&) = delete;
        UnifiedSynthesisSolver& operator=(UnifiedSynthesisSolver&&) = delete;

        void SetStreamIndex(std::size_t streamIndex) noexcept override { m_streamIndex = streamIndex; }

        [[nodiscard]] bool Initialize(
            compute::CUDAContext& context,
            const scene::UniverseScene& scene
        ) override;

        void Shutdown() noexcept override;
        [[nodiscard]] bool Step(double dt) override;
        [[nodiscard]] bool StepMultiple(std::uint64_t stepCount, double dt) override;
        [[nodiscard]] bool Synchronize() const noexcept override;
        [[nodiscard]] bool SyncToHost(std::vector<scene::BodyEntity>& outBodies) override;
        [[nodiscard]] bool ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) override;
        [[nodiscard]] bool GetLocalProperTimes(std::vector<double>& outProperTimes) override;

        /** @brief Downloads per-body Essonne vacuum pressure ratios $P_i/P_0$ from the GPU. */
        [[nodiscard]] bool GetVacuumPressureRatios(std::vector<double>& outPressureRatios);

        [[nodiscard]] bool IsInitialized() const noexcept override { return m_initialized; }

        [[nodiscard]] scene::ModelId GetModelId() const noexcept override {
            return scene::ModelId::UnifiedSynthesis;
        }

        [[nodiscard]] const char* GetSolverName() const noexcept override {
            return "Unified::Synthesis (All 5 Thinkers Composed, FP64 CUDA)";
        }

        [[nodiscard]] scene::PhysicsProfileType GetProfileType() const noexcept override {
            return scene::PhysicsProfileType::Unified;
        }

        [[nodiscard]] scene::IntegratorType GetIntegratorType() const noexcept override {
            return m_integrator;
        }

        [[nodiscard]] std::size_t GetBodyCount() const noexcept override { return m_bodyCount; }
        [[nodiscard]] std::uint64_t GetStepCount() const noexcept override { return m_stepCount; }
        [[nodiscard]] double GetSimulationTime() const noexcept override { return m_simulationTime; }

    private:
        [[nodiscard]] bool LaunchSynthesisFieldKernel();
        [[nodiscard]] bool DownloadStateToStaging();

        compute::CUDAContext* m_context{nullptr};
        std::size_t m_streamIndex{0U};
        cudaStream_t m_stream{nullptr};

        bool m_initialized{false};
        std::size_t m_bodyCount{0U};
        scene::IntegratorType m_integrator{scene::IntegratorType::VelocityVerlet};

        double m_gravitationalConstant{1.0};
        double m_softeningLength{0.0};
        scene::UnifiedFieldParams m_unifiedParams{};

        // Derived Pais & Vogel/Willis invariants cached at Initialize().
        double m_paisMaxPoyntingFlux{0.0};
        double m_paisInertialMassFactor{1.0};
        double m_paisAccelAmplification{1.0};
        double m_vogelDrivePower{0.0};
        double m_totalBodyPhryll{0.0};
        double m_cumulativeCrystalEnergy{0.0};

        std::uint64_t m_stepCount{0ULL};
        double m_simulationTime{0.0};
        double m_initialTotalEnergy{0.0};

        // GPU SoA FP64 Device Buffers
        compute::DeviceBuffer<double> m_posX, m_posY, m_posZ;
        compute::DeviceBuffer<double> m_velX, m_velY, m_velZ;
        compute::DeviceBuffer<double> m_accX, m_accY, m_accZ;
        compute::DeviceBuffer<double> m_mass, m_radius;
        compute::DeviceBuffer<double> m_vibrationalDensity, m_coherence, m_phryllCharge;
        compute::DeviceBuffer<double> m_accelScale;
        compute::DeviceBuffer<double> m_vacuumPressureRatio, m_localProperTime;

        // Reusable Host Staging Buffers
        std::vector<double> m_hostPosX, m_hostPosY, m_hostPosZ;
        std::vector<double> m_hostVelX, m_hostVelY, m_hostVelZ;
        std::vector<double> m_hostMass, m_hostRadius;
        std::vector<double> m_hostDensity, m_hostCoherence, m_hostPhryll;
        std::vector<double> m_hostAccelScale;
        std::vector<double> m_hostPressureRatio, m_hostProperTime;
    };

} // namespace umgebung::models::unified
