/**
 * @file EssonneSolver.hpp
 * @brief Tier 2 Standalone Thinker Module (`Thinker::Essonne`): FP64 CUDA solver
 *        implementing **Chris Essonne**'s vacuum energy density gradient gravity
 *        (`8052.pdf`) and 2D spiral-galaxy aether-vortex model (`10100.pdf`).
 *
 * @details
 * Runs Chris Essonne's equations in complete isolation from other Thinkers
 * (zero Haramein torque, zero Pais inertial reduction, zero Danaan Love, zero
 * Vogel/Willis Phryll), allowing clean A/B comparison against
 * `Mainstream::Newtonian` and `Mainstream::DarkMatterNFW`.
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/compute/DeviceBuffer.hpp"
#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace umgebung::models::thinkers {

    /**
     * @class EssonneSolver
     * @brief Standalone FP64 CUDA solver for `ModelId::ThinkerEssonne`.
     */
    class EssonneSolver final : public IPhysicsSolver {
    public:
        static constexpr int kTileSize = 256;

        EssonneSolver() = default;
        ~EssonneSolver() override;

        EssonneSolver(const EssonneSolver&) = delete;
        EssonneSolver& operator=(const EssonneSolver&) = delete;
        EssonneSolver(EssonneSolver&&) = delete;
        EssonneSolver& operator=(EssonneSolver&&) = delete;

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

        /** @brief Downloads per-body Essonne vacuum pressure ratios $P_i/P_0$ (`8052.pdf` Eq. 21). */
        [[nodiscard]] bool GetVacuumPressureRatios(std::vector<double>& outPressureRatios);

        [[nodiscard]] bool IsInitialized() const noexcept override { return m_initialized; }

        [[nodiscard]] scene::ModelId GetModelId() const noexcept override {
            return scene::ModelId::ThinkerEssonne;
        }

        [[nodiscard]] const char* GetSolverName() const noexcept override {
            return "Thinker::Essonne (Vacuum Gradient & 2D Aether Vortex, FP64 CUDA)";
        }

        [[nodiscard]] scene::PhysicsProfileType GetProfileType() const noexcept override {
            return scene::PhysicsProfileType::Custom;
        }

        [[nodiscard]] scene::IntegratorType GetIntegratorType() const noexcept override {
            return m_integrator;
        }

        [[nodiscard]] std::size_t GetBodyCount() const noexcept override { return m_bodyCount; }
        [[nodiscard]] std::uint64_t GetStepCount() const noexcept override { return m_stepCount; }
        [[nodiscard]] double GetSimulationTime() const noexcept override { return m_simulationTime; }

    private:
        [[nodiscard]] bool LaunchEssonneKernel();
        [[nodiscard]] bool DownloadStateToStaging();

        compute::CUDAContext* m_context{nullptr};
        std::size_t m_streamIndex{0U};
        cudaStream_t m_stream{nullptr};

        bool m_initialized{false};
        std::size_t m_bodyCount{0U};
        scene::IntegratorType m_integrator{scene::IntegratorType::VelocityVerlet};
        double m_gravitationalConstant{1.0};
        double m_softeningLength{0.0};
        double m_refLightSpeed{299792458.0};
        bool m_aetherVortexEnabled{false};
        double m_vortexRc{1.0};
        double m_vortexH{1.0};
        double m_vortexGz{0.0};

        std::uint64_t m_stepCount{0ULL};
        double m_simulationTime{0.0};
        double m_initialTotalEnergy{0.0};

        compute::DeviceBuffer<double> m_posX, m_posY, m_posZ;
        compute::DeviceBuffer<double> m_velX, m_velY, m_velZ;
        compute::DeviceBuffer<double> m_accX, m_accY, m_accZ;
        compute::DeviceBuffer<double> m_mass;
        compute::DeviceBuffer<double> m_vacuumPressureRatio;
        compute::DeviceBuffer<double> m_localProperTime;

        std::vector<double> m_hostPosX, m_hostPosY, m_hostPosZ;
        std::vector<double> m_hostVelX, m_hostVelY, m_hostVelZ;
        std::vector<double> m_hostMass;
        std::vector<double> m_hostPressureRatio;
        std::vector<double> m_hostProperTime;
    };

} // namespace umgebung::models::thinkers
