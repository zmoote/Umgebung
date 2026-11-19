/**
 * @file PaisSolver.hpp
 * @brief Tier 2 Standalone Thinker Module (`Thinker::Pais`): FP64 CUDA solver
 *        implementing **Salvatore Pais**'s High-Energy Electromagnetic Field
 *        Generator (HEEMFG) & Inertial Mass Reduction Device (`US10144532.pdf`).
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
     * @class PaisSolver
     * @brief Standalone FP64 CUDA solver for `ModelId::ThinkerPais`.
     */
    class PaisSolver final : public IPhysicsSolver {
    public:
        static constexpr int kTileSize = 256;

        PaisSolver() = default;
        ~PaisSolver() override;

        PaisSolver(const PaisSolver&) = delete;
        PaisSolver& operator=(const PaisSolver&) = delete;
        PaisSolver(PaisSolver&&) = delete;
        PaisSolver& operator=(PaisSolver&&) = delete;

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

        [[nodiscard]] bool IsInitialized() const noexcept override { return m_initialized; }

        [[nodiscard]] scene::ModelId GetModelId() const noexcept override {
            return scene::ModelId::ThinkerPais;
        }

        [[nodiscard]] const char* GetSolverName() const noexcept override {
            return "Thinker::Pais (HEEMFG Inertial Mass Reduction, FP64 CUDA)";
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
        [[nodiscard]] bool LaunchPaisAccelKernel();
        [[nodiscard]] bool DownloadStateToStaging();

        compute::CUDAContext* m_context{nullptr};
        std::size_t m_streamIndex{0U};
        cudaStream_t m_stream{nullptr};

        bool m_initialized{false};
        std::size_t m_bodyCount{0U};
        scene::IntegratorType m_integrator{scene::IntegratorType::VelocityVerlet};
        double m_gravitationalConstant{1.0};
        double m_softeningLength{0.0};
        scene::PaisHeemfgParams m_paisParams{};
        double m_maxPoyntingFlux{0.0};
        double m_inertialMassFactor{1.0};
        double m_accelAmplification{1.0};

        std::uint64_t m_stepCount{0ULL};
        double m_simulationTime{0.0};
        double m_initialTotalEnergy{0.0};

        compute::DeviceBuffer<double> m_posX, m_posY, m_posZ;
        compute::DeviceBuffer<double> m_velX, m_velY, m_velZ;
        compute::DeviceBuffer<double> m_accX, m_accY, m_accZ;
        compute::DeviceBuffer<double> m_mass;
        compute::DeviceBuffer<double> m_accelScale;

        std::vector<double> m_hostPosX, m_hostPosY, m_hostPosZ;
        std::vector<double> m_hostVelX, m_hostVelY, m_hostVelZ;
        std::vector<double> m_hostMass;
        std::vector<double> m_hostAccelScale;
    };

} // namespace umgebung::models::thinkers
