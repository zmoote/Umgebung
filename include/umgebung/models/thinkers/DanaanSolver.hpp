/**
 * @file DanaanSolver.hpp
 * @brief Tier 2 Standalone Thinker Module (`Thinker::Danaan`): FP64 CUDA solver
 *        implementing **Elena Danaan**'s Love binding coherence force,
 *        vibrational density octaves, and gravity-bound non-linear local time.
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
     * @class DanaanSolver
     * @brief Standalone FP64 CUDA solver for `ModelId::ThinkerDanaan`.
     */
    class DanaanSolver final : public IPhysicsSolver {
    public:
        static constexpr int kTileSize = 256;

        DanaanSolver() = default;
        ~DanaanSolver() override;

        DanaanSolver(const DanaanSolver&) = delete;
        DanaanSolver& operator=(const DanaanSolver&) = delete;
        DanaanSolver(DanaanSolver&&) = delete;
        DanaanSolver& operator=(DanaanSolver&&) = delete;

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

        [[nodiscard]] bool IsInitialized() const noexcept override { return m_initialized; }

        [[nodiscard]] scene::ModelId GetModelId() const noexcept override {
            return scene::ModelId::ThinkerDanaan;
        }

        [[nodiscard]] const char* GetSolverName() const noexcept override {
            return "Thinker::Danaan (Love Coherence & Non-Linear Time, FP64 CUDA)";
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
        [[nodiscard]] bool LaunchDanaanKernel();
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
        double m_loveBindingCoupling{0.0};
        double m_baseVibrationalDensity{3.0};
        double m_phryllAmplitude{0.0};
        bool m_nonLinearTimeEnabled{true};

        std::uint64_t m_stepCount{0ULL};
        double m_simulationTime{0.0};
        double m_initialTotalEnergy{0.0};

        compute::DeviceBuffer<double> m_posX, m_posY, m_posZ;
        compute::DeviceBuffer<double> m_velX, m_velY, m_velZ;
        compute::DeviceBuffer<double> m_accX, m_accY, m_accZ;
        compute::DeviceBuffer<double> m_mass;
        compute::DeviceBuffer<double> m_vibrationalDensity;
        compute::DeviceBuffer<double> m_coherence;
        compute::DeviceBuffer<double> m_phryllCharge;
        compute::DeviceBuffer<double> m_gravTimeFactor;
        compute::DeviceBuffer<double> m_localProperTime;

        std::vector<double> m_hostPosX, m_hostPosY, m_hostPosZ;
        std::vector<double> m_hostVelX, m_hostVelY, m_hostVelZ;
        std::vector<double> m_hostMass, m_hostDensity, m_hostCoherence, m_hostPhryll;
        std::vector<double> m_hostGravTimeFactor, m_hostProperTime;
    };

} // namespace umgebung::models::thinkers
