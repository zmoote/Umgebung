/**
 * @file HarameinSolver.hpp
 * @brief Tier 2 Standalone Thinker Module (`Thinker::Haramein`): FP64 CUDA
 *        solver implementing **Nassim Haramein**'s holographic Planck Spherical
 *        Unit (PSU) tiling (`Haramein342013PRRI3363.pdf`) and Haramein-Rauscher
 *        spacetime-torque Coriolis dynamics via symplectic Strang splitting.
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
     * @class HarameinSolver
     * @brief Standalone FP64 CUDA solver for `ModelId::ThinkerHaramein`.
     */
    class HarameinSolver final : public IPhysicsSolver {
    public:
        static constexpr int kTileSize = 256;

        HarameinSolver() = default;
        ~HarameinSolver() override;

        HarameinSolver(const HarameinSolver&) = delete;
        HarameinSolver& operator=(const HarameinSolver&) = delete;
        HarameinSolver(HarameinSolver&&) = delete;
        HarameinSolver& operator=(HarameinSolver&&) = delete;

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
            return scene::ModelId::ThinkerHaramein;
        }

        [[nodiscard]] const char* GetSolverName() const noexcept override {
            return "Thinker::Haramein (Holographic PSU & Spacetime Torque, FP64 CUDA)";
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
        [[nodiscard]] bool LaunchGravityKernel();
        [[nodiscard]] bool DownloadStateToStaging();

        compute::CUDAContext* m_context{nullptr};
        std::size_t m_streamIndex{0U};
        cudaStream_t m_stream{nullptr};

        bool m_initialized{false};
        std::size_t m_bodyCount{0U};
        scene::IntegratorType m_integrator{scene::IntegratorType::VelocityVerlet};
        double m_gravitationalConstant{1.0};
        double m_softeningLength{0.0};
        scene::HarameinParams m_harameinParams{};

        std::uint64_t m_stepCount{0ULL};
        double m_simulationTime{0.0};
        double m_initialTotalEnergy{0.0};

        compute::DeviceBuffer<double> m_posX, m_posY, m_posZ;
        compute::DeviceBuffer<double> m_velX, m_velY, m_velZ;
        compute::DeviceBuffer<double> m_accX, m_accY, m_accZ;
        compute::DeviceBuffer<double> m_mass;

        std::vector<double> m_hostPosX, m_hostPosY, m_hostPosZ;
        std::vector<double> m_hostVelX, m_hostVelY, m_hostVelZ;
        std::vector<double> m_hostMass;
        std::vector<double> m_hostRadius;
    };

} // namespace umgebung::models::thinkers
