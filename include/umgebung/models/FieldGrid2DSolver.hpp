/**
 * @file FieldGrid2DSolver.hpp
 * @brief GPU-accelerated FP64 CUDA `SimulationDomain::FieldGrid2D` solver
 *        pipeline supporting:
 *          - `Mainstream::MaxwellEM`   (2D TMz Yee-lattice FDTD Maxwell cavity baseline),
 *          - `Thinker::Pais`           (stationary HEEMFG spinning/vibrating cavity Poynting flux `US10144532.pdf`),
 *          - `Thinker::VogelWillis`    (stationary `51.84°/60°` quartz crystal + `4096 Hz` `180°` orthogonal torsion coils `PhryllGenInstructions.pdf`),
 *          - `Thinker::Essonne`        (2D spatial vacuum pressure `P(x,y)/P_0` & local `c'(x,y)/c` field `8052.pdf`),
 *          - `Unified::Synthesis`      (composed vacuum-refractive + Pais + Vogel/Willis 2D GPU cavity).
 *
 * @details
 * Eliminates the physical category error of modeling electromagnetic cavities
 * and benchtop quartz torsion crystals as orbiting point masses. Enforces the
 * strict 2D FDTD Courant-Friedrichs-Lewy (CFL) stability condition
 * $S_{\text{CFL}} = c\,\Delta t / \Delta x \le 0.45 < 1/\sqrt{2}$
 * (formally verified in `formal/Umgebung/Numerical/FdtdStability.lean`).
 *
 * @note Zero GUI dependencies; compiles identically in Desktop GUI and Headless HPC modes.
 */

#pragma once

#include "umgebung/compute/CUDAContext.hpp"
#include "umgebung/compute/DeviceBuffer.hpp"
#include "umgebung/models/IPhysicsSolver.hpp"
#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace umgebung::models {

    /**
     * @class FieldGrid2DSolver
     * @brief FP64 CUDA 2D spatial field & TMz Yee-grid FDTD solver implementing
     *        `IPhysicsSolver` for `SimulationDomain::FieldGrid2D`.
     */
    class FieldGrid2DSolver final : public IPhysicsSolver {
    public:
        /**
         * @brief Constructs a `FieldGrid2DSolver` configured to execute `modelId`
         *        on `SimulationDomain::FieldGrid2D`.
         *
         * @param modelId One of `MainstreamMaxwellEM`, `ThinkerPais`,
         *                `ThinkerVogelWillis`, `ThinkerEssonne`, or `UnifiedSynthesis`.
         */
        explicit FieldGrid2DSolver(scene::ModelId modelId);
        ~FieldGrid2DSolver() override;

        FieldGrid2DSolver(const FieldGrid2DSolver&) = delete;
        FieldGrid2DSolver& operator=(const FieldGrid2DSolver&) = delete;
        FieldGrid2DSolver(FieldGrid2DSolver&&) = delete;
        FieldGrid2DSolver& operator=(FieldGrid2DSolver&&) = delete;

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
        [[nodiscard]] bool FetchFieldGrid2DSnapshot(FieldGrid2DSnapshot& outSnapshot) override;

        [[nodiscard]] scene::SimulationDomain GetDomain() const noexcept override {
            return scene::SimulationDomain::FieldGrid2D;
        }

        [[nodiscard]] bool IsInitialized() const noexcept override { return m_initialized; }
        [[nodiscard]] scene::ModelId GetModelId() const noexcept override { return m_modelId; }
        [[nodiscard]] const char* GetSolverName() const noexcept override { return m_solverName.c_str(); }
        [[nodiscard]] scene::PhysicsProfileType GetProfileType() const noexcept override { return m_profileType; }
        [[nodiscard]] scene::IntegratorType GetIntegratorType() const noexcept override { return m_integrator; }
        [[nodiscard]] std::size_t GetBodyCount() const noexcept override { return m_stationaryAnchors.size(); }
        [[nodiscard]] std::uint64_t GetStepCount() const noexcept override { return m_stepCount; }
        [[nodiscard]] double GetSimulationTime() const noexcept override { return m_simulationTime; }

    private:
        [[nodiscard]] bool ComputeSpatialEssonneField();
        [[nodiscard]] bool DownloadVisualField();

        scene::ModelId m_modelId{scene::ModelId::MainstreamMaxwellEM};
        scene::PhysicsProfileType m_profileType{scene::PhysicsProfileType::Mainstream};
        scene::IntegratorType m_integrator{scene::IntegratorType::VelocityVerlet};
        std::string m_solverName;

        compute::CUDAContext* m_context{nullptr};
        std::size_t m_streamIndex{0U};
        cudaStream_t m_stream{nullptr};
        bool m_initialized{false};

        scene::FieldGrid2DConfig m_gridConfig{};
        scene::UnifiedFieldParams m_unifiedParams{};
        double m_gravitationalConstant{1.0};
        std::vector<scene::BodyEntity> m_stationaryAnchors;

        std::uint32_t m_width{128U};
        std::uint32_t m_height{128U};
        std::size_t m_cellCount{0U};
        double m_dx{0.03125};
        double m_effectiveCfl{0.0};

        // Cached Pais & Vogel/Willis verbatim scalars.
        double m_paisMaxPoyntingFlux{0.0};
        double m_paisInertialMassFactor{1.0};
        double m_paisAmplification{1.0};
        double m_vogelGeomResonance{1.0};
        double m_vogelDrivePower{1.0};
        double m_totalCrystalPhryll{1.0};
        double m_cumulativeCrystalEnergy{0.0};

        std::uint64_t m_stepCount{0ULL};
        double m_simulationTime{0.0};

        // GPU FP64 2D Grid Buffers (Yee TMz FDTD: Ez, Hx, Hy + relative permittivity invEps + visual scalar).
        compute::DeviceBuffer<double> m_ez;
        compute::DeviceBuffer<double> m_hx;
        compute::DeviceBuffer<double> m_hy;
        compute::DeviceBuffer<double> m_invEps;
        compute::DeviceBuffer<double> m_visualScalar;

        // Reusable Host Staging Buffers
        std::vector<double> m_hostVisual;
        std::vector<double> m_hostEz;
        std::vector<double> m_hostHx;
        std::vector<double> m_hostHy;
    };

} // namespace umgebung::models
