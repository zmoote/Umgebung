/**
 * @file RadialScaleProfileSolver.hpp
 * @brief Analytical 1D radial-horizon & multi-scale logarithmic sweep solver
 *        implementing `IPhysicsSolver` for `SimulationDomain::RadialAndScaleProfile`.
 *
 * @details
 * Supports controlled A/B comparison of exact theoretical curves across scales
 * without forcing subatomic/cosmological analytical formulas into N-body orbits:
 *   1. **`Mainstream::SchwarzschildGR`:**
 *        - `"HolographicMassScale"`: Classical Schwarzschild black-hole mass
 *          $M_s(r) = c^2 r / (2G)$ vs. CODATA proton mass across $r \in [10^{-16}, 10^{26}]\,\text{m}$.
 *        - `"VacuumHorizon"`: Classical GR Schwarzschild metric factor $g_{tt}(r) = 1 - R_s/r$
 *          and escape velocity $v_{\text{esc}}(r)/c = \sqrt{R_s/r}$ (event horizon at $r = R_s$).
 *   2. **`Thinker::Haramein` (`Haramein342013PRRI3363.pdf`):**
 *        - `"HolographicMassScale"`: Verbatim PSU holographic Schwarzschild mass
 *          $m_h(r) = (\mathcal{R}/\eta) m_\ell$ AND holographic proton/rest mass
 *          $m_p'(r) = 2\phi(r) m_\ell$ across 42 orders of magnitude (reproducing Cygnus X-1
 *          and the muonic proton rest mass $1.6714 \times 10^{-27}\,\text{kg}$ at $0.84184\,\text{fm}$).
 *   3. **`Thinker::Essonne` (`8052.pdf` Eq. 21, 27, 35, 39, 43):**
 *        - `"VacuumHorizon"`: Piecewise interior ($r \le R$, Eq. 35 showing $P(0)/P_0 = 0$
 *          at $R = \frac{3}{2} R_s$) and exterior ($r > R$, Eq. 21) vacuum pressure $P(r)/P_0$,
 *          plus local speed of light $c'(r)/c = \sqrt{\max(P/P_0, 0)}$ intersecting escape
 *          velocity at Essonne's true light-escape horizon $r = 2 R_s$.
 */

#pragma once

#include "umgebung/models/IPhysicsSolver.hpp"

#include <string>
#include <vector>

namespace umgebung::models {

    /**
     * @class RadialScaleProfileSolver
     * @brief Exact analytical curve & multi-scale sweep solver for
     *        `SimulationDomain::RadialAndScaleProfile`.
     */
    class RadialScaleProfileSolver final : public IPhysicsSolver {
    public:
        explicit RadialScaleProfileSolver(scene::ModelId modelId);
        ~RadialScaleProfileSolver() override = default;

        [[nodiscard]] bool Initialize(
            compute::CUDAContext& context,
            const scene::UniverseScene& scene
        ) override;

        void Shutdown() noexcept override;
        [[nodiscard]] bool Step(double dt) override;
        [[nodiscard]] bool StepMultiple(std::uint64_t stepCount, double dt) override;
        [[nodiscard]] bool Synchronize() const noexcept override { return m_initialized; }
        [[nodiscard]] bool SyncToHost(std::vector<scene::BodyEntity>& outBodies) override;
        [[nodiscard]] bool ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) override;
        [[nodiscard]] bool FetchRadialProfileSnapshot(RadialProfileSnapshot& outSnapshot) override;

        [[nodiscard]] scene::SimulationDomain GetDomain() const noexcept override {
            return scene::SimulationDomain::RadialAndScaleProfile;
        }

        [[nodiscard]] bool IsInitialized() const noexcept override { return m_initialized; }
        [[nodiscard]] scene::ModelId GetModelId() const noexcept override { return m_modelId; }
        [[nodiscard]] const char* GetSolverName() const noexcept override { return m_solverName.c_str(); }
        [[nodiscard]] scene::PhysicsProfileType GetProfileType() const noexcept override { return m_profileType; }
        [[nodiscard]] scene::IntegratorType GetIntegratorType() const noexcept override {
            return scene::IntegratorType::VelocityVerlet;
        }
        [[nodiscard]] std::size_t GetBodyCount() const noexcept override { return 0U; }
        [[nodiscard]] std::uint64_t GetStepCount() const noexcept override { return m_stepCount; }
        [[nodiscard]] double GetSimulationTime() const noexcept override { return m_simulationTime; }

    private:
        void EvaluateProfile();

        scene::ModelId m_modelId{scene::ModelId::MainstreamSchwarzschildGR};
        scene::PhysicsProfileType m_profileType{scene::PhysicsProfileType::Mainstream};
        std::string m_solverName;
        scene::RadialScaleConfig m_config{};
        scene::TitorC204Params m_c204Params{};

        bool m_initialized{false};
        std::uint64_t m_stepCount{0ULL};
        double m_simulationTime{0.0};

        RadialProfileSnapshot m_snapshot{};
        double m_meanPressureRatio{1.0};
        double m_meanLightSpeedRatio{1.0};
        double m_meanHolographicPhi{0.0};
    };

} // namespace umgebung::models
