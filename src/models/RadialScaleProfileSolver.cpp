/**
 * @file RadialScaleProfileSolver.cpp
 * @brief Implementation of `umgebung::models::RadialScaleProfileSolver` for
 *        `SimulationDomain::RadialAndScaleProfile` analytical & multi-scale sweeps.
 */

#include "umgebung/models/RadialScaleProfileSolver.hpp"
#include "umgebung/models/thinkers/EssonneModel.hpp"
#include "umgebung/models/thinkers/HarameinModel.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models {

    RadialScaleProfileSolver::RadialScaleProfileSolver(scene::ModelId modelId)
        : m_modelId(modelId) {
        switch (m_modelId) {
            case scene::ModelId::MainstreamSchwarzschildGR:
                m_profileType = scene::PhysicsProfileType::Mainstream;
                m_solverName = "Mainstream::SchwarzschildGR (Classical GR Metric & Horizon Profile)";
                break;
            case scene::ModelId::ThinkerHaramein:
                m_profileType = scene::PhysicsProfileType::Custom;
                m_solverName = "Thinker::Haramein (PSU Holographic Mass Multi-Scale Sweep)";
                break;
            case scene::ModelId::ThinkerEssonne:
                m_profileType = scene::PhysicsProfileType::Custom;
                m_solverName = "Thinker::Essonne (Vacuum Pressure P/P0 & 2*Rs Horizon Profile)";
                break;
            default:
                m_profileType = scene::PhysicsProfileType::Mainstream;
                m_solverName = "Radial & Scale Analytical Profile Solver";
                break;
        }
    }

    bool RadialScaleProfileSolver::Initialize(
        compute::CUDAContext& context,
        const scene::UniverseScene& scene
    ) {
        (void)context;
        Shutdown();

        std::string err;
        if (!scene.Validate(err)) {
            spdlog::error("RadialScaleProfileSolver::Initialize failed: {}", err);
            return false;
        }

        m_config = scene.physicsProfile.radialScale;
        if (m_config.sampleCount < 8U || m_config.rangeMax <= m_config.rangeMin) {
            spdlog::error("RadialScaleProfileSolver::Initialize failed: invalid radial_scale bounds.");
            return false;
        }
        if (m_config.logarithmicScale && m_config.rangeMin <= 0.0) {
            spdlog::error("RadialScaleProfileSolver::Initialize failed: logarithmic_scale requires rangeMin > 0.");
            return false;
        }

        m_stepCount = 0ULL;
        m_simulationTime = 0.0;
        EvaluateProfile();
        m_initialized = true;

        spdlog::info(
            "RadialScaleProfileSolver initialized [model={}, sweep={}, samples={}, range=[{:.3g}, {:.3g}], log={}].",
            scene::ModelIdToString(m_modelId),
            m_config.sweepKind,
            m_config.sampleCount,
            m_config.rangeMin,
            m_config.rangeMax,
            m_config.logarithmicScale ? "ON" : "OFF"
        );
        return true;
    }

    void RadialScaleProfileSolver::Shutdown() noexcept {
        m_initialized = false;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;
        m_snapshot = RadialProfileSnapshot{};
    }

    void RadialScaleProfileSolver::EvaluateProfile() {
        using namespace thinkers;

        const std::size_t n = static_cast<std::size_t>(m_config.sampleCount);
        m_snapshot = RadialProfileSnapshot{};
        m_snapshot.xValues.resize(n);
        m_snapshot.primaryYValues.resize(n);
        m_snapshot.secondaryYValues.resize(n);
        m_snapshot.xLabel = m_config.xAxisLabel;
        m_snapshot.logarithmicX = m_config.logarithmicScale;

        const bool isMassScale = (m_config.sweepKind == "HolographicMassScale");
        m_snapshot.logarithmicY = isMassScale;

        const double logMin = m_config.logarithmicScale ? std::log10(m_config.rangeMin) : 0.0;
        const double logMax = m_config.logarithmicScale ? std::log10(m_config.rangeMax) : 0.0;

        double sumP = 0.0;
        double sumC = 0.0;
        double sumPhi = 0.0;

        for (std::size_t k = 0U; k < n; ++k) {
            const double frac = (n > 1U) ? (static_cast<double>(k) / static_cast<double>(n - 1U)) : 0.0;
            const double x = m_config.logarithmicScale
                ? std::pow(10.0, logMin + (frac * (logMax - logMin)))
                : (m_config.rangeMin + (frac * (m_config.rangeMax - m_config.rangeMin)));
            m_snapshot.xValues[k] = x;

            if (isMassScale) {
                // Multi-scale sweep: radius r in meters -> mass in kg.
                sumPhi += ComputeHolographicRatioPhi(x);
                if (m_modelId == scene::ModelId::ThinkerHaramein) {
                    // Haramein (2013) Eq. 9-13 (m_h) & Eq. 24-27 (m_p' = 2*phi*m_l).
                    m_snapshot.primaryYValues[k] = ComputeHolographicSchwarzschildMass(x);
                    m_snapshot.secondaryYValues[k] = ComputeHolographicProtonMass(x);
                    m_snapshot.primaryYLabel = "Haramein m_h(r) [kg]";
                    m_snapshot.secondaryYLabel = "Haramein m_p'(r) = 2 phi m_l [kg]";
                } else {
                    // Mainstream classical Schwarzschild mass M_s(r) = c^2 r / (2G) & constant CODATA proton mass.
                    const double c = constants::kSpeedOfLightSI;
                    const double G = constants::kGravitationalConstantSI;
                    m_snapshot.primaryYValues[k] = (c * c * x) / (2.0 * G);
                    m_snapshot.secondaryYValues[k] = constants::kCodataProtonMassSI;
                    m_snapshot.primaryYLabel = "GR Schwarzschild M_s(r) [kg]";
                    m_snapshot.secondaryYLabel = "CODATA Proton Mass m_p [kg]";
                }
            } else {
                // Radial horizon sweep: x = r / R_s (or physical r with referenceSchwarzschildRadius).
                const double Rs = m_config.referenceSchwarzschildRadius;
                const double Rbody = m_config.referenceBodyRadius;

                if (m_modelId == scene::ModelId::ThinkerEssonne) {
                    // Verbatim Chris Essonne 8052.pdf Eq. (21), (27), (35), (39), (43):
                    // Interior pressure vanishes at r=0 when R = 1.5*Rs; c'(r) meets v_L(r) at r = 2*Rs!
                    const double pRatio = (x <= Rbody)
                        ? ComputeEssonneInteriorPressureRatio(x, Rbody, Rs)
                        : ComputeEssonneVacuumPressureRatio(x, Rs);
                    const double cRatio = (pRatio > 0.0) ? std::sqrt(pRatio) : 0.0;
                    m_snapshot.primaryYValues[k] = pRatio;
                    m_snapshot.secondaryYValues[k] = cRatio;
                    m_snapshot.primaryYLabel = "Essonne P(r)/P_0 (Interior+Exterior)";
                    m_snapshot.secondaryYLabel = "Essonne c'(r)/c = sqrt(P/P_0)";
                    sumP += std::max(0.0, pRatio);
                    sumC += cRatio;
                } else {
                    // Mainstream Schwarzschild GR: g_tt = 1 - R_s / r outside R_s, clamped to 0 inside horizon r <= R_s,
                    // and classical escape velocity ratio v_esc / c = sqrt(R_s / r) (clamped at 1.0 at r = R_s).
                    const double gtt = (x > Rs) ? (1.0 - (Rs / x)) : 0.0;
                    const double vEscRatio = (x > Rs) ? std::sqrt(Rs / x) : 1.0;
                    m_snapshot.primaryYValues[k] = gtt;
                    m_snapshot.secondaryYValues[k] = vEscRatio;
                    m_snapshot.primaryYLabel = "GR Metric g_tt = 1 - R_s/r";
                    m_snapshot.secondaryYLabel = "Escape Speed v_esc(r)/c";
                    sumP += gtt;
                    sumC += (gtt > 0.0) ? std::sqrt(gtt) : 0.0;
                }
            }
        }

        const double invN = (n > 0U) ? (1.0 / static_cast<double>(n)) : 0.0;
        m_meanPressureRatio = isMassScale ? 1.0 : (sumP * invN);
        m_meanLightSpeedRatio = isMassScale ? 1.0 : (sumC * invN);
        m_meanHolographicPhi = sumPhi * invN;
    }

    bool RadialScaleProfileSolver::Step(double dt) {
        if (!m_initialized || !std::isfinite(dt) || dt == 0.0) {
            return false;
        }
        ++m_stepCount;
        m_simulationTime += dt;
        return true;
    }

    bool RadialScaleProfileSolver::StepMultiple(std::uint64_t stepCount, double dt) {
        if (!m_initialized || !std::isfinite(dt) || dt == 0.0) {
            return false;
        }
        m_stepCount += stepCount;
        m_simulationTime += static_cast<double>(stepCount) * dt;
        return true;
    }

    bool RadialScaleProfileSolver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
        outBodies.clear();
        return m_initialized;
    }

    bool RadialScaleProfileSolver::FetchRadialProfileSnapshot(RadialProfileSnapshot& outSnapshot) {
        if (!m_initialized) {
            return false;
        }
        outSnapshot = m_snapshot;
        return true;
    }

    bool RadialScaleProfileSolver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        if (!m_initialized) {
            return false;
        }
        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.modelId = m_modelId;
        outDiagnostics.domain = scene::SimulationDomain::RadialAndScaleProfile;
        outDiagnostics.stepCount = m_stepCount;
        outDiagnostics.simulationTime = m_simulationTime;
        outDiagnostics.hasUnifiedTelemetry = (m_modelId != scene::ModelId::MainstreamSchwarzschildGR);
        outDiagnostics.meanVacuumPressureRatio = m_meanPressureRatio;
        outDiagnostics.meanLocalLightSpeedRatio = m_meanLightSpeedRatio;
        outDiagnostics.meanHolographicPhi = m_meanHolographicPhi;
        outDiagnostics.solverSummary = m_solverName;
        return true;
    }

} // namespace umgebung::models
