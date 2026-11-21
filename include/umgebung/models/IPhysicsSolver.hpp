/**
 * @file IPhysicsSolver.hpp
 * @brief Pluggable polymorphism boundary decoupling `core::Application`,
 *        `ModelRegistry`, and CLI/GUI runners from concrete physics models
 *        across all three architectural tiers (`models/mainstream/`,
 *        `models/thinkers/`, `models/unified/`) and all three explicit
 *        `SimulationDomain`s (`NBodyDynamics`, `FieldGrid2D`, `RadialAndScaleProfile`).
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

namespace umgebung::models {

    /**
     * @struct FieldGrid2DSnapshot
     * @brief Host-accessible 2D spatial field grid snapshot produced by
     *        `SimulationDomain::FieldGrid2D` GPU solvers for live heatmap rendering,
     *        A/B difference maps, probe waveform tracking, and headless JSON export.
     */
    struct FieldGrid2DSnapshot {
        /** @brief Grid cell width $N_x$. */
        std::uint32_t width{0U};

        /** @brief Grid cell height $N_y$. */
        std::uint32_t height{0U};

        /** @brief Physical half-extent $L$ (coordinates span $[-L, +L] \times [-L, +L]$). */
        double domainHalfExtent{2.0};

        /** @brief Row-major (`y * width + x`) scalar field values visualized in the heatmap. */
        std::vector<double> scalarField;

        /** @brief Minimum value across `scalarField`. */
        double minVal{0.0};

        /** @brief Maximum value across `scalarField`. */
        double maxVal{0.0};

        /** @brief Spatial L2 norm $\sqrt{\frac{1}{N_x N_y} \sum_{x,y} f(x,y)^2}$. */
        double l2Norm{0.0};

        /** @brief Peak absolute magnitude $\max_{x,y} |f(x,y)|$. */
        double peakAbsVal{0.0};

        /** @brief Instantaneous scalar probe reading at the cavity center / sensor for time-series waveforms. */
        double probeSampleValue{0.0};

        /** @brief Human-readable quantity label (e.g., `"|S(x,y)| Poynting Flux"`). */
        std::string fieldLabel;
    };

    /**
     * @struct RadialProfileSnapshot
     * @brief 1D radial horizon or logarithmic multi-scale analytical profile
     *        produced by `SimulationDomain::RadialAndScaleProfile` solvers.
     */
    struct RadialProfileSnapshot {
        /** @brief Sampled independent coordinates $x_k$ (e.g., radius $r$ in meters or $r/R_s$). */
        std::vector<double> xValues;

        /** @brief Primary theoretical curve $y_1(x_k)$ (e.g., $m_h(r)$ or $P(r)/P_0$). */
        std::vector<double> primaryYValues;

        /** @brief Secondary theoretical curve $y_2(x_k)$ (e.g., $m_p'(r)$ or $c'(r)/c$). */
        std::vector<double> secondaryYValues;

        /** @brief Axis & legend labels for CLI export and interactive plotting. */
        std::string xLabel;
        std::string primaryYLabel;
        std::string secondaryYLabel;

        /** @brief True when horizontal/vertical axes should be rendered logarithmically ($\log_{10}$). */
        bool logarithmicX{false};
        bool logarithmicY{false};
    };

    /**
     * @struct SolverDiagnostics
     * @brief Unified telemetry record reporting universal conservation/field
     *        invariants, Mainstream NFW halo energy, and verbatim per-Thinker /
     *        Unified field diagnostics across all `SimulationDomain`s.
     */
    struct SolverDiagnostics {
        scene::ModelId modelId{scene::ModelId::MainstreamNewtonian};
        scene::SimulationDomain domain{scene::SimulationDomain::NBodyDynamics};
        std::uint64_t stepCount{0ULL};
        double simulationTime{0.0};

        double kineticEnergy{0.0};
        double potentialEnergy{0.0};
        double totalEnergy{0.0};
        double initialTotalEnergy{0.0};
        double relativeEnergyDrift{0.0};

        scene::Vec3d totalLinearMomentum{};
        scene::Vec3d totalAngularMomentum{};
        scene::Vec3d centerOfMass{};

        // Tier 1 NFW Halo Telemetry
        double nfwHaloPotentialEnergy{0.0};

        // FieldGrid2D Telemetry
        double fieldEnergy{0.0};
        double fieldL2Norm{0.0};
        double fieldPeakValue{0.0};
        double fieldProbeValue{0.0};
        double effectiveCflNumber{0.0};

        // Tier 2 & Tier 3 Thinker / Unified Field Telemetry
        bool hasUnifiedTelemetry{false};
        double meanVacuumPressureRatio{1.0};
        double meanLocalLightSpeedRatio{1.0};
        double meanLocalTimeRate{1.0};
        double meanLocalProperTime{0.0};
        double meanHolographicPhi{0.0};
        double harameinSpacetimeTorqueOmegaZ{0.0};
        double paisMaxPoyntingFlux{0.0};
        double paisInertialMassFactor{1.0};
        double totalPhryllFlux{0.0};
        double vogelCrystalSaturationRatio{0.0};
        bool vogelCrystalSinging{false};

        // Tier 2 John Titor C204 Gravity Distortion Time Displacement Unit Telemetry
        /** @brief True when the active solver populates C204 operational & phenomenological metric fields. */
        bool hasC204Telemetry{false};
        /** @brief Human-readable operational state (`Standby`, `VGL Baseline Lock`, `Ramp-Up`, `Temporal Transit`, etc.). */
        std::string c204OperationalState;
        /** @brief Active matrix power setting in percent (`50`, `75`, `100`). */
        double c204PowerLevelPercent{0.0};
        /** @brief Active electron injection charge in Coulombs (`126k / 201.6k / 288k C`). */
        double c204InjectedChargeCoulombs{0.0};
        /** @brief Cumulative electrical charge expended (drive + VGL pulses) in Coulombs. */
        double c204ExpendedChargeCoulombs{0.0};
        /** @brief Instantaneous wave penetration rate in external years per pilot hour (`3.4 / 6.2 / 10.0`). */
        double c204PenetrationRateYrsPerHr{0.0};
        /** @brief Dimensionless temporal translation velocity $v_t = |dt_{\text{ext}} / d\tau_{\text{pilot}}|$ (`87,660 s/s` at 100%). */
        double c204TimeVelocitySecPerSec{0.0};
        /** @brief Current external worldline calendar year (e.g. `2036.0 -> 1975.0`). */
        double c204CurrentWorldlineYear{2036.0};
        /** @brief Signed temporal displacement elapsed from departure in years. */
        double c204ElapsedDisplacedYears{0.0};
        /** @brief Cumulative estimated worldline divergence percentage (`~1% / 60 yrs` + VGL/Borda drift). */
        double c204WorldlineDivergencePct{0.0};
        /** @brief Sustained cabin gravitational acceleration toward the C204 unit in Earth G's (`1.5–2.0 G`). */
        double c204CabinGForce{0.0};
        /** @brief Exterior vehicle surface temperature in degrees Fahrenheit (`100–120 F` in transit). */
        double c204ExteriorTempF{72.0};
        /** @brief Effective 3+1 ADM metric $g_{00}^{\text{eff}}$ evaluated at the pilot cabin probe. */
        double c204MetricG00AtCabin{-1.0};
        /** @brief Static-observer metric gradient $|\nabla g_{00}^{\text{local}}| = 2a/c^2$ at the pilot cabin ($\text{m}^{-1}$). */
        double c204MetricGradG00AtCabin{0.0};
        /** @brief Optical horizon state label (`Normal`, `Rainbow Lensing + UV Burst`, `Total Blackout`). */
        std::string c204OpticalPhase;
        /** @brief Completed 240-second VGL baseline gravity pulse checks. */
        std::uint64_t c204VglCheckCount{0ULL};
        /** @brief Seconds remaining until the next 240-second VGL gravity sampling pulse. */
        double c204VglNextCheckCountdownSec{240.0};
        /** @brief Latest triple-computer Borda consensus spread across active gravity sensors. */
        double c204BordaConsensusSpread{0.0};
        /** @brief Trapped air-pocket cabin oxygen remaining percentage (`0..100%`). */
        double c204CabinOxygenPct{100.0};
        /** @brief True when the cabin carbon sensor flags hazardous air quality. */
        bool c204CarbonSensorAlert{false};
        /** @brief True during the 1-second ramp-up UV radiation burst (sunglasses / eyes closed required). */
        bool c204UvBurstActive{false};
        /** @brief True once the optical trapping horizon closes into total exterior blackout. */
        bool c204OpticalBlackoutActive{false};
        /** @brief True if Automatic Origin Return / dynamic abort was triggered. */
        bool c204OriginReturnTriggered{false};
        /** @brief Reason string if Automatic Origin Return was triggered. */
        std::string c204AbortReason;
        /** @brief True when traveling forward in time (`targetDisplacementYears > 0`, lower energy Positive Time mode). */
        bool c204IsForwardTravel{false};
        /** @brief Count of GPU 2D grid cells inside the Negative Time Event Horizon ($g_{00}^{\text{eff}} > 0$, time reversal / CTC core). */
        std::uint64_t c204NegativeHorizonCells{0ULL};
        /** @brief Count of GPU 2D grid cells along the Null Time Event Horizon / Cancellation Shell ($g_{00}^{\text{eff}} \approx 0$ or zero-crossing). */
        std::uint64_t c204NullHorizonCells{0ULL};
        /** @brief Count of GPU 2D grid cells inside the active Positive Time Field ($g_{00}^{\text{eff}} < 0$ inside the `22 m x 6 m` envelope). */
        std::uint64_t c204PositiveHorizonCells{0ULL};
        /** @brief Kerr equatorial prograde circular photon orbit radius $r_1 / M$ (Pohlman `US20060073976A1`). */
        double c204KerrProgradePhotonR{0.0};
        /** @brief Kerr equatorial retrograde circular photon orbit radius $r_2 / M$ (Pohlman `US20060073976A1`). */
        double c204KerrRetrogradePhotonR{0.0};

        std::string solverSummary;
    };

    /**
     * @class IPhysicsSolver
     * @brief Abstract runtime interface implemented by every GPU/analytical
     *        solver across `NBodyDynamics`, `FieldGrid2D`, and `RadialAndScaleProfile`.
     */
    class IPhysicsSolver {
    public:
        virtual ~IPhysicsSolver() = default;

        virtual void SetStreamIndex(std::size_t streamIndex) noexcept {
            (void)streamIndex;
        }

        [[nodiscard]] virtual bool Initialize(
            compute::CUDAContext& context,
            const scene::UniverseScene& scene
        ) = 0;

        virtual void Shutdown() noexcept = 0;
        [[nodiscard]] virtual bool Step(double dt) = 0;
        [[nodiscard]] virtual bool StepMultiple(std::uint64_t stepCount, double dt) = 0;
        [[nodiscard]] virtual bool Synchronize() const noexcept = 0;
        [[nodiscard]] virtual bool SyncToHost(std::vector<scene::BodyEntity>& outBodies) = 0;
        [[nodiscard]] virtual bool ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) = 0;

        [[nodiscard]] virtual bool GetLocalProperTimes(std::vector<double>& outProperTimes) {
            outProperTimes.assign(GetBodyCount(), GetSimulationTime());
            return true;
        }

        /** @brief Returns the `SimulationDomain` implemented by this solver instance. */
        [[nodiscard]] virtual scene::SimulationDomain GetDomain() const noexcept {
            return scene::SimulationDomain::NBodyDynamics;
        }

        /** @brief Downloads a 2D spatial field snapshot (valid when `GetDomain() == FieldGrid2D`). */
        [[nodiscard]] virtual bool FetchFieldGrid2DSnapshot(FieldGrid2DSnapshot& outSnapshot) {
            (void)outSnapshot;
            return false;
        }

        /** @brief Evaluates/copies the 1D radial or multi-scale profile (valid when `GetDomain() == RadialAndScaleProfile`). */
        [[nodiscard]] virtual bool FetchRadialProfileSnapshot(RadialProfileSnapshot& outSnapshot) {
            (void)outSnapshot;
            return false;
        }

        [[nodiscard]] virtual bool IsInitialized() const noexcept = 0;
        [[nodiscard]] virtual scene::ModelId GetModelId() const noexcept = 0;
        [[nodiscard]] virtual const char* GetSolverName() const noexcept = 0;
        [[nodiscard]] virtual scene::PhysicsProfileType GetProfileType() const noexcept = 0;
        [[nodiscard]] virtual scene::IntegratorType GetIntegratorType() const noexcept = 0;
        [[nodiscard]] virtual std::size_t GetBodyCount() const noexcept = 0;
        [[nodiscard]] virtual std::uint64_t GetStepCount() const noexcept = 0;
        [[nodiscard]] virtual double GetSimulationTime() const noexcept = 0;
    };

} // namespace umgebung::models
