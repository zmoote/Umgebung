/**
 * @file TitorC204Solver.cu
 * @brief FP64 CUDA implementation of `umgebung::models::thinkers::TitorC204Solver`
 *        for the **GE Army Model C204 Gravity Distortion Time Displacement Unit**
 *        (`SimulationDomain::FieldGrid2D`).
 *
 * @details
 * Executes a 2D GPU spatial slice ($N_x \times N_y$ cells in IEEE-754 double
 * precision) of the C204 **Effective Phenomenological 3+1 ADM Metric**
 * $g_{\mu\nu}^{\text{eff}}(t, x, y, 0)$ calibrated directly from:
 *   - `C204_Operators_Manual.pdf` (`TMIM-200S/4005`, GE 2034–2035),
 *   - `CompleteTitor.pdf` (John Titor / TimeTravel_0 logs 2000–2001),
 *   - Marlin B. Pohlman `US 2006/0073976 A1` (Van Den Broeck / Alcubierre warp
 *     envelope, Kerr photon orbits, inverse Blandford-Znajek, Geroch topology),
 *   - Frank J. Tipler (1974, Phys. Rev. D 9, 2203) & W. J. van Stockum (1937).
 *
 * Simultaneously advances the C204 onboard triple-computer Borda flight control
 * state machine (`Standby` $\to$ `VglBaselineLock` $\to$ `RampUp` $\to$
 * `TemporalTransit` $\leftrightarrow$ `VglPulseCheck` $\to$ `ArrivalCooldown`,
 * with automatic `OriginReturnAbort` failsafe on VGL gravity anomaly, payload
 * overload $> 2500\,\text{kg}$, or hazardous cabin carbon-sensor alert).
 */

#include "umgebung/models/thinkers/TitorC204Solver.hpp"
#include "umgebung/compute/CudaUtils.hpp"

#include <algorithm>
#include <cmath>

#include <spdlog/spdlog.h>

namespace umgebung::models::thinkers {

    namespace {

        /**
         * @struct C204KernelParams
         * @brief Flat POD parameter block passed by value in CUDA constant memory
         *        to `ComputeTitorC204MetricGrid2DKernel` for zero-overhead GPU evaluation.
         */
        struct C204KernelParams {
            std::uint32_t width;
            std::uint32_t height;
            double halfExtent;
            double simTime;
            double powerFraction;
            double rampFraction;
            double offsetXM;
            double semiAxisA;
            double semiAxisB;
            double semiAxisC;
            double singularitySpacingM;
            double cabinProbeDistM;
            double tiplerWavelengthM;
            double dampingScaleM;
            double wallSteepness;
            double xrayVentConeDeg;
            double tiplerAR;
            bool isForwardTravel;
            double vglPulseEnvelope;
        };

        /**
         * @brief FP64 CUDA kernel evaluating the C204 **Effective Phenomenological
         *        3+1 ADM Metric** across every cell $(i_x, i_y)$ of the 2D equatorial
         *        slice $z = 0$ spanning $[-L, +L] \times [-L, +L]$ meters.
         *
         * @details
         * Each GPU thread maps one grid cell $(i_x, i_y)$ to physical Cartesian
         * coordinates $(x, y, 0)$ relative to the dual-singularity center and
         * evaluates `EvaluateC204EffectiveMetricPoint(...)` in IEEE-754 double precision:
         *   - `visualDistortion[idx]` receives $|1 + g_{00}^{\text{eff}}(x, y, t)|$
         *     (`0` outside the `22 m × 6 m` Van Den Broeck bubble, `1` along the
         *     Null Time Event Horizon, and `> 1` inside the Negative Time core),
         *   - `g00Out[idx]` receives the signed ADM metric component $g_{00}^{\text{eff}}$,
         *   - `accelGOut[idx]` receives local proper gravitational acceleration in G's,
         *   - `horizonZoneOut[idx]` receives `-1` (Negative), `0` (Null), or `+1` (Positive).
         *
         * Thread layout: 2D blocks of `16 x 16 = 256` threads with coalesced row-major
         * global memory writes (`idx = iy * width + ix`).
         */
        __global__ void ComputeTitorC204MetricGrid2DKernel(
            double* __restrict__ visualDistortion,
            double* __restrict__ g00Out,
            double* __restrict__ accelGOut,
            int* __restrict__ horizonZoneOut,
            C204KernelParams kp
        ) {
            const std::uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
            if (ix >= kp.width || iy >= kp.height) {
                return;
            }

            const double dx = (2.0 * kp.halfExtent) / static_cast<double>(kp.width - 1U);
            const double dy = (2.0 * kp.halfExtent) / static_cast<double>(kp.height - 1U);
            const double x = -kp.halfExtent + static_cast<double>(ix) * dx;
            const double y = -kp.halfExtent + static_cast<double>(iy) * dy;

            const C204MetricPointSample sample = EvaluateC204EffectiveMetricPoint(
                x, y, 0.0,
                kp.simTime,
                kp.powerFraction,
                kp.rampFraction,
                kp.offsetXM,
                kp.semiAxisA,
                kp.semiAxisB,
                kp.semiAxisC,
                kp.singularitySpacingM,
                kp.cabinProbeDistM,
                kp.tiplerWavelengthM,
                kp.dampingScaleM,
                kp.wallSteepness,
                kp.xrayVentConeDeg,
                kp.tiplerAR,
                kp.isForwardTravel,
                kp.vglPulseEnvelope
            );

            const std::size_t idx = static_cast<std::size_t>(iy) * kp.width + ix;
            visualDistortion[idx] = sample.distortionMagnitude;
            g00Out[idx] = sample.metricG00;
            accelGOut[idx] = sample.properAccelG;
            horizonZoneOut[idx] = sample.horizonZone;
        }

    } // anonymous namespace

    TitorC204Solver::TitorC204Solver() = default;

    TitorC204Solver::~TitorC204Solver() {
        Shutdown();
    }

    bool TitorC204Solver::Initialize(
        compute::CUDAContext& context,
        const scene::UniverseScene& scene
    ) {
        Shutdown();

        std::string err;
        if (!context.IsInitialized() || !scene.Validate(err)) {
            spdlog::error("TitorC204Solver::Initialize failed: {}", err);
            return false;
        }

        m_context = &context;
        m_stream = context.GetStream(m_streamIndex);
        m_gridConfig = scene.physicsProfile.fieldGrid2D;
        m_params = scene.physicsProfile.unifiedField.titorC204;
        m_integrator = scene.physicsProfile.integrator;
        m_stationaryAnchors = scene.bodies;

        m_width = std::max(16U, m_gridConfig.gridWidth);
        m_height = std::max(16U, m_gridConfig.gridHeight);
        m_cellCount = static_cast<std::size_t>(m_width) * static_cast<std::size_t>(m_height);
        m_dx = (2.0 * m_gridConfig.domainHalfExtent) / static_cast<double>(m_width - 1U);

        // Allocate FP64 GPU buffers for the 2D spatial metric slice.
        if (!m_visualDistortion.Allocate(context, m_cellCount) ||
            !m_g00Grid.Allocate(context, m_cellCount) ||
            !m_accelGGrid.Allocate(context, m_cellCount) ||
            !m_horizonZoneGrid.Allocate(context, m_cellCount)) {
            spdlog::error("TitorC204Solver: Failed allocating FP64 2D metric GPU buffers.");
            Shutdown();
            return false;
        }

        m_hostVisual.assign(m_cellCount, 0.0);
        m_hostG00.assign(m_cellCount, -1.0);
        m_hostAccelG.assign(m_cellCount, 0.0);
        m_hostHorizonZone.assign(m_cellCount, 1);

        // Calibrate electrical charge and wave penetration rate from verbatim manual tables,
        // applying Titor's forward-travel energy reduction (`kForwardTravelEnergyFactor`) when Delta t > 0.
        const double pPct = std::clamp(m_params.powerLevelPercent, 0.0, 100.0);
        const bool isForward = (m_params.targetDisplacementYears > 0.0);
        const double baseCharge = (m_params.injectedChargeCoulombs > 0.0)
            ? m_params.injectedChargeCoulombs
            : ComputeC204ChargeForPower(pPct);
        m_effectiveChargeCoulombs = isForward
            ? (baseCharge * c204::kForwardTravelEnergyFactor)
            : baseCharge;
        m_penetrationRateYrsPerHr = ComputeC204PenetrationRateYrsPerHr(pPct);
        m_timeVelocitySecPerSec = ComputeC204CoordinateTimeVelocity(m_penetrationRateYrsPerHr);

        // Reset onboard flight computer & mission telemetry to departure state.
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;
        m_rampFraction = 0.0;
        m_stateTimerSec = 0.0;
        m_vglTimerSec = 0.0;
        m_vglCheckCount = 0ULL;
        m_bordaConsensusSpread = 0.0;
        m_expendedChargeCoulombs = 0.0;
        m_elapsedDisplacedYears = 0.0;
        m_currentWorldlineYear = m_params.departureYear;
        m_worldlineDivergencePct = 0.0;
        m_cabinOxygenPct = 100.0;
        m_carbonSensorAlert = false;
        m_uvBurstActive = false;
        m_opticalBlackoutActive = false;
        m_originReturnTriggered = false;
        m_abortReason.clear();

        // Enforce verbatim C204 pre-flight safety check: Gross loading limit <= 2,500 kg.
        if (m_params.payloadMassKg > c204::kMaxPayloadMassKg) {
            m_state = C204OperationalState::OriginReturnAbort;
            m_originReturnTriggered = true;
            m_abortReason = "Pre-flight mass overload: payload exceeds 2500 kg C204 loading limit.";
            spdlog::warn("TitorC204Solver: {}", m_abortReason);
        } else {
            // Engage initial VGL baseline gravity lock snapshot before ramp-up.
            m_state = C204OperationalState::VglBaselineLock;
        }

        if (!LaunchMetricGridKernel() || !Synchronize()) {
            Shutdown();
            return false;
        }

        m_initialized = true;
        spdlog::info(
            "TitorC204Solver initialized [grid={}x{}, L={:.1f}m, power={:.0f}%, Q={:.0f} C, rate={:.2f} yrs/hr, target={:+.1f} yr].",
            m_width, m_height, m_gridConfig.domainHalfExtent,
            pPct, m_effectiveChargeCoulombs, m_penetrationRateYrsPerHr,
            m_params.targetDisplacementYears
        );
        return true;
    }

    void TitorC204Solver::Shutdown() noexcept {
        if (m_initialized && m_context != nullptr && m_context->IsInitialized()) {
            (void)Synchronize();
        }
        m_visualDistortion.Release();
        m_g00Grid.Release();
        m_accelGGrid.Release();
        m_horizonZoneGrid.Release();
        m_context = nullptr;
        m_stream = nullptr;
        m_initialized = false;
        m_stepCount = 0ULL;
        m_simulationTime = 0.0;
    }

    void TitorC204Solver::AdvanceOperationalStateMachine(double dt) {
        const double absDt = std::abs(dt);
        m_stateTimerSec += absDt;

        // Refresh calibrated rates in case parameters were tuned.
        const double pPct = std::clamp(m_params.powerLevelPercent, 0.0, 100.0);
        const bool isForward = (m_params.targetDisplacementYears > 0.0);
        const double baseCharge = (m_params.injectedChargeCoulombs > 0.0)
            ? m_params.injectedChargeCoulombs
            : ComputeC204ChargeForPower(pPct);
        m_effectiveChargeCoulombs = isForward
            ? (baseCharge * c204::kForwardTravelEnergyFactor)
            : baseCharge;
        m_penetrationRateYrsPerHr = ComputeC204PenetrationRateYrsPerHr(pPct);
        m_timeVelocitySecPerSec = ComputeC204CoordinateTimeVelocity(m_penetrationRateYrsPerHr);

        // Evaluate 3-computer Borda gravity sensor consensus (with optional single-computer failure test).
        const double baseGravityReading = 1.0 + m_params.simulatedVglAnomaly;
        const double s0 = baseGravityReading + 0.0004 * std::sin(0.37 * m_simulationTime);
        const double s1 = baseGravityReading - 0.0003 * std::cos(0.53 * m_simulationTime);
        const double s2 = baseGravityReading + 0.0002 * std::sin(0.71 * m_simulationTime);
        double consensusGravity = 1.0;
        double bordaSpread = 0.0;
        const bool computersHealthy = EvaluateC204BordaConsensus(
            s0, s1, s2,
            true,
            true,
            !m_params.simulateSingleComputerFailure,
            consensusGravity,
            bordaSpread
        );
        m_bordaConsensusSpread = bordaSpread;

        // Check hard failsafe conditions triggering Automatic Origin Return (TMIM-200S/4005).
        if (!m_originReturnTriggered) {
            if (m_params.payloadMassKg > c204::kMaxPayloadMassKg) {
                m_originReturnTriggered = true;
                m_state = C204OperationalState::OriginReturnAbort;
                m_abortReason = "Payload mass exceeds 2500 kg C204 structural loading limit.";
            } else if (!computersHealthy) {
                m_originReturnTriggered = true;
                m_state = C204OperationalState::OriginReturnAbort;
                m_abortReason = "Navigation computer bus redundancy lost (< 2 active computers).";
            } else if (std::abs(consensusGravity - 1.0) > m_params.vglFaultThreshold) {
                m_originReturnTriggered = true;
                m_state = C204OperationalState::OriginReturnAbort;
                m_abortReason = "VGL detected radical terrestrial gravity anomaly; Automatic Origin Return engaged.";
            }
        }

        // Update trapped air-pocket oxygen depletion when windows are sealed under active field.
        // Realistic finite cabin air pocket lasts ~8 hours without auxiliary rebreather.
        if (m_rampFraction > 0.1) {
            const double o2DrainPctPerSec = 100.0 / (8.0 * c204::kSecondsPerHour);
            m_cabinOxygenPct = std::max(0.0, m_cabinOxygenPct - (o2DrainPctPerSec * absDt));
            m_carbonSensorAlert = (m_cabinOxygenPct < 25.0);
        }

        const double rampDur = std::max(2.0, m_params.rampUpDurationSec);
        const double vglInterval = std::max(10.0, m_params.vglCheckIntervalSec);
        const double targetAbsYears = std::abs(m_params.targetDisplacementYears);
        const double travelSign = (m_params.targetDisplacementYears < 0.0) ? -1.0 : 1.0;

        switch (m_state) {
            case C204OperationalState::Standby:
                m_rampFraction = 0.0;
                m_uvBurstActive = false;
                m_opticalBlackoutActive = false;
                if (m_stateTimerSec >= 1.0) {
                    m_state = C204OperationalState::VglBaselineLock;
                    m_stateTimerSec = 0.0;
                }
                break;

            case C204OperationalState::VglBaselineLock:
                // Baseline gravity gradiometer lock takes 2.0 pilot seconds before countdown alarm.
                m_rampFraction = 0.05 * std::clamp(m_stateTimerSec / 2.0, 0.0, 1.0);
                m_uvBurstActive = false;
                m_opticalBlackoutActive = false;
                if (m_stateTimerSec >= 2.0) {
                    m_state = C204OperationalState::RampUp;
                    m_stateTimerSec = 0.0;
                }
                break;

            case C204OperationalState::RampUp: {
                // Smooth S-curve field expansion across `rampUpDurationSec`; final 1.0 s triggers
                // rainbow gravitational blueshift & UV radiation burst prior to optical blackout.
                const double frac = std::clamp(m_stateTimerSec / rampDur, 0.0, 1.0);
                m_rampFraction = 0.05 + 0.95 * (frac * frac * (3.0 - 2.0 * frac));
                const double timeLeft = rampDur - m_stateTimerSec;
                m_uvBurstActive = (timeLeft <= c204::kUvBurstDurationSec && timeLeft > 0.0);
                m_opticalBlackoutActive = false;
                m_expendedChargeCoulombs += (m_effectiveChargeCoulombs / 3600.0) * m_rampFraction * absDt;
                if (m_stateTimerSec >= rampDur) {
                    m_rampFraction = 1.0;
                    m_uvBurstActive = false;
                    m_opticalBlackoutActive = true;
                    m_state = C204OperationalState::TemporalTransit;
                    m_stateTimerSec = 0.0;
                    m_vglTimerSec = 0.0;
                }
                break;
            }

            case C204OperationalState::TemporalTransit: {
                m_rampFraction = 1.0;
                m_uvBurstActive = false;
                m_opticalBlackoutActive = true;

                // Advance displaced external worldline years using exact wave penetration rate.
                // `m_effectiveChargeCoulombs` already incorporates `kForwardTravelEnergyFactor` when moving forward.
                const double deltaYearsMag = (m_penetrationRateYrsPerHr / c204::kSecondsPerHour) * absDt;
                const double currentAbs = std::abs(m_elapsedDisplacedYears);
                const double nextAbs = std::min(targetAbsYears, currentAbs + deltaYearsMag);
                m_elapsedDisplacedYears = travelSign * nextAbs;
                m_currentWorldlineYear = m_params.departureYear + m_elapsedDisplacedYears;

                // Integrate Coulomb expenditure across transit.
                m_expendedChargeCoulombs += (m_effectiveChargeCoulombs / c204::kSecondsPerHour) * absDt;

                // Advance 240-second Variable Gravity Lock (VGL) pulse timer.
                m_vglTimerSec += absDt;
                if (m_vglTimerSec >= vglInterval) {
                    const std::uint64_t pulses = static_cast<std::uint64_t>(m_vglTimerSec / vglInterval);
                    m_vglCheckCount += pulses;
                    m_vglTimerSec = std::fmod(m_vglTimerSec, vglInterval);
                    // Each VGL pulse draws a fraction of the 54,000 C VGL total budget.
                    m_expendedChargeCoulombs += static_cast<double>(pulses) * (m_params.vglChargeBudgetCoulombs / 10.0);
                    m_state = C204OperationalState::VglPulseCheck;
                    m_stateTimerSec = 0.0;
                }

                // Update cumulative worldline divergence percentage.
                m_worldlineDivergencePct = ComputeC204WorldlineDivergencePct(
                    m_elapsedDisplacedYears, m_vglCheckCount, m_bordaConsensusSpread
                );

                // Transition to arrival cooldown once target worldline coordinate is reached.
                if (targetAbsYears > 0.0 && nextAbs >= targetAbsYears - 1.0e-9) {
                    m_state = C204OperationalState::ArrivalCooldown;
                    m_stateTimerSec = 0.0;
                }
                break;
            }

            case C204OperationalState::VglPulseCheck: {
                // Brief 2-second pulsed terrestrial gravity re-synchronization check before resuming transit.
                m_rampFraction = 1.0;
                m_uvBurstActive = false;
                m_opticalBlackoutActive = true;
                m_worldlineDivergencePct = ComputeC204WorldlineDivergencePct(
                    m_elapsedDisplacedYears, m_vglCheckCount, m_bordaConsensusSpread
                );
                if (m_stateTimerSec >= 2.0) {
                    m_state = C204OperationalState::TemporalTransit;
                    m_stateTimerSec = 0.0;
                }
                break;
            }

            case C204OperationalState::ArrivalCooldown: {
                // Step down distortion field over 5 seconds; optical horizon opens & daylight returns.
                const double coolDur = 5.0;
                const double frac = std::clamp(m_stateTimerSec / coolDur, 0.0, 1.0);
                m_rampFraction = 1.0 - frac;
                m_uvBurstActive = false;
                m_opticalBlackoutActive = (m_rampFraction > 0.35);
                break;
            }

            case C204OperationalState::OriginReturnAbort: {
                // Failsafe Automatic Origin Return: smoothly rewind displaced years back to origin
                // and step down the distortion envelope to safe standby.
                const double rewindYears = (m_penetrationRateYrsPerHr / c204::kSecondsPerHour) * absDt;
                if (std::abs(m_elapsedDisplacedYears) > rewindYears) {
                    m_elapsedDisplacedYears -= std::copysign(rewindYears, m_elapsedDisplacedYears);
                    m_currentWorldlineYear = m_params.departureYear + m_elapsedDisplacedYears;
                    m_rampFraction = std::max(0.25, m_rampFraction * 0.98);
                } else {
                    m_elapsedDisplacedYears = 0.0;
                    m_currentWorldlineYear = m_params.departureYear;
                    m_rampFraction = std::max(0.0, m_rampFraction - (0.2 * absDt));
                }
                m_uvBurstActive = false;
                m_opticalBlackoutActive = (m_rampFraction > 0.45);
                break;
            }
        }
    }

    bool TitorC204Solver::LaunchMetricGridKernel() {
        if (m_context == nullptr || !m_context->MakeCurrent()) {
            return false;
        }

        // Smooth bell-curve VGL re-synchronization pulse active strictly during the 2-second VglPulseCheck.
        const double vglPulseEnv = (m_state == C204OperationalState::VglPulseCheck)
            ? std::sin(constants::kPi * std::clamp(m_stateTimerSec / 2.0, 0.0, 1.0))
            : 0.0;

        const C204KernelParams kp{
            m_width,
            m_height,
            m_gridConfig.domainHalfExtent,
            m_simulationTime,
            std::clamp(m_params.powerLevelPercent / 100.0, 0.0, 1.0),
            m_rampFraction,
            m_params.envelopeOffsetXM,
            m_params.envelopeHalfLengthA,
            m_params.envelopeHalfWidthB,
            m_params.envelopeHalfHeightC,
            m_params.singularitySpacingM,
            m_params.cabinProbeDistanceM,
            m_params.tiplerWavelengthM,
            m_params.dampingScaleM,
            m_params.wallSteepnessSigma,
            m_params.xrayVentConeDeg,
            m_params.tiplerCylinderParamAR,
            (m_params.targetDisplacementYears > 0.0),
            vglPulseEnv
        };

        const dim3 block(16U, 16U);
        const dim3 grid(
            (m_width + block.x - 1U) / block.x,
            (m_height + block.y - 1U) / block.y
        );

        ComputeTitorC204MetricGrid2DKernel<<<grid, block, 0, m_stream>>>(
            m_visualDistortion.Data(),
            m_g00Grid.Data(),
            m_accelGGrid.Data(),
            m_horizonZoneGrid.Data(),
            kp
        );
        return UMGEBUNG_CUDA_CHECK_LAST_ERROR();
    }

    bool TitorC204Solver::Step(double dt) {
        if (!m_initialized || m_context == nullptr || !std::isfinite(dt) || dt == 0.0) {
            return false;
        }

        AdvanceOperationalStateMachine(dt);
        ++m_stepCount;
        m_simulationTime += dt;

        return LaunchMetricGridKernel();
    }

    bool TitorC204Solver::StepMultiple(std::uint64_t stepCount, double dt) {
        for (std::uint64_t s = 0ULL; s < stepCount; ++s) {
            if (!Step(dt)) {
                return false;
            }
        }
        return Synchronize();
    }

    bool TitorC204Solver::Synchronize() const noexcept {
        return (m_context != nullptr && m_context->IsInitialized()) &&
               m_context->SynchronizeStream(m_streamIndex);
    }

    bool TitorC204Solver::SyncToHost(std::vector<scene::BodyEntity>& outBodies) {
        outBodies = m_stationaryAnchors;
        return m_initialized;
    }

    bool TitorC204Solver::DownloadGridBuffers() {
        if (!m_initialized || m_context == nullptr) {
            return false;
        }
        if (!m_visualDistortion.CopyToHost(m_hostVisual, m_stream) ||
            !m_g00Grid.CopyToHost(m_hostG00, m_stream) ||
            !m_accelGGrid.CopyToHost(m_hostAccelG, m_stream) ||
            !m_horizonZoneGrid.CopyToHost(m_hostHorizonZone, m_stream)) {
            return false;
        }
        return Synchronize();
    }

    bool TitorC204Solver::FetchFieldGrid2DSnapshot(FieldGrid2DSnapshot& outSnapshot) {
        if (!DownloadGridBuffers()) {
            return false;
        }

        double minV = m_hostVisual.empty() ? 0.0 : m_hostVisual[0];
        double maxV = minV;
        double sumSq = 0.0;
        double peakAbs = 0.0;

        for (double v : m_hostVisual) {
            minV = std::min(minV, v);
            maxV = std::max(maxV, v);
            sumSq += (v * v);
            peakAbs = std::max(peakAbs, std::abs(v));
        }

        // Evaluate exact cabin probe sample at (x = cabinProbeDistanceM, y = 0, z = 0) for live strip-chart.
        const double vglPulseEnv = (m_state == C204OperationalState::VglPulseCheck)
            ? std::sin(constants::kPi * std::clamp(m_stateTimerSec / 2.0, 0.0, 1.0))
            : 0.0;
        const C204MetricPointSample cabinSample = EvaluateC204EffectiveMetricPoint(
            m_params.cabinProbeDistanceM, 0.0, 0.0,
            m_simulationTime,
            std::clamp(m_params.powerLevelPercent / 100.0, 0.0, 1.0),
            m_rampFraction,
            m_params.envelopeOffsetXM,
            m_params.envelopeHalfLengthA,
            m_params.envelopeHalfWidthB,
            m_params.envelopeHalfHeightC,
            m_params.singularitySpacingM,
            m_params.cabinProbeDistanceM,
            m_params.tiplerWavelengthM,
            m_params.dampingScaleM,
            m_params.wallSteepnessSigma,
            m_params.xrayVentConeDeg,
            m_params.tiplerCylinderParamAR,
            (m_params.targetDisplacementYears > 0.0),
            vglPulseEnv
        );

        outSnapshot = FieldGrid2DSnapshot{};
        outSnapshot.width = m_width;
        outSnapshot.height = m_height;
        outSnapshot.domainHalfExtent = m_gridConfig.domainHalfExtent;
        outSnapshot.scalarField = m_hostVisual;
        outSnapshot.minVal = minV;
        outSnapshot.maxVal = maxV;
        outSnapshot.l2Norm = (m_cellCount > 0U) ? std::sqrt(sumSq / static_cast<double>(m_cellCount)) : 0.0;
        outSnapshot.peakAbsVal = peakAbs;
        outSnapshot.probeSampleValue = cabinSample.metricG00;
        outSnapshot.fieldLabel = "C204 Effective Metric Distortion |1 + g_00^eff(x,y)|";
        return true;
    }

    bool TitorC204Solver::ComputeSolverDiagnostics(SolverDiagnostics& outDiagnostics) {
        FieldGrid2DSnapshot snap{};
        if (!FetchFieldGrid2DSnapshot(snap)) {
            return false;
        }

        // Integrate distortion field energy & classify active horizon cell populations from GPU buffers.
        // Also detects discrete 4-neighbor sign crossings of g_00 so the Null Time Event Horizon
        // contour can never be missed between adjacent grid cells.
        double fieldEnergy = 0.0;
        std::uint64_t negHorizonCount = 0ULL;
        std::uint64_t nullHorizonCount = 0ULL;
        std::uint64_t posHorizonCount = 0ULL;
        const double cellArea = m_dx * m_dx;

        for (std::uint32_t iy = 0U; iy < m_height; ++iy) {
            for (std::uint32_t ix = 0U; ix < m_width; ++ix) {
                const std::size_t idx = static_cast<std::size_t>(iy) * m_width + ix;
                const double d = m_hostVisual[idx];
                fieldEnergy += 0.5 * (d * d) * cellArea;

                const int zone = m_hostHorizonZone[idx];
                const double gHere = m_hostG00[idx];
                bool crossesZero = false;
                if (zone != 0 && zone != 2) {
                    if (ix + 1U < m_width && (gHere * m_hostG00[idx + 1U] < 0.0)) { crossesZero = true; }
                    if (iy + 1U < m_height && (gHere * m_hostG00[idx + m_width] < 0.0)) { crossesZero = true; }
                }

                if (zone == 0 || crossesZero) {
                    ++nullHorizonCount;
                } else if (zone < 0) {
                    ++negHorizonCount;
                } else if (zone == 1) {
                    ++posHorizonCount;
                }
            }
        }

        // Evaluate exact cabin probe point (pilot seat at x = cabinProbeDistanceM, y = 0).
        const double pFrac = std::clamp(m_params.powerLevelPercent / 100.0, 0.0, 1.0);
        const bool isForward = (m_params.targetDisplacementYears > 0.0);
        const double vglPulseEnv = (m_state == C204OperationalState::VglPulseCheck)
            ? std::sin(constants::kPi * std::clamp(m_stateTimerSec / 2.0, 0.0, 1.0))
            : 0.0;
        const C204MetricPointSample cabinSample = EvaluateC204EffectiveMetricPoint(
            m_params.cabinProbeDistanceM, 0.0, 0.0,
            m_simulationTime,
            pFrac,
            m_rampFraction,
            m_params.envelopeOffsetXM,
            m_params.envelopeHalfLengthA,
            m_params.envelopeHalfWidthB,
            m_params.envelopeHalfHeightC,
            m_params.singularitySpacingM,
            m_params.cabinProbeDistanceM,
            m_params.tiplerWavelengthM,
            m_params.dampingScaleM,
            m_params.wallSteepnessSigma,
            m_params.xrayVentConeDeg,
            m_params.tiplerCylinderParamAR,
            isForward,
            vglPulseEnv
        );

        std::string opticalPhase = "Normal Exterior Visibility";
        if (m_opticalBlackoutActive) {
            opticalPhase = "Total Blackout (Optical Trapping Horizon Closed)";
        } else if (m_uvBurstActive) {
            opticalPhase = "Rainbow Lensing + UV Burst (Eye Protection Required!)";
        } else if (m_state == C204OperationalState::RampUp) {
            opticalPhase = "Gravitational Lensing Onset (Blueshift Climbing)";
        }

        const double vglInterval = std::max(10.0, m_params.vglCheckIntervalSec);

        outDiagnostics = SolverDiagnostics{};
        outDiagnostics.modelId = scene::ModelId::ThinkerTitorC204;
        outDiagnostics.domain = scene::SimulationDomain::FieldGrid2D;
        outDiagnostics.stepCount = m_stepCount;
        outDiagnostics.simulationTime = m_simulationTime;
        outDiagnostics.totalEnergy = fieldEnergy;
        outDiagnostics.fieldEnergy = fieldEnergy;
        outDiagnostics.fieldL2Norm = snap.l2Norm;
        outDiagnostics.fieldPeakValue = snap.peakAbsVal;
        outDiagnostics.fieldProbeValue = cabinSample.metricG00;
        outDiagnostics.effectiveCflNumber = 0.0; // Analytical/kinematic metric evaluation (unconditionally stable).

        outDiagnostics.hasUnifiedTelemetry = true;
        outDiagnostics.hasC204Telemetry = true;
        outDiagnostics.c204OperationalState = C204OperationalStateToString(m_state);
        outDiagnostics.c204PowerLevelPercent = m_params.powerLevelPercent;
        outDiagnostics.c204InjectedChargeCoulombs = m_effectiveChargeCoulombs;
        outDiagnostics.c204ExpendedChargeCoulombs = m_expendedChargeCoulombs;
        outDiagnostics.c204PenetrationRateYrsPerHr = m_penetrationRateYrsPerHr;
        outDiagnostics.c204TimeVelocitySecPerSec = m_timeVelocitySecPerSec;
        outDiagnostics.c204CurrentWorldlineYear = m_currentWorldlineYear;
        outDiagnostics.c204ElapsedDisplacedYears = m_elapsedDisplacedYears;
        outDiagnostics.c204WorldlineDivergencePct = m_worldlineDivergencePct;
        outDiagnostics.c204CabinGForce = cabinSample.properAccelG;
        outDiagnostics.c204ExteriorTempF = ComputeC204ExteriorTempF(m_params.powerLevelPercent, m_rampFraction);
        outDiagnostics.c204MetricG00AtCabin = cabinSample.metricG00;
        outDiagnostics.c204MetricGradG00AtCabin = cabinSample.metricGradientInvM;
        outDiagnostics.c204OpticalPhase = std::move(opticalPhase);
        outDiagnostics.c204VglCheckCount = m_vglCheckCount;
        outDiagnostics.c204VglNextCheckCountdownSec = std::max(0.0, vglInterval - m_vglTimerSec);
        outDiagnostics.c204BordaConsensusSpread = m_bordaConsensusSpread;
        outDiagnostics.c204CabinOxygenPct = m_cabinOxygenPct;
        outDiagnostics.c204CarbonSensorAlert = m_carbonSensorAlert;
        outDiagnostics.c204UvBurstActive = m_uvBurstActive;
        outDiagnostics.c204OpticalBlackoutActive = m_opticalBlackoutActive;
        outDiagnostics.c204OriginReturnTriggered = m_originReturnTriggered;
        outDiagnostics.c204AbortReason = m_abortReason;
        outDiagnostics.c204IsForwardTravel = isForward;
        outDiagnostics.c204NegativeHorizonCells = negHorizonCount;
        outDiagnostics.c204NullHorizonCells = nullHorizonCount;
        outDiagnostics.c204PositiveHorizonCells = posHorizonCount;
        outDiagnostics.c204KerrProgradePhotonR = ComputeKerrProgradePhotonRadius(1.0, m_params.kerrSpinParam);
        outDiagnostics.c204KerrRetrogradePhotonR = ComputeKerrRetrogradePhotonRadius(1.0, m_params.kerrSpinParam);

        outDiagnostics.solverSummary = GetSolverName();
        return true;
    }

} // namespace umgebung::models::thinkers
