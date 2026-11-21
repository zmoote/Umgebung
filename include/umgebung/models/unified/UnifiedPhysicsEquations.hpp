/**
 * @file UnifiedPhysicsEquations.hpp
 * @brief Umbrella header re-exporting all standalone verbatim Thinker equation
 *        headers (`models/thinkers/<Thinker>Model.hpp`) into `umgebung::models::unified`
 *        with zero equation duplication.
 *
 * @details
 * Preserves full source compatibility for callers importing
 * `umgebung::models::unified::Compute*` while keeping every equation authored
 * in exactly one canonical Thinker header:
 *   - `EssonneModel.hpp`     (Chris Essonne `8052.pdf` & `10100.pdf`)
 *   - `HarameinModel.hpp`    (Nassim Haramein `Haramein342013PRRI3363.pdf` & spacetime torque)
 *   - `PaisModel.hpp`        (Salvatore Pais `US10144532.pdf` Eq. 1–2 & inertial mass reduction)
 *   - `DanaanModel.hpp`      (Elena Danaan Love binding coherence & non-linear local time)
 *   - `VogelWillisModel.hpp` (Marcel Vogel / Dan Willis / Eredyon `PhryllGenInstructions.pdf`)
 */

#pragma once

#include "umgebung/models/thinkers/DanaanModel.hpp"
#include "umgebung/models/thinkers/EssonneModel.hpp"
#include "umgebung/models/thinkers/HarameinModel.hpp"
#include "umgebung/models/thinkers/PaisModel.hpp"
#include "umgebung/models/thinkers/ThinkerConstants.hpp"
#include "umgebung/models/thinkers/TitorC204Model.hpp"
#include "umgebung/models/thinkers/VogelWillisModel.hpp"

/**
 * @namespace umgebung::models::unified
 * @brief Tier 3 Unified Synthesis composing all standalone Thinker modules
 *        without duplicating a single physical constant or equation.
 */
namespace umgebung::models::unified {
    namespace constants = thinkers::constants;

    using thinkers::ComputeDanaanLoveForceScale;
    using thinkers::ComputeEssonneAetherVortexPotential;
    using thinkers::ComputeEssonneAetherVortexRadialAccel;
    using thinkers::ComputeEssonneFlatRotationVelocity;
    using thinkers::ComputeEssonneInteriorPressureRatio;
    using thinkers::ComputeEssonneLocalLightSpeed;
    using thinkers::ComputeEssonneRefractiveIndex;
    using thinkers::ComputeEssonneVacuumPressureRatio;
    using thinkers::ComputeHarameinCoriolisAccelX;
    using thinkers::ComputeHarameinCoriolisAccelY;
    using thinkers::ComputeHolographicProtonMass;
    using thinkers::ComputeHolographicRatioPhi;
    using thinkers::ComputeHolographicSchwarzschildMass;
    using thinkers::ComputeLocalTimeRate;
    using thinkers::ComputePaisAccelAmplification;
    using thinkers::ComputePaisAcceleratedPoyntingFlux;
    using thinkers::ComputePaisInertialMassFactor;
    using thinkers::ComputePaisMaxPoyntingFlux;
    using thinkers::ComputePlanckEquatorialArea;
    using thinkers::ComputePlanckSphericalVolume;
    using thinkers::ComputeSchwarzschildRadius;
    using thinkers::ComputeSurfacePSUCount;
    using thinkers::ComputeVogelCrystalSaturationRatio;
    using thinkers::ComputeVogelGeometricResonanceFactor;
    using thinkers::ComputeVogelPhryllForceScale;
    using thinkers::ComputeVogelTorsionDrivePower;
    using thinkers::ComputeVolumePSUCount;
    using thinkers::IsVogelVoltageWithinSafeWindow;
} // namespace umgebung::models::unified
