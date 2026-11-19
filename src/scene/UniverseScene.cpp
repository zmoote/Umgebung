/**
 * @file UniverseScene.cpp
 * @brief Implementation of `.umgebung` / `.scenario.umgebung` JSON schema v1
 *        validation, serialization, and filesystem I/O.
 *
 * @details
 * Supports all three model tiers (`Mainstream::*`, standalone `Thinker::*`, and
 * `Unified::Synthesis`), multi-scale `ScaleTier` annotations on every entity,
 * and backward-compatible defaults when optional Thinker or NFW blocks are omitted.
 */

#include "umgebung/scene/UniverseScene.hpp"

#include <cmath>
#include <fstream>
#include <unordered_set>

namespace umgebung::scene {

    namespace {

        [[nodiscard]] bool IsFiniteVec3(const Vec3d& v) noexcept {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        }

        [[nodiscard]] nlohmann::json Vec3ToJson(const Vec3d& v) {
            return nlohmann::json::array({v.x, v.y, v.z});
        }

        [[nodiscard]] bool JsonToVec3(
            const nlohmann::json& arr,
            Vec3d& outVec,
            const char* fieldName,
            std::string& outError
        ) {
            if (!arr.is_array() || arr.size() != 3U) {
                outError = std::string("Field '") + fieldName + "' must be a 3-element numeric array [x, y, z].";
                return false;
            }
            if (!arr[0].is_number() || !arr[1].is_number() || !arr[2].is_number()) {
                outError = std::string("Field '") + fieldName + "' contains non-numeric components.";
                return false;
            }
            outVec.x = arr[0].get<double>();
            outVec.y = arr[1].get<double>();
            outVec.z = arr[2].get<double>();
            return true;
        }

        [[nodiscard]] ModelId InferDefaultModelFromProfileType(PhysicsProfileType type) noexcept {
            switch (type) {
                case PhysicsProfileType::Mainstream: return ModelId::MainstreamNewtonian;
                case PhysicsProfileType::Unified:    return ModelId::UnifiedSynthesis;
                case PhysicsProfileType::Custom:     return ModelId::UnifiedSynthesis;
            }
            return ModelId::MainstreamNewtonian;
        }

    } // anonymous namespace

    bool UniverseScene::Validate(std::string& outError) const {
        if (schemaVersion != kCurrentSchemaVersion) {
            outError = "Unsupported .umgebung schema_version " + std::to_string(schemaVersion) +
                       " (expected " + std::to_string(kCurrentSchemaVersion) + ").";
            return false;
        }

        if (universeId.empty()) {
            outError = "Scene requires a non-empty 'universe_id' (or 'scenario_id').";
            return false;
        }

        if (name.empty()) {
            outError = "Scene requires a non-empty 'name'.";
            return false;
        }

        if (!std::isfinite(physicsProfile.timestep) || physicsProfile.timestep <= 0.0) {
            outError = "PhysicsProfile 'timestep' must be finite and strictly positive.";
            return false;
        }

        if (!std::isfinite(physicsProfile.mainstreamGravity.gravitationalConstant) ||
            physicsProfile.mainstreamGravity.gravitationalConstant < 0.0) {
            outError = "Mainstream gravity 'gravitational_constant' must be finite and >= 0.";
            return false;
        }

        if (!std::isfinite(physicsProfile.mainstreamGravity.softeningLength) ||
            physicsProfile.mainstreamGravity.softeningLength < 0.0) {
            outError = "Mainstream gravity 'softening_length' must be finite and >= 0.";
            return false;
        }

        if (!std::isfinite(physicsProfile.darkMatterNFW.characteristicDensityRho0) ||
            physicsProfile.darkMatterNFW.characteristicDensityRho0 < 0.0 ||
            !std::isfinite(physicsProfile.darkMatterNFW.scaleRadiusRs) ||
            physicsProfile.darkMatterNFW.scaleRadiusRs <= 0.0) {
            outError = "DarkMatterNFW requires finite rho0 >= 0 and scale radius r_s > 0.";
            return false;
        }

        if (!std::isfinite(physicsProfile.unifiedField.referenceLightSpeed) ||
            physicsProfile.unifiedField.referenceLightSpeed <= 0.0) {
            outError = "Unified field 'reference_light_speed' must be finite and > 0.";
            return false;
        }

        if (physicsProfile.unifiedField.aetherVortexEnabled) {
            if (!std::isfinite(physicsProfile.unifiedField.aetherVortexCriticalRadius) ||
                physicsProfile.unifiedField.aetherVortexCriticalRadius <= 0.0) {
                outError = "Unified field 'aether_vortex_critical_radius' must be finite and > 0.";
                return false;
            }
            if (!std::isfinite(physicsProfile.unifiedField.aetherVortexScaleHeight) ||
                physicsProfile.unifiedField.aetherVortexScaleHeight <= 0.0) {
                outError = "Unified field 'aether_vortex_scale_height' must be finite and > 0.";
                return false;
            }
        }

        std::unordered_set<std::string> seenIds;
        seenIds.reserve(bodies.size());

        for (std::size_t i = 0U; i < bodies.size(); ++i) {
            const BodyEntity& body = bodies[i];
            if (body.id.empty()) {
                outError = "Body at index " + std::to_string(i) + " has an empty 'id'.";
                return false;
            }
            if (!seenIds.insert(body.id).second) {
                outError = "Duplicate body 'id' detected in scene '" + universeId + "': " + body.id;
                return false;
            }
            if (!std::isfinite(body.mass) || body.mass <= 0.0) {
                outError = "Body '" + body.id + "' must have a finite strictly positive mass.";
                return false;
            }
            if (!std::isfinite(body.radius) || body.radius < 0.0) {
                outError = "Body '" + body.id + "' must have a finite non-negative radius.";
                return false;
            }
            if (!IsFiniteVec3(body.position)) {
                outError = "Body '" + body.id + "' has non-finite position components.";
                return false;
            }
            if (!IsFiniteVec3(body.velocity)) {
                outError = "Body '" + body.id + "' has non-finite velocity components.";
                return false;
            }
        }

        return true;
    }

    nlohmann::json UniverseScene::ToJson() const {
        nlohmann::json bodiesJson = nlohmann::json::array();
        for (const BodyEntity& body : bodies) {
            bodiesJson.push_back({
                {"id", body.id},
                {"name", body.name},
                {"category", body.category},
                {"scale_tier", ScaleTierToString(body.scaleTier)},
                {"parent_id", body.parentId},
                {"mass", body.mass},
                {"radius", body.radius},
                {"position", Vec3ToJson(body.position)},
                {"velocity", Vec3ToJson(body.velocity)},
                {"unified_properties", {
                    {"vibrational_density", body.vibrationalDensity},
                    {"consciousness_coherence", body.consciousnessCoherence},
                    {"phryll_charge", body.phryllCharge}
                }}
            });
        }

        const auto& uf = physicsProfile.unifiedField;

        return nlohmann::json{
            {"schema_version", schemaVersion},
            {"scope", SimulationScopeToString(scope)},
            {"universe_id", universeId},
            {"name", name},
            {"description", description},
            {"physics_profile", {
                {"type", PhysicsProfileTypeToString(physicsProfile.type)},
                {"default_model", ModelIdToString(physicsProfile.defaultModel)},
                {"recommended_compare_model", physicsProfile.recommendedCompareModel},
                {"integrator", IntegratorTypeToString(physicsProfile.integrator)},
                {"unit_system", physicsProfile.unitSystem},
                {"timestep", physicsProfile.timestep},
                {"default_steps", physicsProfile.defaultSteps},
                {"mainstream_gravity", {
                    {"gravitational_constant", physicsProfile.mainstreamGravity.gravitationalConstant},
                    {"softening_length", physicsProfile.mainstreamGravity.softeningLength}
                }},
                {"dark_matter_nfw", {
                    {"characteristic_density_rho0", physicsProfile.darkMatterNFW.characteristicDensityRho0},
                    {"scale_radius_rs", physicsProfile.darkMatterNFW.scaleRadiusRs}
                }},
                {"unified_field", {
                    {"harmonic_vacuum_density", uf.harmonicVacuumDensity},
                    {"reference_light_speed", uf.referenceLightSpeed},
                    {"aether_vortex_enabled", uf.aetherVortexEnabled},
                    {"aether_vortex_critical_radius", uf.aetherVortexCriticalRadius},
                    {"aether_vortex_scale_height", uf.aetherVortexScaleHeight},
                    {"aether_vortex_vertical_field_gz", uf.aetherVortexVerticalFieldGz},
                    {"love_binding_coupling", uf.loveBindingCoupling},
                    {"phryll_field_amplitude", uf.phryllFieldAmplitude},
                    {"base_vibrational_density", uf.baseVibrationalDensity},
                    {"non_linear_time_enabled", uf.nonLinearTimeEnabled},
                    {"haramein", {
                        {"harmonic_vacuum_density", uf.haramein.harmonicVacuumDensity},
                        {"spacetime_torque_omega_z", uf.haramein.spacetimeTorqueOmegaZ}
                    }},
                    {"pais", {
                        {"geometric_factor", uf.pais.geometricFactor},
                        {"surface_charge_density", uf.pais.surfaceChargeDensity},
                        {"rotation_radius", uf.pais.rotationRadius},
                        {"angular_frequency_rad_s", uf.pais.angularFrequencyRadS},
                        {"vibration_amplitude", uf.pais.vibrationAmplitude},
                        {"vibration_frequency_hz", uf.pais.vibrationFrequencyHz},
                        {"translation_speed", uf.pais.translationSpeed},
                        {"inertial_reduction_coupling", uf.pais.inertialReductionCoupling}
                    }},
                    {"vogel_willis", {
                        {"torsion_coil_frequency_hz", uf.vogelWillis.torsionCoilFrequencyHz},
                        {"drive_voltage_v", uf.vogelWillis.driveVoltageV},
                        {"base_termination_angle_deg", uf.vogelWillis.baseTerminationAngleDeg},
                        {"top_termination_angle_deg", uf.vogelWillis.topTerminationAngleDeg},
                        {"saturation_threshold", uf.vogelWillis.saturationThreshold},
                        {"phryll_coupling", uf.vogelWillis.phryllCoupling}
                    }}
                }}
            }},
            {"bodies", std::move(bodiesJson)}
        };
    }

    bool UniverseScene::FromJson(
        const nlohmann::json& jsonObj,
        UniverseScene& outScene,
        std::string& outError
    ) {
        if (!jsonObj.is_object()) {
            outError = "Root of .umgebung document must be a JSON object.";
            return false;
        }

        try {
            UniverseScene parsed{};
            parsed.schemaVersion = jsonObj.value("schema_version", 0U);

            if (jsonObj.contains("scope") && jsonObj["scope"].is_string()) {
                const std::string scopeStr = jsonObj["scope"].get<std::string>();
                if (!SimulationScopeFromString(scopeStr, parsed.scope)) {
                    outError = "Unknown scene 'scope': '" + scopeStr + "'.";
                    return false;
                }
            } else {
                parsed.scope = SimulationScope::Scenario;
            }

            // Accept either `universe_id` or `scenario_id` so Scenario files read naturally.
            if (jsonObj.contains("scenario_id") && jsonObj["scenario_id"].is_string()) {
                parsed.universeId = jsonObj["scenario_id"].get<std::string>();
            } else {
                parsed.universeId = jsonObj.value("universe_id", std::string{});
            }

            parsed.name = jsonObj.value("name", std::string{});
            parsed.description = jsonObj.value("description", std::string{});

            if (jsonObj.contains("physics_profile") && jsonObj["physics_profile"].is_object()) {
                const auto& prof = jsonObj["physics_profile"];

                const std::string typeStr = prof.value("type", std::string("Mainstream"));
                if (!PhysicsProfileTypeFromString(typeStr, parsed.physicsProfile.type)) {
                    outError = "Unknown physics_profile.type '" + typeStr + "'.";
                    return false;
                }

                if (prof.contains("default_model") && prof["default_model"].is_string()) {
                    const std::string modelStr = prof["default_model"].get<std::string>();
                    if (!ModelIdFromString(modelStr, parsed.physicsProfile.defaultModel)) {
                        outError = "Unknown physics_profile.default_model '" + modelStr + "'.";
                        return false;
                    }
                } else {
                    parsed.physicsProfile.defaultModel =
                        InferDefaultModelFromProfileType(parsed.physicsProfile.type);
                }

                parsed.physicsProfile.recommendedCompareModel =
                    prof.value("recommended_compare_model", std::string{});

                const std::string integStr = prof.value("integrator", std::string("VelocityVerlet"));
                if (!IntegratorTypeFromString(integStr, parsed.physicsProfile.integrator)) {
                    outError = "Unknown physics_profile.integrator '" + integStr + "'.";
                    return false;
                }

                parsed.physicsProfile.unitSystem = prof.value("unit_system", std::string("Normalized"));
                parsed.physicsProfile.timestep = prof.value("timestep", 1.0e-3);
                parsed.physicsProfile.defaultSteps = prof.value("default_steps", 1000ULL);

                if (prof.contains("mainstream_gravity") && prof["mainstream_gravity"].is_object()) {
                    const auto& grav = prof["mainstream_gravity"];
                    parsed.physicsProfile.mainstreamGravity.gravitationalConstant =
                        grav.value("gravitational_constant", 6.67430e-11);
                    parsed.physicsProfile.mainstreamGravity.softeningLength =
                        grav.value("softening_length", 0.0);
                }

                if (prof.contains("dark_matter_nfw") && prof["dark_matter_nfw"].is_object()) {
                    const auto& nfw = prof["dark_matter_nfw"];
                    parsed.physicsProfile.darkMatterNFW.characteristicDensityRho0 =
                        nfw.value("characteristic_density_rho0", 0.0);
                    parsed.physicsProfile.darkMatterNFW.scaleRadiusRs =
                        nfw.value("scale_radius_rs", 2.0);
                }

                if (prof.contains("unified_field") && prof["unified_field"].is_object()) {
                    const auto& uf = prof["unified_field"];
                    auto& dst = parsed.physicsProfile.unifiedField;

                    dst.harmonicVacuumDensity = uf.value("harmonic_vacuum_density", 5.155e96);
                    dst.haramein.harmonicVacuumDensity = dst.harmonicVacuumDensity;
                    dst.referenceLightSpeed = uf.value("reference_light_speed", 299792458.0);
                    dst.aetherVortexEnabled = uf.value("aether_vortex_enabled", false);
                    dst.aetherVortexCriticalRadius = uf.value("aether_vortex_critical_radius", 1.0);
                    dst.aetherVortexScaleHeight = uf.value("aether_vortex_scale_height", 1.0);
                    dst.aetherVortexVerticalFieldGz = uf.value("aether_vortex_vertical_field_gz", 0.0);
                    dst.loveBindingCoupling = uf.value("love_binding_coupling", 0.0);
                    dst.phryllFieldAmplitude = uf.value("phryll_field_amplitude", 0.0);
                    dst.baseVibrationalDensity = uf.value("base_vibrational_density", 3.0);
                    dst.nonLinearTimeEnabled = uf.value("non_linear_time_enabled", false);

                    if (uf.contains("haramein") && uf["haramein"].is_object()) {
                        const auto& h = uf["haramein"];
                        dst.haramein.harmonicVacuumDensity = h.value("harmonic_vacuum_density", dst.harmonicVacuumDensity);
                        dst.haramein.spacetimeTorqueOmegaZ = h.value("spacetime_torque_omega_z", 0.0);
                    }

                    if (uf.contains("pais") && uf["pais"].is_object()) {
                        const auto& p = uf["pais"];
                        dst.pais.geometricFactor = p.value("geometric_factor", 1.0);
                        dst.pais.surfaceChargeDensity = p.value("surface_charge_density", 0.0);
                        dst.pais.rotationRadius = p.value("rotation_radius", 2.0);
                        dst.pais.angularFrequencyRadS = p.value("angular_frequency_rad_s", 0.0);
                        dst.pais.vibrationAmplitude = p.value("vibration_amplitude", 0.0);
                        dst.pais.vibrationFrequencyHz = p.value("vibration_frequency_hz", 0.0);
                        dst.pais.translationSpeed = p.value("translation_speed", 0.0);
                        dst.pais.inertialReductionCoupling = p.value("inertial_reduction_coupling", 0.0);
                    }

                    if (uf.contains("vogel_willis") && uf["vogel_willis"].is_object()) {
                        const auto& vw = uf["vogel_willis"];
                        dst.vogelWillis.torsionCoilFrequencyHz = vw.value("torsion_coil_frequency_hz", 4096.0);
                        dst.vogelWillis.driveVoltageV = vw.value("drive_voltage_v", 20.0);
                        dst.vogelWillis.baseTerminationAngleDeg = vw.value("base_termination_angle_deg", 51.84);
                        dst.vogelWillis.topTerminationAngleDeg = vw.value("top_termination_angle_deg", 60.0);
                        dst.vogelWillis.saturationThreshold = vw.value("saturation_threshold", 1000.0);
                        dst.vogelWillis.phryllCoupling = vw.value("phryll_coupling", 0.0);
                    }
                }
            }

            if (jsonObj.contains("bodies")) {
                if (!jsonObj["bodies"].is_array()) {
                    outError = "Field 'bodies' must be a JSON array.";
                    return false;
                }

                parsed.bodies.reserve(jsonObj["bodies"].size());
                for (const auto& bodyJson : jsonObj["bodies"]) {
                    if (!bodyJson.is_object()) {
                        outError = "Every element in 'bodies' must be a JSON object.";
                        return false;
                    }

                    BodyEntity body{};
                    body.id = bodyJson.value("id", std::string{});
                    body.name = bodyJson.value("name", body.id);
                    body.category = bodyJson.value("category", std::string("Body"));
                    body.parentId = bodyJson.value("parent_id", std::string{});
                    body.mass = bodyJson.value("mass", 1.0);
                    body.radius = bodyJson.value("radius", 0.05);

                    if (bodyJson.contains("scale_tier") && bodyJson["scale_tier"].is_string()) {
                        ScaleTier tier{};
                        if (ScaleTierFromString(bodyJson["scale_tier"].get<std::string>(), tier)) {
                            body.scaleTier = tier;
                        }
                    }

                    if (!bodyJson.contains("position") ||
                        !JsonToVec3(bodyJson["position"], body.position, "bodies[].position", outError)) {
                        return false;
                    }
                    if (!bodyJson.contains("velocity") ||
                        !JsonToVec3(bodyJson["velocity"], body.velocity, "bodies[].velocity", outError)) {
                        return false;
                    }

                    if (bodyJson.contains("unified_properties") && bodyJson["unified_properties"].is_object()) {
                        const auto& up = bodyJson["unified_properties"];
                        body.vibrationalDensity = up.value("vibrational_density", 3.0);
                        body.consciousnessCoherence = up.value("consciousness_coherence", 0.0);
                        body.phryllCharge = up.value("phryll_charge", 0.0);
                    }

                    parsed.bodies.push_back(std::move(body));
                }
            }

            if (!parsed.Validate(outError)) {
                return false;
            }

            outScene = std::move(parsed);
            return true;
        } catch (const nlohmann::json::exception& ex) {
            outError = std::string("JSON parsing error: ") + ex.what();
            return false;
        }
    }

    bool UniverseScene::LoadFromFile(
        const std::filesystem::path& filePath,
        UniverseScene& outScene,
        std::string& outError
    ) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            outError = "Failed to open .umgebung file: " + filePath.string();
            return false;
        }

        nlohmann::json parsedJson;
        try {
            file >> parsedJson;
        } catch (const nlohmann::json::exception& ex) {
            outError = "Malformed JSON in '" + filePath.string() + "': " + ex.what();
            return false;
        }

        if (!FromJson(parsedJson, outScene, outError)) {
            return false;
        }

        outScene.sourcePath = filePath;
        return true;
    }

    bool UniverseScene::SaveToFile(
        const std::filesystem::path& filePath,
        std::string& outError
    ) const {
        if (!Validate(outError)) {
            return false;
        }

        std::ofstream file(filePath);
        if (!file.is_open()) {
            outError = "Failed to open destination file for writing: " + filePath.string();
            return false;
        }

        file << ToJson().dump(2) << '\n';
        if (!file.good()) {
            outError = "I/O error while writing .umgebung file: " + filePath.string();
            return false;
        }
        return true;
    }

} // namespace umgebung::scene
