/**
 * @file UniverseScene.cpp
 * @brief Implementation of `.umgebung` JSON schema v1 validation, serialization,
 *        and filesystem I/O (`umgebung::scene::UniverseScene`).
 *
 * @details
 * Enforces strict finite-number and domain validation when parsing `.umgebung`
 * universe files so invalid initial conditions (NaN/Inf coordinates, negative
 * masses, non-positive timesteps, duplicate entity IDs) are rejected cleanly
 * on the host before any GPU memory is allocated.
 */

#include "umgebung/scene/UniverseScene.hpp"

#include <cmath>
#include <fstream>
#include <unordered_set>

namespace umgebung::scene {

    namespace {

        /** @brief Returns `true` iff all three components of `v` are finite IEEE-754 numbers. */
        [[nodiscard]] bool IsFiniteVec3(const Vec3d& v) noexcept {
            return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
        }

        /** @brief Converts a `Vec3d` into a 3-element JSON array `[x, y, z]`. */
        [[nodiscard]] nlohmann::json Vec3ToJson(const Vec3d& v) {
            return nlohmann::json::array({v.x, v.y, v.z});
        }

        /**
         * @brief Parses a 3-element numeric JSON array `[x, y, z]` into `Vec3d`.
         *
         * @param      arr       JSON value expected to be an array of 3 numbers.
         * @param[out] outVec    Populated vector on success.
         * @param      fieldName Field path used in error reporting.
         * @param[out] outError  Diagnostic message on failure.
         * @return `true` on success; `false` otherwise.
         */
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

    } // anonymous namespace

    bool UniverseScene::Validate(std::string& outError) const {
        if (schemaVersion != kCurrentSchemaVersion) {
            outError = "Unsupported .umgebung schema_version " + std::to_string(schemaVersion) +
                       " (expected " + std::to_string(kCurrentSchemaVersion) + ").";
            return false;
        }

        if (universeId.empty()) {
            outError = "UniverseScene requires a non-empty 'universe_id'.";
            return false;
        }

        if (name.empty()) {
            outError = "UniverseScene requires a non-empty 'name'.";
            return false;
        }

        if (!std::isfinite(physicsProfile.timestep) || physicsProfile.timestep <= 0.0) {
            outError = "PhysicsProfile 'timestep' must be a finite strictly positive number.";
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

        if (!std::isfinite(physicsProfile.unifiedField.referenceLightSpeed) ||
            physicsProfile.unifiedField.referenceLightSpeed <= 0.0) {
            outError = "Unified field 'reference_light_speed' must be finite and > 0.";
            return false;
        }

        if (physicsProfile.unifiedField.aetherVortexEnabled) {
            if (!std::isfinite(physicsProfile.unifiedField.aetherVortexCriticalRadius) ||
                physicsProfile.unifiedField.aetherVortexCriticalRadius <= 0.0) {
                outError = "Unified field 'aether_vortex_critical_radius' must be finite and > 0 when enabled.";
                return false;
            }
            if (!std::isfinite(physicsProfile.unifiedField.aetherVortexScaleHeight) ||
                physicsProfile.unifiedField.aetherVortexScaleHeight <= 0.0) {
                outError = "Unified field 'aether_vortex_scale_height' must be finite and > 0 when enabled.";
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
                outError = "Duplicate body 'id' detected in universe '" + universeId + "': " + body.id;
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

        return nlohmann::json{
            {"schema_version", schemaVersion},
            {"universe_id", universeId},
            {"name", name},
            {"description", description},
            {"physics_profile", {
                {"type", PhysicsProfileTypeToString(physicsProfile.type)},
                {"integrator", IntegratorTypeToString(physicsProfile.integrator)},
                {"unit_system", physicsProfile.unitSystem},
                {"timestep", physicsProfile.timestep},
                {"default_steps", physicsProfile.defaultSteps},
                {"mainstream_gravity", {
                    {"gravitational_constant", physicsProfile.mainstreamGravity.gravitationalConstant},
                    {"softening_length", physicsProfile.mainstreamGravity.softeningLength}
                }},
                {"unified_field", {
                    {"harmonic_vacuum_density", physicsProfile.unifiedField.harmonicVacuumDensity},
                    {"reference_light_speed", physicsProfile.unifiedField.referenceLightSpeed},
                    {"aether_vortex_enabled", physicsProfile.unifiedField.aetherVortexEnabled},
                    {"aether_vortex_critical_radius", physicsProfile.unifiedField.aetherVortexCriticalRadius},
                    {"aether_vortex_scale_height", physicsProfile.unifiedField.aetherVortexScaleHeight},
                    {"aether_vortex_vertical_field_gz", physicsProfile.unifiedField.aetherVortexVerticalFieldGz},
                    {"love_binding_coupling", physicsProfile.unifiedField.loveBindingCoupling},
                    {"phryll_field_amplitude", physicsProfile.unifiedField.phryllFieldAmplitude},
                    {"base_vibrational_density", physicsProfile.unifiedField.baseVibrationalDensity},
                    {"non_linear_time_enabled", physicsProfile.unifiedField.nonLinearTimeEnabled}
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
            parsed.universeId = jsonObj.value("universe_id", std::string{});
            parsed.name = jsonObj.value("name", std::string{});
            parsed.description = jsonObj.value("description", std::string{});

            // Parse physics profile block.
            if (jsonObj.contains("physics_profile") && jsonObj["physics_profile"].is_object()) {
                const auto& prof = jsonObj["physics_profile"];

                const std::string typeStr = prof.value("type", std::string("Mainstream"));
                if (!PhysicsProfileTypeFromString(typeStr, parsed.physicsProfile.type)) {
                    outError = "Unknown physics_profile.type '" + typeStr + "'.";
                    return false;
                }

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

                if (prof.contains("unified_field") && prof["unified_field"].is_object()) {
                    const auto& uf = prof["unified_field"];
                    parsed.physicsProfile.unifiedField.harmonicVacuumDensity =
                        uf.value("harmonic_vacuum_density", 5.155e96);
                    parsed.physicsProfile.unifiedField.referenceLightSpeed =
                        uf.value("reference_light_speed", 299792458.0);
                    parsed.physicsProfile.unifiedField.aetherVortexEnabled =
                        uf.value("aether_vortex_enabled", false);
                    parsed.physicsProfile.unifiedField.aetherVortexCriticalRadius =
                        uf.value("aether_vortex_critical_radius", 1.0);
                    parsed.physicsProfile.unifiedField.aetherVortexScaleHeight =
                        uf.value("aether_vortex_scale_height", 1.0);
                    parsed.physicsProfile.unifiedField.aetherVortexVerticalFieldGz =
                        uf.value("aether_vortex_vertical_field_gz", 0.0);
                    parsed.physicsProfile.unifiedField.loveBindingCoupling =
                        uf.value("love_binding_coupling", 0.0);
                    parsed.physicsProfile.unifiedField.phryllFieldAmplitude =
                        uf.value("phryll_field_amplitude", 0.0);
                    parsed.physicsProfile.unifiedField.baseVibrationalDensity =
                        uf.value("base_vibrational_density", 3.0);
                    parsed.physicsProfile.unifiedField.nonLinearTimeEnabled =
                        uf.value("non_linear_time_enabled", false);
                }
            }

            // Parse body entities array.
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
            outError = "Failed to open .umgebung scene file: " + filePath.string();
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
            outError = "I/O error while writing .umgebung scene file: " + filePath.string();
            return false;
        }
        return true;
    }

} // namespace umgebung::scene
