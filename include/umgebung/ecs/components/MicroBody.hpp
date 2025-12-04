#pragma once

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include "umgebung/util/JsonHelpers.hpp"

namespace Umgebung::ecs::components {

    struct MicroBody {
        glm::vec3 velocity{ 0.0f };
        float mass = 1.0f;
        float charge = 0.0f; // Future proofing
    };

    inline void to_json(nlohmann::json& j, const MicroBody& c) {
        j = nlohmann::json{
            {"velocity", c.velocity},
            {"mass", c.mass},
            {"charge", c.charge}
        };
    }

    inline void from_json(const nlohmann::json& j, MicroBody& c) {
        if(j.contains("velocity")) j.at("velocity").get_to(c.velocity);
        if(j.contains("mass")) j.at("mass").get_to(c.mass);
        if(j.contains("charge")) j.at("charge").get_to(c.charge);
    }

} // namespace Umgebung::ecs::components
