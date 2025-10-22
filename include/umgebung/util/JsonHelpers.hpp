#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

// This is a special namespace that nlohmann::json looks into
namespace nlohmann {

    // --- Add this serializer for glm::vec3 ---
    template <>
    struct adl_serializer<glm::vec3> {
        // Convert from glm::vec3 to json array [x, y, z]
        static void to_json(json& j, const glm::vec3& vec) {
            j = { vec.x, vec.y, vec.z };
        }

        // Convert from json array to glm::vec3
        static void from_json(const json& j, glm::vec3& vec) {
            j.at(0).get_to(vec.x);
            j.at(1).get_to(vec.y);
            j.at(2).get_to(vec.z);
        }
    };

    // --- Add this serializer for glm::quat ---
    template <>
    struct adl_serializer<glm::quat> {
        // Convert from glm::quat to json array [w, x, y, z]
        static void to_json(json& j, const glm::quat& q) {
            j = { q.w, q.x, q.y, q.z };
        }

        // Convert from json array to glm::quat
        static void from_json(const json& j, glm::quat& q) {
            j.at(0).get_to(q.w);
            j.at(1).get_to(q.x);
            j.at(2).get_to(q.y);
            j.at(3).get_to(q.z);
        }
    };
}