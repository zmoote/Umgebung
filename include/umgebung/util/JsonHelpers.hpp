#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

/**
 * @file JsonHelpers.hpp
 * @brief Contains helper functions for serializing and deserializing glm types with nlohmann::json.
 */
#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

namespace nlohmann {

    /**
     * @brief A serializer for glm::vec3.
     */
    template <>
    struct adl_serializer<glm::vec3> {
        /**
         * @brief Converts a glm::vec3 to a json array.
         * 
         * @param j The json object.
         * @param vec The vector.
         */
        static void to_json(json& j, const glm::vec3& vec) {
            j = { vec.x, vec.y, vec.z };
        }

        /**
         * @brief Converts a json array to a glm::vec3.
         * 
         * @param j The json object.
         * @param vec The vector.
         */
        static void from_json(const json& j, glm::vec3& vec) {
            j.at(0).get_to(vec.x);
            j.at(1).get_to(vec.y);
            j.at(2).get_to(vec.z);
        }
    };

    /**
     * @brief A serializer for glm::quat.
     */
    template <>
    struct adl_serializer<glm::quat> {
        /**
         * @brief Converts a glm::quat to a json array.
         * 
         * @param j The json object.
         * @param q The quaternion.
         */
        static void to_json(json& j, const glm::quat& q) {
            j = { q.w, q.x, q.y, q.z };
        }

        /**
         * @brief Converts a json array to a glm::quat.
         * 
         * @param j The json object.
         * @param q The quaternion.
         */
        static void from_json(const json& j, glm::quat& q) {
            j.at(0).get_to(q.w);
            j.at(1).get_to(q.x);
            j.at(2).get_to(q.y);
            j.at(3).get_to(q.z);
        }
    };

    /**
     * @brief A serializer for glm::vec4.
     */
    template <>
    struct adl_serializer<glm::vec4> {
        /**
         * @brief Converts a glm::vec4 to a json array.
         * 
         * @param j The json object.
         * @param v The vector.
         */
        static void to_json(json& j, const glm::vec4& v) {
            j = { v.w, v.x, v.y, v.z };
        }

        /**
         * @brief Converts a json array to a glm::vec4.
         * 
         * @param j The json object.
         * @param v The vector.
         */
        static void from_json(const json& j, glm::vec4& v) {
            j.at(0).get_to(v.w);
            j.at(1).get_to(v.x);
            j.at(2).get_to(v.y);
            j.at(3).get_to(v.z);
        }
    };
}