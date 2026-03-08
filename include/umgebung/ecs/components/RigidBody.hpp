#pragma once

#include "umgebung/util/JsonHelpers.hpp"
#include <nlohmann/json.hpp>

// Forward declaration for PhysX class
namespace physx
{
    class PxRigidActor;
}

namespace Umgebung
{
    namespace ecs
    {
        namespace components
        {

            struct RigidBody
            {
                enum class BodyType { Static, Dynamic };
                BodyType type = BodyType::Static;

                float mass = 1.0f;
                float restitution = 0.5f;
                float friction = 0.5f;

                glm::vec3 linearVelocity = { 0.0f, 0.0f, 0.0f };
                glm::vec3 angularVelocity = { 0.0f, 0.0f, 0.0f };

                // Runtime only, not serialized
                physx::PxRigidActor* runtimeActor = nullptr;
                bool dirty = false;
            };

            inline void to_json(nlohmann::json& j, const RigidBody& rb) {
                j = nlohmann::json{
                    {"type", rb.type},
                    {"mass", rb.mass},
                    {"restitution", rb.restitution},
                    {"friction", rb.friction},
                    {"linearVelocity", rb.linearVelocity},
                    {"angularVelocity", rb.angularVelocity}
                };
            }

            inline void from_json(const nlohmann::json& j, RigidBody& rb) {
                if (j.contains("type")) j.at("type").get_to(rb.type);
                if (j.contains("mass")) j.at("mass").get_to(rb.mass);
                if (j.contains("restitution")) j.at("restitution").get_to(rb.restitution);
                else rb.restitution = 0.5f;
                if (j.contains("friction")) j.at("friction").get_to(rb.friction);
                else rb.friction = 0.5f;
                if (j.contains("linearVelocity")) j.at("linearVelocity").get_to(rb.linearVelocity);
                else rb.linearVelocity = { 0.0f, 0.0f, 0.0f };
                if (j.contains("angularVelocity")) j.at("angularVelocity").get_to(rb.angularVelocity);
                else rb.angularVelocity = { 0.0f, 0.0f, 0.0f };
            }

            NLOHMANN_JSON_SERIALIZE_ENUM(RigidBody::BodyType, {
                {RigidBody::BodyType::Static, "Static"},
                {RigidBody::BodyType::Dynamic, "Dynamic"}
            })

        } // namespace components
    } // namespace ecs
} // namespace Umgebung
