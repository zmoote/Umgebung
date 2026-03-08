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

                NLOHMANN_DEFINE_TYPE_INTRUSIVE(RigidBody, type, mass, restitution, friction, linearVelocity, angularVelocity)
            };

            NLOHMANN_JSON_SERIALIZE_ENUM(RigidBody::BodyType, {
                {RigidBody::BodyType::Static, "Static"},
                {RigidBody::BodyType::Dynamic, "Dynamic"}
            })

        } // namespace components
    } // namespace ecs
} // namespace Umgebung
