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

                // Runtime only, not serialized
                physx::PxRigidActor* runtimeActor = nullptr;

                NLOHMANN_DEFINE_TYPE_INTRUSIVE(RigidBody, type, mass)
            };

            NLOHMANN_JSON_SERIALIZE_ENUM(RigidBody::BodyType, {
                {RigidBody::BodyType::Static, "Static"},
                {RigidBody::BodyType::Dynamic, "Dynamic"}
            })

        } // namespace components
    } // namespace ecs
} // namespace Umgebung
