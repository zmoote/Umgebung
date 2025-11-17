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

            struct RigidBodyComponent
            {
                enum class BodyType { Static, Dynamic };
                BodyType type = BodyType::Static;

                float mass = 1.0f;

                // Runtime only, not serialized
                physx::PxRigidActor* runtimeActor = nullptr;

                NLOHMANN_DEFINE_TYPE_INTRUSIVE(RigidBodyComponent, type, mass)
            };

            NLOHMANN_JSON_SERIALIZE_ENUM(RigidBodyComponent::BodyType, {
                {RigidBodyComponent::BodyType::Static, "Static"},
                {RigidBodyComponent::BodyType::Dynamic, "Dynamic"}
            })

        } // namespace components
    } // namespace ecs
} // namespace Umgebung
