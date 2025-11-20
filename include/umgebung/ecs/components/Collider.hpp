#pragma once

#include "umgebung/util/JsonHelpers.hpp"
#include <nlohmann/json.hpp>
#include <glm/glm.hpp>

namespace Umgebung
{
    namespace ecs
    {
        namespace components
        {
            struct Collider
            {
                enum class ColliderType { Box, Sphere };
                ColliderType type = ColliderType::Box;

                // Box properties
                glm::vec3 boxSize = { 0.5f, 0.5f, 0.5f };

                // Sphere properties
                float sphereRadius = 0.5f;

                // Runtime only, not serialized
                bool dirty = false;

                NLOHMANN_DEFINE_TYPE_INTRUSIVE(Collider, type, boxSize, sphereRadius)
            };

            NLOHMANN_JSON_SERIALIZE_ENUM(Collider::ColliderType, {
                {Collider::ColliderType::Box, "Box"},
                {Collider::ColliderType::Sphere, "Sphere"}
            })

        } // namespace components
    } // namespace ecs
} // namespace Umgebung
