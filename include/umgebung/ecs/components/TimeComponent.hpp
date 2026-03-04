#pragma once

#include <nlohmann/json.hpp>

namespace Umgebung
{
    namespace ecs
    {
        namespace components
        {
            struct TimeComponent
            {
                // Density controls the frequency/vibration of the entity.
                // 1.0 - 3.0: Physical matter.
                // 5.0 - 9.0: Higher density (Plasma/Etheric).
                // 10.0+: Pure energy/Source.
                float density = 3.0f;

                // Time multiplier applied to this specific entity (local dt)
                float localTimeMultiplier = 1.0f;

                // The computed dt that the entity actually experienced this frame
                float subjectiveDt = 0.0f;

                // Set to true if the entity is within a GravitySphere (experiences linear time)
                bool isTargetedByGravity = true;

                // Helper for nlohmann/json serialization
                NLOHMANN_DEFINE_TYPE_INTRUSIVE(TimeComponent, density, localTimeMultiplier, isTargetedByGravity, subjectiveDt)
            };
        } // namespace components
    } // namespace ecs
} // namespace Umgebung