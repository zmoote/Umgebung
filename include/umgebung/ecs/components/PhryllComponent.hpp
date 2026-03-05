/**
 * @file PhryllComponent.hpp
 * @brief Contains the PhryllComponent for life force and observer interaction.
 */
#pragma once

#include <nlohmann/json.hpp>

namespace Umgebung::ecs::components {

    /**
     * @brief A component that models the interaction between consciousness (the observer) 
     * and the scalar field (life force energy / Phryll).
     */
    struct PhryllComponent {
        
        float density = 0.5f;          ///< Local Phryll density (0.0 to 1.0).
        float observerInfluence = 0.0f; ///< How much the current observer is affecting this entity.
        bool isManifesting = true;     ///< Whether the entity is "solidified" in the current density.
        
        float baseFrequency = 432.0f;  ///< Base vibrational frequency in Hz.
        float currentFrequency = 432.0f; ///< Current frequency, modified by observation.

        PhryllComponent() = default;
    };

    /**
     * @brief JSON serialization.
     */
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PhryllComponent, density, observerInfluence, isManifesting, baseFrequency, currentFrequency)

}