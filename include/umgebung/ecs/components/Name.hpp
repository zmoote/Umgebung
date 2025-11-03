/**
 * @file Name.hpp
 * @brief Contains the Name component.
 */
#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace Umgebung::ecs::components {

    /**
     * @brief A component that gives an entity a name.
     */
    struct Name {
        std::string name; ///< The name of the entity.

        /**
         * @brief Default constructor.
         */
        Name() = default;

        /**
         * @brief Copy constructor.
         */
        Name(const Name&) = default;

        /**
         * @brief Construct a new Name object with a given name.
         * @param n The name.
         */
        Name(const std::string& n) : name(n) {}
    };

    // Teach the JSON library how to save/load this component
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Name, name)

}