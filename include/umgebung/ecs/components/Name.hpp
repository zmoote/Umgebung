#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace Umgebung::ecs::components {

    struct Name {
        std::string name;

        Name() = default;
        Name(const Name&) = default;
        Name(const std::string& n) : name(n) {}
    };

    // Teach the JSON library how to save/load this component
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Name, name)

} // namespace Umgebung::ecs::components