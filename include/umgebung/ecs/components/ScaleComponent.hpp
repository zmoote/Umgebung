#pragma once

#include <nlohmann/json.hpp>

namespace Umgebung::ecs::components {

    enum class ScaleType {
        Quantum,
        Micro,
        Human,
        Planetary,
        SolarSystem,
        Galactic,
        ExtraGalactic,
        Universal,
        Multiversal
    };

    struct ScaleComponent {
        ScaleType type = ScaleType::Human;
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(ScaleType, {
        {ScaleType::Quantum, "Quantum"},
        {ScaleType::Micro, "Micro"},
        {ScaleType::Human, "Human"},
        {ScaleType::Planetary, "Planetary"},
        {ScaleType::SolarSystem, "SolarSystem"},
        {ScaleType::Galactic, "Galactic"},
        {ScaleType::ExtraGalactic, "ExtraGalactic"},
        {ScaleType::Universal, "Universal"},
        {ScaleType::Multiversal, "Multiversal"}
    })

    inline void to_json(nlohmann::json& j, const ScaleComponent& s) {
        j = nlohmann::json{ {"type", s.type} };
    }

    inline void from_json(const nlohmann::json& j, ScaleComponent& s) {
        j.at("type").get_to(s.type);
    }

} // namespace Umgebung::ecs::components
