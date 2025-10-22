#pragma once

#include "umgebung/renderer/Mesh.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string> // <-- 1. Add this include
#include <nlohmann/json.hpp> // <-- 2. Add this include

namespace Umgebung::ecs::components {

    struct Renderable {
        std::shared_ptr<renderer::Mesh> mesh;
        glm::vec4 color{ 1.0f, 1.0f, 1.0f, 1.0f };

        // --- 3. Add this tag ---
        std::string meshTag;

        // 4. Update constructor (optional but good practice)
        Renderable(std::shared_ptr<renderer::Mesh> m,
            glm::vec4 c = { 1.0f, 1.0f, 1.0f, 1.0f },
            std::string tag = "")
            : mesh(m), color(c), meshTag(tag) {
        }

        // 5. Default constructor for serialization
        Renderable() = default;
    };

    // 6. Teach JSON how to save/load this component
    //    We will NOT save the 'mesh' pointer. We only save the color and tag.
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Renderable, color, meshTag)

} // namespace Umgebung::ecs::components