#pragma once

#include "umgebung/renderer/Mesh.hpp"

#include <glm/glm.hpp>
#include <memory>

namespace Umgebung::ecs::components {

    /**
     * @struct RenderableComponent
     * @brief Attaches a mesh and material properties to an entity, making it visible.
     */
    struct RenderableComponent {
        // A shared pointer to the mesh data on the GPU.
        // Multiple entities can share the same mesh.
        std::shared_ptr<renderer::Mesh> mesh;

        // A color tint for this specific entity.
        glm::vec4 color{ 1.0f, 1.0f, 1.0f, 1.0f }; // Default to white

        RenderableComponent() = default;

        RenderableComponent(std::shared_ptr<renderer::Mesh> m, const glm::vec4& c = { 1.0f, 1.0f, 1.0f, 1.0f })
            : mesh(std::move(m)), color(c) {
        }
    };

} // namespace Umgebung::ecs::components