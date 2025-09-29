#pragma once

#include "umgebung/renderer/Mesh.hpp"

#include <glm/glm.hpp>
#include <memory>

namespace Umgebung::ecs::components {

    struct RenderableComponent {
        
        std::shared_ptr<renderer::Mesh> mesh;

        glm::vec4 color{ 1.0f, 1.0f, 1.0f, 1.0f };

        RenderableComponent() = default;

        RenderableComponent(std::shared_ptr<renderer::Mesh> m, const glm::vec4& c = { 1.0f, 1.0f, 1.0f, 1.0f })
            : mesh(std::move(m)), color(c) {
        }
    };

}