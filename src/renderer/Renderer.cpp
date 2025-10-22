#include "umgebung/renderer/Renderer.hpp"
#include <glad/glad.h>

namespace Umgebung::renderer {
    void Renderer::init() {
        shader_ = std::make_unique<gl::Shader>("assets/shaders/simple.vert", "assets/shaders/simple.frag");
        camera_ = std::make_unique<Camera>();
        camera_->setPerspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
        camera_->setPosition({ 0.0f, 0.0f, 3.0f });

        // Create the primitive triangle mesh
        std::vector<renderer::Vertex> vertices = {
            {{-0.5f, -0.5f, 0.0f}, {0,0,1}, {0,0}},
            {{ 0.5f, -0.5f, 0.0f}, {0,0,1}, {0,0}},
            {{ 0.0f,  0.5f, 0.0f}, {0,0,1}, {0,0}}
        };
        std::vector<uint32_t> indices = { 0, 1, 2 };
        m_TriangleMesh = renderer::Mesh::create(vertices, indices);
    }

    std::shared_ptr<Mesh> Renderer::getTriangleMesh() const {
        return m_TriangleMesh;
    }
}