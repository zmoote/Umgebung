#include "umgebung/renderer/Renderer.hpp"
#include <glad/glad.h>

// Use the new namespace and correct old one
namespace Umgebung::renderer {

    void Renderer::init() {
        // 1. Create and load the main shader program.
        // Make sure these paths are correct for your project structure.
        shader_ = std::make_unique<gl::Shader>("../assets/shaders/simple.vert", "../assets/shaders/simple.frag");

        // 2. Create the camera.
        // You can configure the starting position and projection here.
        camera_ = std::make_unique<Camera>();
        camera_->setPerspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
        camera_->setPosition({ 0.0f, 0.0f, 3.0f });

        // The old vertex and buffer setup is now handled by the Mesh class,
        // so we can remove it from here.
    }

    // The old draw() method is removed, as its job is now done by the RenderSystem.
    /*
    void Renderer::draw() {
        glBindVertexArray(m_triangleVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    */

} // namespace Umgebung::renderer