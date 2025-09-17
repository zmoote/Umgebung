#pragma once

#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/renderer/Camera.hpp"

#include <memory>

// Use the new namespace
namespace Umgebung::renderer {

    class Renderer {
    public:
        void init();

        // The old draw() method is no longer needed, as the RenderSystem handles drawing.
        // void draw(); 

        // --- New Methods ---
        gl::Shader& getShader() { return *shader_; }
        const glm::mat4& getViewMatrix() const { return camera_->getViewMatrix(); }
        const glm::mat4& getProjectionMatrix() const { return camera_->getProjectionMatrix(); }

    private:
        // The renderer now owns the main shader and camera.
        std::unique_ptr<gl::Shader> shader_;
        std::unique_ptr<Camera> camera_;

        // The old triangle members are no longer needed.
        // unsigned int m_triangleVAO;
        // unsigned int m_triangleVBO;
    };

} // namespace Umgebung::renderer