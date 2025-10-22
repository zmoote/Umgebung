#pragma once

#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/renderer/Camera.hpp"
#include "umgebung/renderer/Mesh.hpp"

#include <memory>

namespace Umgebung::renderer {

    class Renderer {
    public:
        void init();

        gl::Shader& getShader() { return *shader_; }
        const glm::mat4& getViewMatrix() const { return camera_->getViewMatrix(); }
        const glm::mat4& getProjectionMatrix() const { return camera_->getProjectionMatrix(); }

        Camera& getCamera() { return *camera_; }

        std::shared_ptr<Mesh> getTriangleMesh() const;
    private:
        std::unique_ptr<gl::Shader> shader_;
        std::unique_ptr<Camera> camera_;
        std::shared_ptr<Mesh> m_TriangleMesh;
    };

} // namespace Umgebung::renderer