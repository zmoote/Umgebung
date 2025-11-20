#pragma once

#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/renderer/Camera.hpp"
#include <glm/glm.hpp>
#include <memory>

namespace Umgebung::renderer
{

    class DebugRenderer
    {
    public:
        void init();
        void shutdown();

        void beginFrame(const Camera& camera);
        void endFrame();

        void drawBox(const glm::mat4& transform, const glm::vec4& color);
        void drawSphere(const glm::mat4& transform, const glm::vec4& color);

    private:
        std::unique_ptr<gl::Shader> shader_;
        
        // Cube resources
        unsigned int cubeVAO_ = 0, cubeVBO_ = 0;
        
        // Sphere resources
        unsigned int sphereVAO_ = 0, sphereVBO_ = 0, sphereEBO_ = 0;
        unsigned int sphereIndexCount_ = 0;

        void setupCube();
        void setupSphere();
    };

} // namespace Umgebung::renderer
