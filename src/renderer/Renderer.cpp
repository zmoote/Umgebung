#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/renderer/Mesh.hpp"
#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/renderer/Camera.hpp"
#include "umgebung/asset/ModelLoader.hpp" 
// We are no longer using PrimitiveMeshes
// #include "umgebung/renderer/PrimitiveMeshes.hpp" 

#include <glad/glad.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> // For glm::radians

namespace Umgebung::renderer {

    Renderer::Renderer() = default;
    Renderer::~Renderer() = default;

    void Renderer::init() {
        shader_ = std::make_unique<gl::Shader>("assets/shaders/simple.vert", "assets/shaders/simple.frag");

        // --- FIX: Use default constructor for Camera ---
        camera_ = std::make_unique<Camera>();
        // --- END FIX ---

        // Set default perspective
        camera_->setPerspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 1000.0f);

        m_ModelLoader = std::make_unique<asset::ModelLoader>();

        // --- Create Triangle Mesh ---
        {
            // Note the extra braces for Vertex struct initialization
            std::vector<renderer::Vertex> vertices = {
                {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                {{ 0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                {{ 0.0f,  0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}}
            };
            std::vector<uint32_t> indices = { 0, 1, 2 };
            m_TriangleMesh = renderer::Mesh::create(vertices, indices);
        }
    }

    void Renderer::shutdown() {
        // All unique_ptrs are auto-destroyed
    }

    gl::Shader& Renderer::getShader() {
        return *shader_;
    }

    Camera& Renderer::getCamera() {
        return *camera_;
    }

    // --- ADD THESE GETTERS BACK ---
    const glm::mat4& Renderer::getViewMatrix() const {
        return camera_->getViewMatrix();
    }

    const glm::mat4& Renderer::getProjectionMatrix() const {
        return camera_->getProjectionMatrix();
    }
    // --- END ADD ---

    std::shared_ptr<Mesh> Renderer::getTriangleMesh() const {
        return m_TriangleMesh;
    }

    asset::ModelLoader* Renderer::getModelLoader() const {
        return m_ModelLoader.get();
    }

} // namespace Umgebung::renderer