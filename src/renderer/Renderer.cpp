/**
 * @file Renderer.cpp
 * @brief Implements the Renderer class.
 */
#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/renderer/Mesh.hpp"
#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/renderer/Camera.hpp"
#include "umgebung/asset/ModelLoader.hpp" 

#include <glad/glad.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> // For glm::radians

namespace Umgebung::renderer {

    Renderer::Renderer() = default;
    Renderer::~Renderer() = default;

    void Renderer::init() {
        // Enable global OpenGL states
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_DEPTH_TEST);

        shader_ = std::make_unique<gl::Shader>("assets/shaders/simple.vert", "assets/shaders/simple.frag");
        pointShader_ = std::make_unique<gl::Shader>("assets/shaders/point_lod.vert", "assets/shaders/point_lod.frag");

        camera_ = std::make_unique<Camera>();

        // Set default perspective
        camera_->setPerspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 1000.0f);

        m_ModelLoader = std::make_unique<asset::ModelLoader>();

        // Create Triangle Mesh
        {
            std::vector<renderer::Vertex> vertices = {
                {{-0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                {{ 0.5f, -0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                {{ 0.0f,  0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}}
            };
            std::vector<uint32_t> indices = { 0, 1, 2 };
            m_TriangleMesh = renderer::Mesh::create(vertices, indices);
        }

        // Create Point Mesh (Single vertex at origin)
        {
            std::vector<renderer::Vertex> vertices = {
                {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}}
            };
            std::vector<uint32_t> indices = {}; // No indices needed for GL_POINTS
            m_PointMesh = renderer::Mesh::create(vertices, indices);
            m_PointMesh->setDrawMode(GL_POINTS);
        }
    }

    void Renderer::shutdown() {
        // All unique_ptrs are auto-destroyed
    }

    gl::Shader& Renderer::getShader() {
        return *shader_;
    }

    gl::Shader& Renderer::getPointShader() {
        return *pointShader_;
    }

    Camera& Renderer::getCamera() {
        return *camera_;
    }

    const glm::mat4& Renderer::getViewMatrix() const {
        return camera_->getViewMatrix();
    }

    const glm::mat4& Renderer::getProjectionMatrix() const {
        return camera_->getProjectionMatrix();
    }

    std::shared_ptr<Mesh> Renderer::getTriangleMesh() const {
        return m_TriangleMesh;
    }

    std::shared_ptr<Mesh> Renderer::getPointMesh() const {
        return m_PointMesh;
    }

    asset::ModelLoader* Renderer::getModelLoader() const {
        return m_ModelLoader.get();
    }

} // namespace Umgebung::renderer