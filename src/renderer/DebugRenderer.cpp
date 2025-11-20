#include "umgebung/renderer/DebugRenderer.hpp"
#include "umgebung/util/LogMacros.hpp"
#include <glad/glad.h>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>

namespace Umgebung::renderer
{

    void DebugRenderer::init()
    {
        UMGEBUNG_LOG_INFO("Initializing DebugRenderer");
        shader_ = std::make_unique<gl::Shader>("assets/shaders/debug.vert", "assets/shaders/debug.frag");
        setupCube();
        setupSphere();
    }

    void DebugRenderer::shutdown()
    {
        UMGEBUNG_LOG_INFO("Shutting down DebugRenderer");
        glDeleteVertexArrays(1, &cubeVAO_);
        glDeleteBuffers(1, &cubeVBO_);
        glDeleteVertexArrays(1, &sphereVAO_);
        glDeleteBuffers(1, &sphereVBO_);
        glDeleteBuffers(1, &sphereEBO_);
    }

    void DebugRenderer::beginFrame(const Camera& camera)
    {
        shader_->bind();
        shader_->setMat4("view", camera.getViewMatrix());
        shader_->setMat4("projection", camera.getProjectionMatrix());
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glEnable(GL_DEPTH_TEST);
    }

    void DebugRenderer::endFrame()
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    void DebugRenderer::drawBox(const glm::mat4& transform, const glm::vec4& color)
    {
        shader_->bind();
        shader_->setMat4("model", transform);
        shader_->setVec4("color", color);
        glBindVertexArray(cubeVAO_);
        glDrawArrays(GL_LINES, 0, 24);
        glBindVertexArray(0);
    }

    void DebugRenderer::drawSphere(const glm::mat4& transform, const glm::vec4& color)
    {
        shader_->bind();
        shader_->setMat4("model", transform);
        shader_->setVec4("color", color);
        glBindVertexArray(sphereVAO_);
        glDrawElements(GL_LINES, sphereIndexCount_, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    void DebugRenderer::setupCube()
    {
        float vertices[] = {
            // positions
            -0.5f, -0.5f, -0.5f,
             0.5f, -0.5f, -0.5f,
             0.5f,  0.5f, -0.5f,
            -0.5f,  0.5f, -0.5f,
            -0.5f, -0.5f,  0.5f,
             0.5f, -0.5f,  0.5f,
             0.5f,  0.5f,  0.5f,
            -0.5f,  0.5f,  0.5f,
        };

        unsigned int lines[] = {
            0, 1, 1, 2, 2, 3, 3, 0, // bottom face
            4, 5, 5, 6, 6, 7, 7, 4, // top face
            0, 4, 1, 5, 2, 6, 3, 7  // connecting lines
        };
        
        float lineVertices[24 * 3];
        for(int i = 0; i < 12; ++i) {
            lineVertices[i*6+0] = vertices[lines[i*2+0]*3+0];
            lineVertices[i*6+1] = vertices[lines[i*2+0]*3+1];
            lineVertices[i*6+2] = vertices[lines[i*2+0]*3+2];
            lineVertices[i*6+3] = vertices[lines[i*2+1]*3+0];
            lineVertices[i*6+4] = vertices[lines[i*2+1]*3+1];
            lineVertices[i*6+5] = vertices[lines[i*2+1]*3+2];
        }

        glGenVertexArrays(1, &cubeVAO_);
        glGenBuffers(1, &cubeVBO_);
        glBindVertexArray(cubeVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(lineVertices), lineVertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    void DebugRenderer::setupSphere()
    {
        std::vector<glm::vec3> positions;
        std::vector<unsigned int> indices;
        const unsigned int X_SEGMENTS = 16;
        const unsigned int Y_SEGMENTS = 16;
        const float PI = 3.14159265359f;
        for (unsigned int y = 0; y <= Y_SEGMENTS; ++y)
        {
            for (unsigned int x = 0; x <= X_SEGMENTS; ++x)
            {
                float xSegment = (float)x / (float)X_SEGMENTS;
                float ySegment = (float)y / (float)Y_SEGMENTS;
                float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
                float yPos = std::cos(ySegment * PI);
                float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
                positions.push_back(glm::vec3(xPos, yPos, zPos) * 0.5f);
            }
        }

        for (unsigned int y = 0; y < Y_SEGMENTS; ++y)
        {
            for (unsigned int x = 0; x < X_SEGMENTS; ++x)
            {
                indices.push_back(y * (X_SEGMENTS + 1) + x);
                indices.push_back(y * (X_SEGMENTS + 1) + x + 1);
                indices.push_back(y * (X_SEGMENTS + 1) + x);
                indices.push_back((y+1) * (X_SEGMENTS + 1) + x);
            }
        }
        sphereIndexCount_ = indices.size();

        glGenVertexArrays(1, &sphereVAO_);
        glGenBuffers(1, &sphereVBO_);
        glGenBuffers(1, &sphereEBO_);

        glBindVertexArray(sphereVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, sphereVBO_);
        glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(glm::vec3), &positions[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glBindVertexArray(0);
    }
} // namespace Umgebung::renderer
