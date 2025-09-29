#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <vector>
#include <memory>

namespace Umgebung::renderer {

    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 texCoords;
    };

    class Mesh {
    public:
        static std::shared_ptr<Mesh> create(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices);

        ~Mesh();

        void draw() const;

    private:
        Mesh(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices);

        void setupMesh(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices);

        GLuint VAO_{ 0 };
        GLuint VBO_{ 0 };
        GLuint EBO_{ 0 };

        GLsizei indexCount_{ 0 };
    };

}