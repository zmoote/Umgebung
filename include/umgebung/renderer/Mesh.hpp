#pragma once

#include <glad/glad.h> // For GLuint
#include <glm/glm.hpp>

#include <vector>
#include <memory> // For std::shared_ptr

namespace Umgebung::renderer {

    /**
     * @struct Vertex
     * @brief Represents a single vertex with position, normal, and texture coordinates.
     */
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 texCoords;
    };

    /**
     * @class Mesh
     * @brief Manages the vertex data and OpenGL buffers (VAO, VBO, EBO) for a renderable model.
     */
    class Mesh {
    public:
        // Factory function for creating a shared_ptr to a Mesh
        static std::shared_ptr<Mesh> create(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices);

        // Destructor to clean up GPU resources
        ~Mesh();

        // Bind the mesh's VAO and draw it
        void draw() const;

    private:
        // Private constructor to enforce creation via the factory function
        Mesh(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices);

        // Helper function to set up the OpenGL buffers
        void setupMesh(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices);

        GLuint VAO_{ 0 }; // Vertex Array Object
        GLuint VBO_{ 0 }; // Vertex Buffer Object
        GLuint EBO_{ 0 }; // Element Buffer Object

        GLsizei indexCount_{ 0 }; // How many indices to draw
    };

} // namespace Umgebung::renderer