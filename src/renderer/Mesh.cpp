#include "umgebung/renderer/Mesh.hpp"

namespace Umgebung::renderer {

    // Factory function implementation
    std::shared_ptr<Mesh> Mesh::create(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices) {
        return std::shared_ptr<Mesh>(new Mesh(vertices, indices));
    }

    // Private constructor called by the factory
    Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices) {
        indexCount_ = indices.size();
        setupMesh(vertices, indices);
    }

    // Destructor
    Mesh::~Mesh() {
        // Clean up the buffers from the GPU when the mesh is destroyed
        glDeleteVertexArrays(1, &VAO_);
        glDeleteBuffers(1, &VBO_);
        glDeleteBuffers(1, &EBO_);
    }

    void Mesh::setupMesh(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices) {
        // 1. Create buffers/arrays
        glGenVertexArrays(1, &VAO_);
        glGenBuffers(1, &VBO_);
        glGenBuffers(1, &EBO_);

        // 2. Bind the VAO. From now on, any buffer and attribute pointer calls will be stored in this VAO.
        glBindVertexArray(VAO_);

        // 3. Copy our vertices array into a vertex buffer for OpenGL to use
        glBindBuffer(GL_ARRAY_BUFFER, VBO_);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        // 4. Copy our index array into an element buffer
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);

        // 5. Set the vertex attribute pointers
        // Vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // Vertex Normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
        // Vertex Texture Coords
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));

        // Unbind the VAO. It's good practice to unbind to prevent accidental modifications.
        glBindVertexArray(0);
    }

    void Mesh::draw() const {
        // Bind the VAO that contains our buffer configurations
        glBindVertexArray(VAO_);
        // Draw the mesh
        glDrawElements(GL_TRIANGLES, indexCount_, GL_UNSIGNED_INT, 0);
        // Unbind the VAO
        glBindVertexArray(0);
    }

} // namespace Umgebung::renderer