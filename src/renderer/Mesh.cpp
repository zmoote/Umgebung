/**
 * @file Mesh.cpp
 * @brief Implements the Mesh class.
 */
#include "umgebung/renderer/Mesh.hpp"

namespace Umgebung::renderer {

    std::shared_ptr<Mesh> Mesh::create(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices) {
        return std::shared_ptr<Mesh>(new Mesh(vertices, indices));
    }

    Mesh::Mesh(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices) 
    : vertices_(vertices), indices_(indices)
    {
        indexCount_ = indices.size();
        setupMesh(vertices, indices);
    }

    // Destructor
    Mesh::~Mesh() {
        glDeleteVertexArrays(1, &VAO_);
        glDeleteBuffers(1, &VBO_);
        glDeleteBuffers(1, &EBO_);
    }

    void Mesh::setupMesh(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices) {
        glGenVertexArrays(1, &VAO_);
        glGenBuffers(1, &VBO_);
        glGenBuffers(1, &EBO_);

        glBindVertexArray(VAO_);

        glBindBuffer(GL_ARRAY_BUFFER, VBO_);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        if (!indices.empty()) {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), &indices[0], GL_STATIC_DRAW);
        }

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));

        glBindVertexArray(0);
    }

    void Mesh::draw() const {
        glBindVertexArray(VAO_);

        if (drawMode_ == GL_POINTS) {
             glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(vertices_.size()));
        } else {
             glDrawElements(drawMode_, indexCount_, GL_UNSIGNED_INT, 0);
        }

        glBindVertexArray(0);
    }

}