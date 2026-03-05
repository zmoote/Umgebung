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

        if (instanceVAO_) glDeleteVertexArrays(1, &instanceVAO_);
        if (instanceVBO_) glDeleteBuffers(1, &instanceVBO_);
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

    void Mesh::drawInstanced(const std::vector<InstanceData>& instanceData) const {
        if (instanceData.empty()) return;

        if (instanceVAO_ == 0) {
            glGenVertexArrays(1, &instanceVAO_);
            glGenBuffers(1, &instanceVBO_);

            glBindVertexArray(instanceVAO_);

            glBindBuffer(GL_ARRAY_BUFFER, VBO_);
            if (!indices_.empty()) glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_);

            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
            glEnableVertexAttribArray(2);
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));

            glBindBuffer(GL_ARRAY_BUFFER, instanceVBO_);

            // Model Matrix (slots 3, 4, 5, 6)
            for (int i = 0; i < 4; i++) {
                glEnableVertexAttribArray(3 + i);
                glVertexAttribPointer(3 + i, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)(offsetof(InstanceData, modelMatrix) + i * sizeof(glm::vec4)));
                glVertexAttribDivisor(3 + i, 1);
            }

            // Color (slot 7)
            glEnableVertexAttribArray(7);
            glVertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, color));
            glVertexAttribDivisor(7, 1);

            // Density (slot 8)
            glEnableVertexAttribArray(8);
            glVertexAttribPointer(8, 1, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, density));
            glVertexAttribDivisor(8, 1);

            // Phryll Influence (slot 9)
            glEnableVertexAttribArray(9);
            glVertexAttribPointer(9, 1, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, phryllInfluence));
            glVertexAttribDivisor(9, 1);

            // Selected (slot 10)
            glEnableVertexAttribArray(10);
            glVertexAttribPointer(10, 1, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, selected));
            glVertexAttribDivisor(10, 1);

            // Is Manifesting (slot 11)
            glEnableVertexAttribArray(11);
            glVertexAttribPointer(11, 1, GL_FLOAT, GL_FALSE, sizeof(InstanceData), (void*)offsetof(InstanceData, isManifesting));
            glVertexAttribDivisor(11, 1);

            glBindVertexArray(0);
        }

        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO_);
        glBufferData(GL_ARRAY_BUFFER, instanceData.size() * sizeof(InstanceData), instanceData.data(), GL_STREAM_DRAW);

        glBindVertexArray(instanceVAO_);
        if (drawMode_ == GL_POINTS) {
            glDrawArraysInstanced(GL_POINTS, 0, static_cast<GLsizei>(vertices_.size()), static_cast<GLsizei>(instanceData.size()));
        } else {
            glDrawElementsInstanced(drawMode_, indexCount_, GL_UNSIGNED_INT, 0, static_cast<GLsizei>(instanceData.size()));
        }
        glBindVertexArray(0);
    }

}