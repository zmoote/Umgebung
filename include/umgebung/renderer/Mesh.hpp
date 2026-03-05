/**
 * @file Mesh.hpp
 * @brief Contains the Mesh class.
 */
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <vector>
#include <memory>

namespace Umgebung::renderer {

    /**
     * @brief A struct representing a vertex.
     */
    struct Vertex {
        glm::vec3 position;  ///< The position of the vertex.
        glm::vec3 normal;    ///< The normal of the vertex.
        glm::vec2 texCoords; ///< The texture coordinates of the vertex.
    };

    /**
     * @brief A struct representing per-instance data for instanced rendering.
     */
    struct InstanceData {
        glm::mat4 modelMatrix;
        glm::vec4 color;
        float density;
        float phryllInfluence;
        float selected; // 0.0f or 1.0f
        float isManifesting; // 0.0f or 1.0f
    };

    /**
     * @brief A class representing a 3D mesh.
     */
    class Mesh {
    public:
        /**
         * @brief Creates a new Mesh object.
         * 
         * @param vertices The vertices of the mesh.
         * @param indices The indices of the mesh.
         * @return A shared pointer to the new Mesh object.
         */
        static std::shared_ptr<Mesh> create(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices);

        /**
         * @brief Destroy the Mesh object.
         */
        ~Mesh();

        /**
         * @brief Draws the mesh.
         */
        void draw() const;

        /**
         * @brief Draws multiple instances of the mesh.
         * @param instanceData A vector containing data for each instance.
         */
        void drawInstanced(const std::vector<InstanceData>& instanceData) const;

        void setDrawMode(GLenum mode) { drawMode_ = mode; }

        /**
         * @brief Gets the vertices of the mesh.
         * @return A const reference to the vector of vertices.
         */
        const std::vector<Vertex>& getVertices() const { return vertices_; }

        /**
         * @brief Gets the indices of the mesh.
         * @return A const reference to the vector of indices.
         */
        const std::vector<GLuint>& getIndices() const { return indices_; }

    private:
        /**
         * @brief Construct a new Mesh object.
         * 
         * @param vertices The vertices of the mesh.
         * @param indices The indices of the mesh.
         */
        Mesh(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices);

        /**
         * @brief Sets up the mesh.
         * 
         * @param vertices The vertices of the mesh.
         * @param indices The indices of the mesh.
         */
        void setupMesh(const std::vector<Vertex>& vertices, const std::vector<GLuint>& indices);

        std::vector<Vertex> vertices_;
        std::vector<GLuint> indices_;

        GLuint VAO_{ 0 }; ///< The vertex array object.
        GLuint VBO_{ 0 }; ///< The vertex buffer object.
        GLuint EBO_{ 0 }; ///< The element buffer object.

        mutable GLuint instanceVAO_{ 0 }; ///< VAO for instanced rendering.
        mutable GLuint instanceVBO_{ 0 }; ///< VBO for instance data.

        GLsizei indexCount_{ 0 }; ///< The number of indices in the mesh.
        GLenum drawMode_ = GL_TRIANGLES;
    };

}