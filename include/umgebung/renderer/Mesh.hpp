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

        GLuint VAO_{ 0 }; ///< The vertex array object.
        GLuint VBO_{ 0 }; ///< The vertex buffer object.
        GLuint EBO_{ 0 }; ///< The element buffer object.

        GLsizei indexCount_{ 0 }; ///< The number of indices in the mesh.
    };

}