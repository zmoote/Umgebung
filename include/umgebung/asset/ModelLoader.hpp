#pragma once

#include "umgebung/renderer/Mesh.hpp" // For Mesh and Vertex
#include <string>
#include <vector>
#include <memory> // For std::shared_ptr
#include <map>    // For caching loaded models

// Forward declarations for Assimp classes
struct aiNode;
struct aiMesh;
struct aiScene;

namespace Umgebung::asset {

    /**
 * @brief A class for loading 3D models from files.
 * 
 * This class uses Assimp to load 3D models from files and caches the loaded meshes.
 */
class ModelLoader {
public:
    /**
     * @brief Construct a new Model Loader object.
     */
    ModelLoader();

    /**
     * @brief Destroy the Model Loader object.
     */
    ~ModelLoader();

    /**
     * @brief Loads a model file and returns the first mesh found.
     * 
     * Caches the mesh so subsequent calls for the same file are fast.
     * @param filepath Path to the model file (e.g., "assets/models/Cube.glb").
     * @return A shared_ptr to the loaded Mesh, or nullptr if loading fails.
     */
    std::shared_ptr<renderer::Mesh> loadMesh(const std::string& filepath);

private:
    /**
     * @brief Recursively processes nodes in the Assimp scene graph.
     * 
     * @param node The current Assimp node.
     * @param scene The Assimp scene.
     * @param outVertices The output vector of vertices.
     * @param outIndices The output vector of indices.
     */
    void processNode(aiNode* node, const aiScene* scene, std::vector<renderer::Vertex>& outVertices, std::vector<uint32_t>& outIndices);

    /**
     * @brief Processes a single Assimp mesh and converts it to our Vertex/index format.
     * 
     * @param mesh The Assimp mesh.
     * @param scene The Assimp scene.
     * @return A pair of vectors containing the vertices and indices of the mesh.
     */
    std::pair<std::vector<renderer::Vertex>, std::vector<uint32_t>> processMesh(aiMesh* mesh, const aiScene* scene);

    // Cache for already loaded meshes
    std::map<std::string, std::shared_ptr<renderer::Mesh>> m_MeshCache; ///< The cache of loaded meshes.
};

} // namespace Umgebung::asset