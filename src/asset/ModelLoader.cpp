#include "umgebung/asset/ModelLoader.hpp"
#include "umgebung/util/LogMacros.hpp" // For logging

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

namespace Umgebung::asset {

    ModelLoader::ModelLoader() {}
    ModelLoader::~ModelLoader() {}

    std::shared_ptr<renderer::Mesh> ModelLoader::loadMesh(const std::string& filepath) {
        // Check if mesh is already in cache
        if (m_MeshCache.count(filepath)) {
            return m_MeshCache[filepath];
        }

        UMGEBUNG_INFO("Loading model: {}", filepath);

        Assimp::Importer importer;
        // aiProcess_Triangulate: Guarantees all faces are triangles
        // aiProcess_GenNormals: Creates normals if they don't exist
        // aiProcess_CalcTangentSpace: Calculates tangents (good for lighting)
        // aiProcess_FlipUVs: Flips texture coordinates (common for OpenGL)
        const aiScene* scene = importer.ReadFile(filepath,
            aiProcess_Triangulate |
            aiProcess_GenNormals |
            aiProcess_CalcTangentSpace |
            aiProcess_FlipUVs
        );

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            UMGEBUNG_ERROR("Assimp Error while loading {}: {}", filepath, importer.GetErrorString());
            return nullptr;
        }

        // For now, we just process the first mesh we find in the file.
        // A more complex model might have multiple meshes.
        if (scene->mNumMeshes == 0) {
            UMGEBUNG_ERROR("No meshes found in file: {}", filepath);
            return nullptr;
        }

        aiMesh* firstMesh = scene->mMeshes[0];
        auto [vertices, indices] = processMesh(firstMesh, scene);

        if (vertices.empty() || indices.empty()) {
            UMGEBUNG_ERROR("Failed to process mesh data from: {}", filepath);
            return nullptr;
        }

        // Create the mesh and store it in the cache
        std::shared_ptr<renderer::Mesh> mesh = renderer::Mesh::create(vertices, indices);
        m_MeshCache[filepath] = mesh;

        return mesh;
    }

    // This function is complex, but it's just "data plumbing"
    // from Assimp's format to our Vertex struct format.
    std::pair<std::vector<renderer::Vertex>, std::vector<uint32_t>> ModelLoader::processMesh(aiMesh* mesh, const aiScene* scene) {
        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;

        vertices.reserve(mesh->mNumVertices);
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            renderer::Vertex vertex;

            // Position
            vertex.position.x = mesh->mVertices[i].x;
            vertex.position.y = mesh->mVertices[i].y;
            vertex.position.z = mesh->mVertices[i].z;

            // Normals
            if (mesh->HasNormals()) {
                vertex.normal.x = mesh->mNormals[i].x;
                vertex.normal.y = mesh->mNormals[i].y;
                vertex.normal.z = mesh->mNormals[i].z;
            }
            else {
                vertex.normal = { 0.0f, 0.0f, 0.0f };
            }

            // Texture Coordinates (UVs)
            // We just take the first set of UVs (index 0)
            if (mesh->HasTextureCoords(0)) {
                vertex.texCoords.x = mesh->mTextureCoords[0][i].x;
                vertex.texCoords.y = mesh->mTextureCoords[0][i].y;
            }
            else {
                vertex.texCoords = { 0.0f, 0.0f };
            }

            // Tangents (placeholder for now)
            // if (mesh->HasTangentsAndBitangents()) { ... }

            vertices.push_back(vertex);
        }

        // Indices
        // Iterate over all faces (which are guaranteed to be triangles)
        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                indices.push_back(face.mIndices[j]);
            }
        }

        return { vertices, indices };
    }

    // We don't need this yet, but it's here for completeness if you load complex scenes
    void ModelLoader::processNode(aiNode* node, const aiScene* scene, std::vector<renderer::Vertex>& outVertices, std::vector<uint32_t>& outIndices) {
        // Process all the node's meshes (if any)
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            auto [vertices, indices] = processMesh(mesh, scene);

            uint32_t vertexOffset = static_cast<uint32_t>(outVertices.size());
            outVertices.insert(outVertices.end(), vertices.begin(), vertices.end());
            for (uint32_t index : indices) {
                outIndices.push_back(index + vertexOffset);
            }
        }
        // Then recurse on all of the node's children
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNode(node->mChildren[i], scene, outVertices, outIndices);
        }
    }


} // namespace Umgebung::asset