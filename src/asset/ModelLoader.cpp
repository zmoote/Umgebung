/**
 * @file ModelLoader.cpp
 * @brief Implements the ModelLoader class.
 */
#include "umgebung/asset/ModelLoader.hpp"
#include "umgebung/util/LogMacros.hpp" // <-- 1. ADD THIS INCLUDE

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <vector> // <-- Add include for std::vector
#include <string> // <-- Add include for std::string
#include <map>    // <-- Add include for std::map

namespace Umgebung::asset {

    ModelLoader::ModelLoader() {}
    ModelLoader::~ModelLoader() {}

    std::shared_ptr<renderer::Mesh> ModelLoader::loadMesh(const std::string& filepath) {
        // Check if mesh is already in cache
        if (m_MeshCache.count(filepath)) {
            return m_MeshCache[filepath];
        }

        UMGEBUNG_LOG_INFO("Loading model: {}", filepath); // <-- This will now compile

        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(filepath,
            aiProcess_Triangulate |
            aiProcess_GenNormals |
            aiProcess_CalcTangentSpace |
            aiProcess_FlipUVs
        );

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            UMGEBUNG_LOG_ERROR("Assimp Error while loading {}: {}", filepath, importer.GetErrorString()); // <-- This will now compile
            return nullptr;
        }

        if (scene->mNumMeshes == 0) {
            UMGEBUNG_LOG_ERROR("No meshes found in file: {}", filepath); // <-- This will now compile
            return nullptr;
        }

        aiMesh* firstMesh = scene->mMeshes[0];
        auto [vertices, indices] = processMesh(firstMesh, scene);

        if (vertices.empty()) { // Indices can be empty for non-indexed meshes, but vertices shouldn't
            UMGEBUNG_LOG_ERROR("Failed to process mesh data from: {}", filepath); // <-- This will now compile
            return nullptr;
        }

        std::shared_ptr<renderer::Mesh> mesh = renderer::Mesh::create(vertices, indices);
        m_MeshCache[filepath] = mesh;

        return mesh;
    }

    std::pair<std::vector<renderer::Vertex>, std::vector<uint32_t>> ModelLoader::processMesh(aiMesh* mesh, const aiScene* scene) {
        std::vector<renderer::Vertex> vertices;
        std::vector<uint32_t> indices;

        vertices.reserve(mesh->mNumVertices);
        for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
            renderer::Vertex vertex;

            vertex.position.x = mesh->mVertices[i].x;
            vertex.position.y = mesh->mVertices[i].y;
            vertex.position.z = mesh->mVertices[i].z;

            if (mesh->HasNormals()) {
                vertex.normal.x = mesh->mNormals[i].x;
                vertex.normal.y = mesh->mNormals[i].y;
                vertex.normal.z = mesh->mNormals[i].z;
            }
            else {
                vertex.normal = { 0.0f, 0.0f, 0.0f };
            }

            if (mesh->HasTextureCoords(0)) {
                vertex.texCoords.x = mesh->mTextureCoords[0][i].x;
                vertex.texCoords.y = mesh->mTextureCoords[0][i].y;
            }
            else {
                vertex.texCoords = { 0.0f, 0.0f };
            }

            vertices.push_back(vertex);
        }

        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            aiFace face = mesh->mFaces[i];
            for (unsigned int j = 0; j < face.mNumIndices; j++) {
                indices.push_back(face.mIndices[j]);
            }
        }

        return { vertices, indices };
    }

    void ModelLoader::processNode(aiNode* node, const aiScene* scene, std::vector<renderer::Vertex>& outVertices, std::vector<uint32_t>& outIndices) {
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            auto [vertices, indices] = processMesh(mesh, scene);

            uint32_t vertexOffset = static_cast<uint32_t>(outVertices.size());
            outVertices.insert(outVertices.end(), vertices.begin(), vertices.end());
            for (uint32_t index : indices) {
                outIndices.push_back(index + vertexOffset);
            }
        }
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNode(node->mChildren[i], scene, outVertices, outIndices);
        }
    }


} // namespace Umgebung::asset