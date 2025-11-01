#include "umgebung/scene/SceneSerializer.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/renderer/Renderer.hpp"     // <-- 1. Add include
#include "umgebung/asset/ModelLoader.hpp"   // <-- 2. Add include
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Name.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/util/JsonHelpers.hpp"
#include "umgebung/util/LogMacros.hpp" // <-- 3. Add include

#include <entt/entt.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>

namespace Umgebung::scene {

    using Transform = Umgebung::ecs::components::Transform;
    using Name = Umgebung::ecs::components::Name;
    using Renderable = Umgebung::ecs::components::Renderable;

    SceneSerializer::SceneSerializer(Scene* scene, renderer::Renderer* renderer)
        : m_Scene(scene), m_Renderer(renderer) {
    }

    // --- serialize function (SAVING) is unchanged ---
    void SceneSerializer::serialize(const std::string& filepath) {
        if (!m_Scene) return;

        nlohmann::json sceneJson;
        auto& registry = m_Scene->getRegistry();

        nlohmann::json entityList = nlohmann::json::array();

        registry.view<entt::entity>().each([&](auto entity) {
            nlohmann::json entityJson;
            entityJson["id"] = entity;

            if (registry.all_of<Transform>(entity)) {
                entityJson["transform"] = registry.get<Transform>(entity);
            }
            if (registry.all_of<Name>(entity)) {
                entityJson["name"] = registry.get<Name>(entity);
            }
            if (registry.all_of<Renderable>(entity)) {
                entityJson["renderable"] = registry.get<Renderable>(entity);
            }

            entityList.push_back(entityJson);
            });

        sceneJson["entities"] = entityList;

        std::ofstream outFile(filepath);
        if (outFile.is_open()) {
            outFile << sceneJson.dump(4);
            outFile.close();
            UMGEBUNG_LOG_INFO("Scene saved to {}", filepath);
        }
        else {
            UMGEBUNG_LOG_ERROR("Could not open file for writing: {}", filepath);
        }
    }


    // --- deserialize function (LOADING) is CHANGED ---
    bool SceneSerializer::deserialize(const std::string& filepath) {
        if (!m_Scene || !m_Renderer) {
            UMGEBUNG_LOG_ERROR("Serializer not initialized.");
            return false;
        }

        std::ifstream inFile(filepath);
        if (!inFile.is_open()) {
            UMGEBUNG_LOG_ERROR("Could not open file for reading: {}", filepath);
            return false;
        }
        nlohmann::json sceneJson;
        try {
            sceneJson = nlohmann::json::parse(inFile);
            inFile.close();
        }
        catch (nlohmann::json::parse_error& e) {
            UMGEBUNG_LOG_ERROR("Failed to parse scene file {}: {}", filepath, e.what());
            inFile.close();
            return false;
        }

        auto& registry = m_Scene->getRegistry();
        registry.clear();
        m_Scene->setSelectedEntity(entt::null);

        if (!sceneJson.contains("entities")) {
            UMGEBUNG_LOG_WARN("Scene file is empty or invalid: {}", filepath);
            return false;
        }

        for (const auto& entityJson : sceneJson["entities"]) {

            entt::entity entity = registry.create(entityJson["id"].get<entt::entity>());

            if (entityJson.contains("transform")) {
                registry.emplace<Transform>(entity, entityJson["transform"].get<Transform>());
            }
            if (entityJson.contains("name")) {
                registry.emplace<Name>(entity, entityJson["name"].get<Name>());
            }

            // --- 4. Update Renderable deserialization logic ---
            if (entityJson.contains("renderable")) {
                auto renderable = entityJson["renderable"].get<Renderable>();

                // Check the tag and re-assign the mesh pointer
                if (renderable.meshTag == "primitive_triangle") {
                    renderable.mesh = m_Renderer->getTriangleMesh();
                }
                // --- REMOVE THE CUBE-SPECIFIC LOGIC ---
                // else if (renderable.meshTag == "primitive_cube") { ... }

                // --- ADD THIS NEW LOGIC ---
                else if (!renderable.meshTag.empty()) {
                    // If the tag is not empty and not a primitive,
                    // assume it's a file path and try to load it.
                    renderable.mesh = m_Renderer->getModelLoader()->loadMesh(renderable.meshTag);

                    if (!renderable.mesh) {
                        // Log a warning if loading failed
                        UMGEBUNG_LOG_WARN("Failed to load mesh: {}", renderable.meshTag);
                    }
                }
                // --- End of new logic ---

                // Emplace the component (with either a valid mesh or nullptr)
                registry.emplace<Renderable>(entity, renderable);
            }
        }

        UMGEBUNG_LOG_INFO("Scene loaded from {}", filepath);
        return true;
    }

} // namespace Umgebung::scene