#include "umgebung/scene/SceneSerializer.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/renderer/Renderer.hpp" // <-- 1. Add include
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Name.hpp"
#include "umgebung/ecs/components/Renderable.hpp" // <-- 2. Add include
#include "umgebung/util/JsonHelpers.hpp"

#include <entt/entt.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

namespace Umgebung::scene {

    // --- 3. Add aliases ---
    using Transform = Umgebung::ecs::components::Transform;
    using Name = Umgebung::ecs::components::Name;
    using Renderable = Umgebung::ecs::components::Renderable;

    // --- 4. Update constructor ---
    SceneSerializer::SceneSerializer(Scene* scene, renderer::Renderer* renderer)
        : m_Scene(scene), m_Renderer(renderer) {
    }

    void SceneSerializer::serialize(const std::string& filepath) {
        if (!m_Scene) return;

        nlohmann::json sceneJson;
        auto& registry = m_Scene->getRegistry();

        nlohmann::json entityList = nlohmann::json::array();

        registry.view<entt::entity>().each([&](auto entity) {
            nlohmann::json entityJson;
            entityJson["id"] = entity;

            // --- Serialize components ---
            if (registry.all_of<Transform>(entity)) {
                entityJson["transform"] = registry.get<Transform>(entity);
            }
            if (registry.all_of<Name>(entity)) {
                entityJson["name"] = registry.get<Name>(entity);
            }

            // --- 5. Add block to serialize Renderable ---
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
            std::cout << "Scene saved to " << filepath << std::endl;
        }
        else {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        }
    }

    bool SceneSerializer::deserialize(const std::string& filepath) {
        // --- 6. Check if m_Renderer is valid ---
        if (!m_Scene || !m_Renderer) return false;

        // ... (file loading and JSON parsing) ...
        std::ifstream inFile(filepath);
        if (!inFile.is_open()) {
            std::cerr << "Error: Could not open file for reading: " << filepath << std::endl;
            return false;
        }
        nlohmann::json sceneJson;
        try {
            sceneJson = nlohmann::json::parse(inFile);
            inFile.close();
        }
        catch (nlohmann::json::parse_error& e) {
            std::cerr << "Error: Failed to parse scene file: " << e.what() << std::endl;
            inFile.close();
            return false;
        }

        auto& registry = m_Scene->getRegistry();
        registry.clear();
        m_Scene->setSelectedEntity(entt::null);

        if (!sceneJson.contains("entities")) {
            std::cout << "Scene file is empty or invalid." << std::endl;
            return false;
        }

        for (const auto& entityJson : sceneJson["entities"]) {

            entt::entity entity = registry.create(entityJson["id"].get<entt::entity>());

            // --- Deserialize components ---
            if (entityJson.contains("transform")) {
                registry.emplace<Transform>(entity, entityJson["transform"].get<Transform>());
            }
            if (entityJson.contains("name")) {
                registry.emplace<Name>(entity, entityJson["name"].get<Name>());
            }

            // --- 7. Add block to deserialize Renderable ---
            if (entityJson.contains("renderable")) {
                // Get the saved data (color, tag)
                auto renderable = entityJson["renderable"].get<Renderable>();

                // Check the tag and re-assign the mesh pointer
                if (renderable.meshTag == "primitive_triangle") {
                    renderable.mesh = m_Renderer->getTriangleMesh();
                }
                // (else, mesh remains nullptr)

                // Emplace the component with the restored mesh
                registry.emplace<Renderable>(entity, renderable);
            }
        }

        std::cout << "Scene loaded from " << filepath << std::endl;
        return true;
    }

} // namespace Umgebung::scene