#include "umgebung/scene/SceneSerializer.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/components/Transform.hpp" // Assumes this is where TransformComponent is
#include "umgebung/util/JsonHelpers.hpp"         // Our helpers from Step 5.A

#include <entt/entt.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

namespace Umgebung::scene {

    // Helper: A macro to shorten the component name
    using TransformComponent = Umgebung::ecs::components::TransformComponent;

    SceneSerializer::SceneSerializer(Scene* scene)
        : m_Scene(scene) {
    }

    // --- NEW SERIALIZE FUNCTION ---
    void SceneSerializer::serialize(const std::string& filepath) {
        if (!m_Scene) return;

        nlohmann::json sceneJson;
        auto& registry = m_Scene->getRegistry();

        // We will create a JSON array called "entities"
        nlohmann::json entityList = nlohmann::json::array();

        // Iterate over all entities in the registry
        registry.view<entt::entity>().each([&](auto entity) {
            nlohmann::json entityJson;
            // Store the entity's ID
            entityJson["id"] = entity;

            // --- Serialize components ---

            // Check for and serialize TransformComponent
            if (registry.all_of<TransformComponent>(entity)) {
                // This will use our JsonHelpers.hpp automatically
                entityJson["transform"] = registry.get<TransformComponent>(entity);
            }

            // (Add other components like Soul, Consciousness here later)
            // if (registry.all_of<Soul>(entity)) { ... }

            // Add the entity's JSON to our list
            entityList.push_back(entityJson);
            });

        sceneJson["entities"] = entityList;

        // Write the JSON object to a file
        std::ofstream outFile(filepath);
        if (outFile.is_open()) {
            outFile << sceneJson.dump(4); // .dump(4) formats it nicely
            outFile.close();
            std::cout << "Scene saved to " << filepath << std::endl;
        }
        else {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        }
    }

    // --- NEW DESERIALIZE FUNCTION ---
    bool SceneSerializer::deserialize(const std::string& filepath) {
        if (!m_Scene) return false;

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
        // Clear the current scene before loading
        registry.clear();
        m_Scene->setSelectedEntity(entt::null);

        if (!sceneJson.contains("entities")) {
            std::cout << "Scene file is empty or invalid." << std::endl;
            return false; // Nothing to load
        }

        // Iterate over the JSON array of entities
        for (const auto& entityJson : sceneJson["entities"]) {

            // Create a new entity, using the *saved ID*
            // This is important for preserving relationships (which we'll add later)
            entt::entity entity = registry.create(entityJson["id"].get<entt::entity>());

            // --- Deserialize components ---

            // Check for and deserialize TransformComponent
            if (entityJson.contains("transform")) {
                // This will use our JsonHelpers.hpp automatically
                auto transform = entityJson["transform"].get<TransformComponent>();
                registry.emplace<TransformComponent>(entity, transform);
            }

            // (Add other components here later)
            // if (entityJson.contains("soul")) { ... }
        }

        std::cout << "Scene loaded from " << filepath << std::endl;
        return true;
    }

} // namespace Umgebung::scene