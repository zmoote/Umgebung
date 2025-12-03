/**
 * @file SceneSerializer.cpp
 * @brief Implements the SceneSerializer class.
 */
#include "umgebung/scene/SceneSerializer.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/renderer/Renderer.hpp"
#include "umgebung/asset/ModelLoader.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/Name.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/RigidBody.hpp"
#include "umgebung/ecs/components/Collider.hpp"
#include "umgebung/ecs/components/ScaleComponent.hpp"
#include "umgebung/util/JsonHelpers.hpp"
#include "umgebung/util/LogMacros.hpp"

#include <entt/entt.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>

namespace Umgebung::scene {

    using Transform = Umgebung::ecs::components::Transform;
    using Name = Umgebung::ecs::components::Name;
    using Renderable = Umgebung::ecs::components::Renderable;
    using RigidBody = Umgebung::ecs::components::RigidBody;
    using Collider = Umgebung::ecs::components::Collider;
    using ScaleComponent = Umgebung::ecs::components::ScaleComponent;

    SceneSerializer::SceneSerializer(Scene* scene, renderer::Renderer* renderer)
        : m_Scene(scene), m_Renderer(renderer) {
    }

    void SceneSerializer::serialize(const std::filesystem::path& filepath) {
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
            if (registry.all_of<RigidBody>(entity)) {
                entityJson["rigidbody"] = registry.get<RigidBody>(entity);
            }
            if (registry.all_of<Collider>(entity)) {
                entityJson["collider"] = registry.get<Collider>(entity);
            }
            if (registry.all_of<ScaleComponent>(entity)) {
                entityJson["scale"] = registry.get<ScaleComponent>(entity);
            }

            entityList.push_back(entityJson);
            });

        sceneJson["entities"] = entityList;

        std::ofstream outFile(filepath);
        if (outFile.is_open()) {
            outFile << sceneJson.dump(4);
            outFile.close();
            UMGEBUNG_LOG_INFO("Scene saved to {}", filepath.string());
        }
        else {
            UMGEBUNG_LOG_ERROR("Could not open file for writing: {}", filepath.string());
        }
    }

    bool SceneSerializer::deserialize(const std::filesystem::path& filepath) {
        if (!m_Scene || !m_Renderer) {
            UMGEBUNG_LOG_ERROR("Serializer not initialized.");
            return false;
        }

        std::ifstream inFile(filepath);
        if (!inFile.is_open()) {
            UMGEBUNG_LOG_ERROR("Could not open file for reading: {}", filepath.string());
            return false;
        }
        nlohmann::json sceneJson;
        try {
            sceneJson = nlohmann::json::parse(inFile);
            inFile.close();
        }
        catch (nlohmann::json::parse_error& e) {
            UMGEBUNG_LOG_ERROR("Failed to parse scene file {}: {}", filepath.string(), e.what());
            inFile.close();
            return false;
        }

        auto& registry = m_Scene->getRegistry();
        registry.clear();
        m_Scene->setSelectedEntity(entt::null);

        if (!sceneJson.contains("entities")) {
            UMGEBUNG_LOG_WARN("Scene file is empty or invalid: {}", filepath.string());
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
            if (entityJson.contains("rigidbody")) {
                registry.emplace<RigidBody>(entity, entityJson["rigidbody"].get<RigidBody>());
            }
            if (entityJson.contains("collider")) {
                registry.emplace<Collider>(entity, entityJson["collider"].get<Collider>());
            }
            if (entityJson.contains("scale")) {
                registry.emplace<ScaleComponent>(entity, entityJson["scale"].get<ScaleComponent>());
            }

            if (entityJson.contains("renderable")) {
                auto renderable = entityJson["renderable"].get<Renderable>();

                // Check the tag and re-assign the mesh pointer
                if (renderable.meshTag == "primitive_triangle") {
                    renderable.mesh = m_Renderer->getTriangleMesh();
                }
                else if (!renderable.meshTag.empty()) {
                    // If the tag is not empty and not a primitive,
                    // assume it's a file path and try to load it.
                    renderable.mesh = m_Renderer->getModelLoader()->loadMesh(renderable.meshTag);

                    if (!renderable.mesh) {
                        // Log a warning if loading failed
                        UMGEBUNG_LOG_WARN("Failed to load mesh: {}", renderable.meshTag);
                    }
                }

                // Emplace the component (with either a valid mesh or nullptr)
                registry.emplace<Renderable>(entity, renderable);
            }
        }

        UMGEBUNG_LOG_INFO("Scene loaded from {}", filepath.string());
        return true;
    }

} // namespace Umgebung::scene
