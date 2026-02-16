#include "umgebung/ecs/systems/ObserverSystem.hpp"
#include "umgebung/util/LogMacros.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <nlohmann/json.hpp> 

namespace Umgebung::ecs::systems {

    ObserverSystem::ObserverSystem() = default;
    ObserverSystem::~ObserverSystem() = default;

    void ObserverSystem::init() {
        loadConfig();
    }

    void ObserverSystem::loadConfig() {
        std::ifstream file("assets/config/CameraLevels.json");
        if (!file.is_open()) {
            UMGEBUNG_LOG_ERROR("Failed to open assets/config/CameraLevels.json");
            return;
        }

        nlohmann::json j;
        file >> j;

        auto& levels = j["cameraLevels"];

        // Helper to load config
        auto load = [&](const std::string& key, components::ScaleType type) {
            if (levels.contains(key)) {
                config_[type] = {
                    levels[key]["nearPlane"],
                    levels[key]["farPlane"],
                    levels[key]["units"]
                };
            }
        };

        load("Planetary", components::ScaleType::Planetary);
        load("SolarSystem", components::ScaleType::SolarSystem);
        load("Galactic", components::ScaleType::Galactic);
        load("ExtraGalactic", components::ScaleType::ExtraGalactic);
        load("Universal", components::ScaleType::Universal);
        load("Multiversal", components::ScaleType::Multiversal);
        
        // Default fallback for Human/Micro if not in JSON (though we should add them or map them)
        // For now, let's assume Human maps to Planetary settings or similar, or add explicit defaults.
        // Actually, let's map Human to a default strict setting.
        config_[components::ScaleType::Human] = { 0.1f, 10000.0f, "meters" };
        config_[components::ScaleType::Micro] = { 0.001f, 100.0f, "micrometers" };
        config_[components::ScaleType::Quantum] = { 0.00001f, 10.0f, "nanometers" };
    }

    void ObserverSystem::onUpdate(renderer::Camera& camera, entt::entity selectedEntity, entt::registry* registry) {
        glm::vec3 pos = camera.getPosition();
        float distFromOrigin = glm::length(pos);

        components::ScaleType newScale = components::ScaleType::Human;

        // 1. Try to determine scale from selected entity first (Contextual Scaling)
        bool scaleFound = false;
        if (selectedEntity != entt::null && registry) {
            if (registry->all_of<components::ScaleComponent, components::Transform>(selectedEntity)) {
                const auto& transform = registry->get<components::Transform>(selectedEntity);
                const auto& scaleComp = registry->get<components::ScaleComponent>(selectedEntity);
                
                float distToEntity = glm::distance(pos, transform.position);
                
                // If we are "close" to the selected entity relative to its own scale, switch to its scale
                // threshold: 1000 units in that scale's world
                float threshold = 1000.0f; 
                // However, for Micro/Quantum, we need to be much more sensitive
                if (scaleComp.type == components::ScaleType::Micro) threshold = 1.0f;
                if (scaleComp.type == components::ScaleType::Quantum) threshold = 0.01f;

                if (distToEntity < threshold) {
                    newScale = scaleComp.type;
                    scaleFound = true;
                }
            }
        }

        // 2. Fallback to origin-based distance if no selection or too far from selection
        if (!scaleFound) {
            if (distFromOrigin < 100.0f) {
                newScale = components::ScaleType::Human;
            } else if (distFromOrigin < 100000.0f) {
                 newScale = components::ScaleType::Planetary;
            } else if (distFromOrigin < 100000000.0f) {
                 newScale = components::ScaleType::SolarSystem;
            } else if (distFromOrigin < 1000000000000.0f) {
                 newScale = components::ScaleType::Galactic;
            } else {
                 newScale = components::ScaleType::Universal;
            }
        }

        if (newScale != currentScale_ || firstUpdate_) {
            currentScale_ = newScale;
            UMGEBUNG_LOG_INFO("Observer Scale Changed to: {} (Context: {})", 
                config_.count(currentScale_) ? config_[currentScale_].units : "Unknown",
                scaleFound ? "Entity" : "Position");
            updateCameraSettings(camera);
            firstUpdate_ = false;
        }
    }

    void ObserverSystem::updateCameraSettings(renderer::Camera& camera) {
        if (config_.find(currentScale_) != config_.end()) {
            const auto& cfg = config_[currentScale_];
            
            camera.setPlanes(cfg.nearPlane, cfg.farPlane);
            UMGEBUNG_LOG_INFO("Updating Camera Planes: Near={}, Far={}", cfg.nearPlane, cfg.farPlane);
        }
    }

}
