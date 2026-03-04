#include "umgebung/ecs/systems/MultiverseSystem.hpp"
#include "umgebung/scene/Scene.hpp"
#include "umgebung/ecs/components/Transform.hpp"
#include "umgebung/ecs/components/ScaleComponent.hpp"
#include "umgebung/ecs/components/Name.hpp"
#include "umgebung/ecs/components/Renderable.hpp"
#include "umgebung/ecs/components/TimeComponent.hpp"
#include "umgebung/util/LogMacros.hpp"

#include <glm/glm.hpp>
#include <vector>
#include <cmath>

namespace Umgebung::ecs::systems {

    MultiverseSystem::MultiverseSystem() = default;
    MultiverseSystem::~MultiverseSystem() = default;

    void MultiverseSystem::generateLattice(scene::Scene& scene, const glm::vec3& center, int layers, float universeSpacing) {
        UMGEBUNG_LOG_INFO("Generating Multiverse Lattice (Flower of Life interconnected)...");

        auto positions = calculateHCPPositions(layers, universeSpacing);

        for (size_t i = 0; i < positions.size(); ++i) {
            auto entity = scene.createEntity();
            scene.getRegistry().replace<components::Name>(entity, "Universe " + std::to_string(i));

            auto& transform = scene.getRegistry().get<components::Transform>(entity);
            transform.position = center + positions[i];

            // "Interconnected" means center-to-center distance equals radius
            // This creates the overlapping "Flower of Life" geometry in 3D
            // Assuming unit sphere mesh (radius 0.5), scale must be 2.0 * spacing
            transform.scale = glm::vec3(universeSpacing * 2.0f);

            scene.getRegistry().emplace<components::ScaleComponent>(entity, components::ScaleComponent{ components::ScaleType::Universal });

            // Randomize density for each universe as a simulation parameter
            float densityValue = 3.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX/9.0f));
            auto& timeComp = scene.getRegistry().emplace<components::TimeComponent>(entity);
            timeComp.density = densityValue;

            // Add Renderable and assign a sphere mesh (bubbles)
            auto& renderable = scene.getRegistry().emplace<components::Renderable>(entity);
            renderable.meshTag = "assets/models/Sphere.glb"; 

            // Give it a subtle color based on its density
            renderable.color = glm::vec4(0.4f, 0.6f, 1.0f, 0.2f); // Lower alpha for overlapping
            if (densityValue > 5.0f) renderable.color = glm::vec4(0.8f, 0.4f, 1.0f, 0.3f);
        }

        UMGEBUNG_LOG_INFO("Generated {} universes in the lattice.", positions.size());
    }

    std::vector<glm::vec3> MultiverseSystem::calculateHCPPositions(int layers, float spacing) {
        std::vector<glm::vec3> positions;
        
        // HCP lattice constants
        float dx = spacing;
        float dy = spacing * sqrt(3.0f) / 2.0f;
        float dz = spacing * sqrt(6.0f) / 3.0f;

        for (int z = -layers; z <= layers; ++z) {
            for (int y = -layers; y <= layers; ++y) {
                for (int x = -layers; x <= layers; ++x) {
                    
                    // Fix modulo for negative numbers to maintain symmetry
                    int xOffsetIdx = (std::abs(y) + std::abs(z)) % 2;
                    int yOffsetIdx = std::abs(z) % 2;

                    float posX = x * dx + (xOffsetIdx * (dx * 0.5f));
                    float posY = y * dy + (yOffsetIdx * (dy * (1.0f / 3.0f)));
                    float posZ = z * dz;

                    glm::vec3 pos(posX, posY, posZ);
                    
                    // Spherical mask with a slight buffer to include all requested layers
                    if (glm::length(pos) <= (layers + 0.1f) * spacing) {
                        positions.push_back(pos);
                    }
                }
            }
        }
        return positions;
    }

} // namespace Umgebung::ecs::systems