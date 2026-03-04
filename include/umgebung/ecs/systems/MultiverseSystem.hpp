#pragma once

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

namespace Umgebung::scene { class Scene; }

namespace Umgebung::ecs::systems {

    /**
     * @brief System responsible for generating and managing the Multiversal lattice structure.
     * Uses 3D Hexagonal Close Packing (Flower of Life) for universe placement.
     */
    class MultiverseSystem {
    public:
        MultiverseSystem();
        ~MultiverseSystem();

        /**
         * @brief Generates a cluster of universes around a central point.
         * @param scene The scene to add entities to.
         * @param center The center of the cluster.
         * @param radius Number of layers in the lattice.
         * @param universeSpacing The distance between universe centers.
         */
        void generateLattice(scene::Scene& scene, const glm::vec3& center, int layers, float universeSpacing);

    private:
        // Helper to get HCP lattice positions
        std::vector<glm::vec3> calculateHCPPositions(int layers, float spacing);
    };

} // namespace Umgebung::ecs::systems