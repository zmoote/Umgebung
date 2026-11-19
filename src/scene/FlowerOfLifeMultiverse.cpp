/**
 * @file FlowerOfLifeMultiverse.cpp
 * @brief Implementation of Elena Danaan's Flower of Life geometric Multiverse
 *        lattice generator and `.multiverse.umgebung` manifest loader.
 */

#include "umgebung/scene/FlowerOfLifeMultiverse.hpp"

#include <cmath>
#include <fstream>
#include <numbers>

#include <spdlog/spdlog.h>

namespace umgebung::scene {

    std::vector<BubbleUniverseNode> FlowerOfLifeMultiverse::GenerateCanonicalLattice(
        double bubbleRadius,
        std::uint32_t maxRing
    ) {
        std::vector<BubbleUniverseNode> nodes;
        const double R = (bubbleRadius > 0.0) ? bubbleRadius : 1.0;

        // Ring 0: Central Seed Sphere at the origin `(0, 0, 0)`.
        BubbleUniverseNode center{};
        center.nodeId = "seed_center";
        center.label = "Center Seed Bubble Universe";
        center.ringIndex = 0U;
        center.sectorIndex = 0U;
        center.latticePosition = {0.0, 0.0, 0.0};
        center.bubbleRadius = R;
        nodes.push_back(std::move(center));

        if (maxRing == 0U) {
            return nodes;
        }

        // Ring 1 (Seed of Life): 6 spheres placed at distance R every 60 degrees.
        // Every Ring-1 sphere passes through the origin AND through the centers
        // of both its adjacent Ring-1 neighbors (exact equilateral Vesica Piscis,
        // formally proven in `formal/Umgebung/Unified/FlowerOfLife.lean`).
        for (std::uint32_t k = 0U; k < 6U; ++k) {
            const double angle = (static_cast<double>(k) * std::numbers::pi) / 3.0;
            BubbleUniverseNode node{};
            node.nodeId = "ring1_" + std::to_string(k);
            node.label = "Seed Ring-1 Bubble #" + std::to_string(k + 1U);
            node.ringIndex = 1U;
            node.sectorIndex = k;
            node.latticePosition = {R * std::cos(angle), R * std::sin(angle), 0.0};
            node.bubbleRadius = R;
            nodes.push_back(std::move(node));
        }

        if (maxRing < 2U) {
            return nodes;
        }

        // Ring 2 (Complete 19-Sphere Flower of Life): 12 spheres arranged as
        // 6 outer vertices at distance 2R plus 6 midpoints at distance sqrt(3)*R.
        for (std::uint32_t k = 0U; k < 6U; ++k) {
            const double a0 = (static_cast<double>(k) * std::numbers::pi) / 3.0;
            const double a1 = (static_cast<double>(k + 1U) * std::numbers::pi) / 3.0;

            // Vertex node at 2*R along ray `a0`.
            BubbleUniverseNode vertexNode{};
            vertexNode.nodeId = "ring2_" + std::to_string(2U * k);
            vertexNode.label = "Flower Ring-2 Bubble #" + std::to_string(2U * k + 1U);
            vertexNode.ringIndex = 2U;
            vertexNode.sectorIndex = 2U * k;
            vertexNode.latticePosition = {2.0 * R * std::cos(a0), 2.0 * R * std::sin(a0), 0.0};
            vertexNode.bubbleRadius = R;
            nodes.push_back(std::move(vertexNode));

            // Midpoint node at vector sum of adjacent Ring-1 centers (distance sqrt(3)*R).
            BubbleUniverseNode midNode{};
            midNode.nodeId = "ring2_" + std::to_string(2U * k + 1U);
            midNode.label = "Flower Ring-2 Bubble #" + std::to_string(2U * k + 2U);
            midNode.ringIndex = 2U;
            midNode.sectorIndex = 2U * k + 1U;
            midNode.latticePosition = {
                R * (std::cos(a0) + std::cos(a1)),
                R * (std::sin(a0) + std::sin(a1)),
                0.0
            };
            midNode.bubbleRadius = R;
            nodes.push_back(std::move(midNode));
        }

        return nodes;
    }

    bool FlowerOfLifeMultiverse::Validate(std::string& outError) const {
        if (m_schemaVersion != kCurrentSchemaVersion) {
            outError = "Unsupported FlowerOfLifeMultiverse schema_version.";
            return false;
        }
        if (m_multiverseId.empty() || m_name.empty()) {
            outError = "FlowerOfLifeMultiverse requires non-empty 'multiverse_id' and 'name'.";
            return false;
        }
        if (!std::isfinite(m_bubbleRadius) || m_bubbleRadius <= 0.0) {
            outError = "FlowerOfLifeMultiverse requires finite bubble_radius > 0.";
            return false;
        }
        if (m_nodes.empty()) {
            outError = "FlowerOfLifeMultiverse must contain at least one lattice node.";
            return false;
        }
        return true;
    }

    bool FlowerOfLifeMultiverse::LoadFromFile(
        const std::filesystem::path& manifestPath,
        std::string& outError
    ) {
        std::ifstream file(manifestPath);
        if (!file.is_open()) {
            outError = "Failed to open .multiverse.umgebung manifest: " + manifestPath.string();
            return false;
        }

        nlohmann::json root;
        try {
            file >> root;
        } catch (const nlohmann::json::exception& ex) {
            outError = "Malformed JSON in '" + manifestPath.string() + "': " + ex.what();
            return false;
        }

        if (!root.is_object()) {
            outError = "Root of .multiverse.umgebung manifest must be a JSON object.";
            return false;
        }

        try {
            FlowerOfLifeMultiverse parsed{};
            parsed.m_schemaVersion = root.value("schema_version", 0U);
            parsed.m_multiverseId = root.value("multiverse_id", std::string{});
            parsed.m_name = root.value("name", std::string{});
            parsed.m_description = root.value("description", std::string{});
            parsed.m_sourcePath = manifestPath;
            parsed.m_bubbleRadius = root.value("bubble_radius", 1.0);

            const std::uint32_t rings = root.value("flower_of_life_rings", 1U);
            parsed.m_nodes = GenerateCanonicalLattice(parsed.m_bubbleRadius, rings);

            if (root.contains("the_void") && root["the_void"].is_object()) {
                const auto& v = root["the_void"];
                parsed.m_voidState.bridgeConductivity = v.value("bridge_conductivity", 1.0);
                parsed.m_voidState.loveBindingFrequency = v.value("love_binding_frequency", 528.0);
                parsed.m_voidState.sourcePhryllPotential = v.value("source_phryll_potential", 4096.0);
            }

            // Map any universe files explicitly assigned to lattice nodes.
            const std::filesystem::path baseDir = manifestPath.parent_path();
            if (root.contains("bubble_universes") && root["bubble_universes"].is_array()) {
                for (const auto& entry : root["bubble_universes"]) {
                    if (!entry.is_object()) {
                        continue;
                    }
                    const std::string targetNodeId = entry.value("node_id", std::string{});
                    const std::string relPath = entry.value("universe_file", std::string{});
                    const std::string customLabel = entry.value("label", std::string{});

                    for (auto& node : parsed.m_nodes) {
                        if (node.nodeId == targetNodeId) {
                            if (!customLabel.empty()) {
                                node.label = customLabel;
                            }
                            if (!relPath.empty()) {
                                std::filesystem::path resolved = std::filesystem::path(relPath);
                                if (resolved.is_relative()) {
                                    resolved = (baseDir / resolved).lexically_normal();
                                }
                                node.universeFilePath = resolved;

                                UniverseScene loaded{};
                                std::string uErr;
                                if (UniverseScene::LoadFromFile(resolved, loaded, uErr)) {
                                    loaded.scope = SimulationScope::BubbleUniverse;
                                    node.loadedUniverse = std::move(loaded);
                                } else {
                                    spdlog::warn(
                                        "FlowerOfLifeMultiverse: Could not load linked Bubble Universe '{}' for node '{}': {}",
                                        resolved.string(),
                                        node.nodeId,
                                        uErr
                                    );
                                }
                            }
                            break;
                        }
                    }
                }
            }

            if (!parsed.Validate(outError)) {
                return false;
            }

            // Default active node to the first node that has a loaded Bubble Universe.
            parsed.m_activeNodeIndex = 0U;
            for (std::size_t i = 0U; i < parsed.m_nodes.size(); ++i) {
                if (parsed.m_nodes[i].loadedUniverse.has_value()) {
                    parsed.m_activeNodeIndex = i;
                    break;
                }
            }

            *this = std::move(parsed);
            spdlog::info(
                "FlowerOfLifeMultiverse: Loaded '{}' ({} Flower-of-Life nodes, bubbleRadius={:.2f}) from '{}'.",
                m_name,
                m_nodes.size(),
                m_bubbleRadius,
                manifestPath.string()
            );
            return true;
        } catch (const nlohmann::json::exception& ex) {
            outError = std::string("JSON parsing error in Multiverse manifest: ") + ex.what();
            return false;
        }
    }

    nlohmann::json FlowerOfLifeMultiverse::ToJson() const {
        nlohmann::json bubblesJson = nlohmann::json::array();
        for (const auto& node : m_nodes) {
            if (!node.universeFilePath.empty()) {
                bubblesJson.push_back({
                    {"node_id", node.nodeId},
                    {"label", node.label},
                    {"ring_index", node.ringIndex},
                    {"lattice_position", {node.latticePosition.x, node.latticePosition.y, node.latticePosition.z}},
                    {"universe_file", node.universeFilePath.string()}
                });
            }
        }

        return nlohmann::json{
            {"schema_version", m_schemaVersion},
            {"scope", SimulationScopeToString(SimulationScope::FlowerOfLifeMultiverse)},
            {"multiverse_id", m_multiverseId},
            {"name", m_name},
            {"description", m_description},
            {"bubble_radius", m_bubbleRadius},
            {"the_void", {
                {"bridge_conductivity", m_voidState.bridgeConductivity},
                {"love_binding_frequency", m_voidState.loveBindingFrequency},
                {"source_phryll_potential", m_voidState.sourcePhryllPotential}
            }},
            {"bubble_universes", std::move(bubblesJson)}
        };
    }

    void FlowerOfLifeMultiverse::Clear() noexcept {
        m_nodes.clear();
        m_activeNodeIndex = 0U;
        m_sourcePath.clear();
    }

    bool FlowerOfLifeMultiverse::SetActiveNodeIndex(std::size_t index) noexcept {
        if (index >= m_nodes.size()) {
            return false;
        }
        m_activeNodeIndex = index;
        return true;
    }

    const BubbleUniverseNode* FlowerOfLifeMultiverse::GetActiveNode() const noexcept {
        if (m_nodes.empty() || m_activeNodeIndex >= m_nodes.size()) {
            return nullptr;
        }
        return &m_nodes[m_activeNodeIndex];
    }

} // namespace umgebung::scene
