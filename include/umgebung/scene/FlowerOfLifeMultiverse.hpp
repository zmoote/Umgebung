/**
 * @file FlowerOfLifeMultiverse.hpp
 * @brief Elena Danaan's Flower of Life Multiverse geometric lattice container
 *        linking intersecting `.umgebung` Bubble Universes bridged by the Void.
 *
 * @details
 * Implements the cosmological topology specific to **Elena Danaan's framework**
 * and **`Unified::Synthesis`** (see `GEMINI.md` Section 3 &
 * `submodules/research/Personal/### ## Audio Script Project Umgebun.txt`):
 *   - The Multiverse ("Body of Source") is structured geometrically as the
 *     **Flower of Life** hexagonal lattice of interlocking spheres:
 *       - Ring 0 (Center Seed): 1 sphere at `(0, 0, 0)`
 *       - Ring 1 (Seed of Life): 6 spheres at distance `R` spaced by `60°`
 *         forming exact Vesica Piscis overlaps (proven equilateral in
 *         `formal/Umgebung/Unified/FlowerOfLife.lean`)
 *       - Ring 2 (Outer Flower of Life): 12 spheres completing the classic
 *         19-sphere Flower of Life motif.
 *   - Each occupied sphere node references an individual `.umgebung` **Bubble
 *     Universe** file on disk, while **the Void** acts as the non-linear
 *     medium separating and bridging parallel Bubble Universes.
 *
 * @note
 * Standalone computational physics **Scenarios** (`*.scenario.umgebung`) do NOT
 * use this wrapper; they are displayed cleanly as `Scenario -> Entities`.
 */

#pragma once

#include "umgebung/scene/UniverseScene.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace umgebung::scene {

    /**
     * @struct VoidBridgeState
     * @brief Models the non-linear Void separating and bridging Bubble Universes
     *        within Elena Danaan's Flower of Life Multiverse.
     */
    struct VoidBridgeState {
        /** @brief Dimensionless non-linear resonance/conductivity bridging parallel bubbles (`[0, 1]`). */
        double bridgeConductivity{1.0};

        /** @brief Ambient Love binding frequency permeating the Multiverse lattice. */
        double loveBindingFrequency{528.0};

        /** @brief Ambient Phryll source potential radiating from the Void into bubble vortices. */
        double sourcePhryllPotential{4096.0};
    };

    /**
     * @struct BubbleUniverseNode
     * @brief Single spherical node on the Flower of Life lattice corresponding
     *        to a parallel Bubble Universe (backed by its own `.umgebung` file).
     */
    struct BubbleUniverseNode {
        /** @brief Unique node identifier within the Flower of Life manifest (e.g., `"seed_0"`, `"ring1_0"`). */
        std::string nodeId;

        /** @brief Human-readable Bubble Universe label. */
        std::string label;

        /** @brief Concentric Flower of Life ring index (`0` = center, `1` = 6-node Seed, `2` = 12-node outer Flower). */
        std::uint32_t ringIndex{0U};

        /** @brief Angular index within `ringIndex`. */
        std::uint32_t sectorIndex{0U};

        /** @brief Geometric center coordinate on the Flower of Life lattice (in bubble-radius units). */
        Vec3d latticePosition{};

        /** @brief Geometric sphere radius $R_{\text{bubble}}$ (adjacent Ring-1 nodes intersect at distance $R_{\text{bubble}}$). */
        double bubbleRadius{1.0};

        /** @brief Relative or absolute path to the `.umgebung` Bubble Universe scene file (`""` = unpopulated node). */
        std::filesystem::path universeFilePath;

        /** @brief Loaded `UniverseScene` when `universeFilePath` resolved cleanly on disk. */
        std::optional<UniverseScene> loadedUniverse{std::nullopt};
    };

    /**
     * @class FlowerOfLifeMultiverse
     * @brief Loads, generates, validates, and indexes a Flower of Life
     *        `.multiverse.umgebung` manifest and its linked `.umgebung` Bubble Universes.
     */
    class FlowerOfLifeMultiverse final {
    public:
        static constexpr std::uint32_t kCurrentSchemaVersion = 1U;

        FlowerOfLifeMultiverse() = default;

        /**
         * @brief Generates the exact geometric node coordinates for a Flower of
         *        Life hexagonal lattice of `maxRing` concentric rings:
         *          - `maxRing = 0`: 1 center sphere
         *          - `maxRing = 1`: 7 spheres (Seed of Life: 1 center + 6 surrounding at distance `bubbleRadius`)
         *          - `maxRing = 2`: 19 spheres (classic Flower of Life: 1 + 6 + 12)
         *
         * @param bubbleRadius Characteristic sphere radius $R > 0$.
         * @param maxRing      Number of concentric rings (`0`, `1`, or `2`).
         * @return Ordered list of `BubbleUniverseNode` descriptors positioned on the lattice.
         */
        [[nodiscard]] static std::vector<BubbleUniverseNode> GenerateCanonicalLattice(
            double bubbleRadius = 1.0,
            std::uint32_t maxRing = 1U
        );

        /**
         * @brief Loads a `.multiverse.umgebung` JSON manifest from disk and
         *        resolves each linked `.umgebung` Bubble Universe file relative
         *        to the manifest's parent directory.
         */
        [[nodiscard]] bool LoadFromFile(
            const std::filesystem::path& manifestPath,
            std::string& outError
        );

        /** @brief Serializes this Multiverse manifest (without embedding full universe files) to JSON. */
        [[nodiscard]] nlohmann::json ToJson() const;

        /** @brief Validates structural invariants of the Flower of Life lattice. */
        [[nodiscard]] bool Validate(std::string& outError) const;

        /** @brief Resets all loaded nodes and Void state. */
        void Clear() noexcept;

        [[nodiscard]] const std::string& GetMultiverseId() const noexcept { return m_multiverseId; }
        [[nodiscard]] const std::string& GetName() const noexcept { return m_name; }
        [[nodiscard]] const std::string& GetDescription() const noexcept { return m_description; }
        [[nodiscard]] const std::filesystem::path& GetSourcePath() const noexcept { return m_sourcePath; }
        [[nodiscard]] double GetBubbleRadius() const noexcept { return m_bubbleRadius; }
        [[nodiscard]] const VoidBridgeState& GetVoidState() const noexcept { return m_voidState; }
        [[nodiscard]] const std::vector<BubbleUniverseNode>& GetNodes() const noexcept { return m_nodes; }
        [[nodiscard]] std::size_t GetActiveNodeIndex() const noexcept { return m_activeNodeIndex; }
        [[nodiscard]] bool IsEmpty() const noexcept { return m_nodes.empty(); }

        [[nodiscard]] bool SetActiveNodeIndex(std::size_t index) noexcept;
        [[nodiscard]] const BubbleUniverseNode* GetActiveNode() const noexcept;

    private:
        std::uint32_t m_schemaVersion{kCurrentSchemaVersion};
        std::string m_multiverseId{"flower_of_life_multiverse"};
        std::string m_name{"Flower of Life Multiverse (Source)"};
        std::string m_description;
        std::filesystem::path m_sourcePath;
        double m_bubbleRadius{1.0};
        VoidBridgeState m_voidState{};
        std::vector<BubbleUniverseNode> m_nodes;
        std::size_t m_activeNodeIndex{0U};
    };

} // namespace umgebung::scene
