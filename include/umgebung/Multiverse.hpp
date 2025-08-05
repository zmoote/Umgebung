#pragma once
#include "Universe.hpp"
#include <vector>

namespace Umgebung {
    class Multiverse {
    protected:
        std::vector<Universe*> universes;
    public:
        Multiverse() = default;
        virtual ~Multiverse() {
            for (auto* u : universes) delete u;
        }

        void addUniverse(Universe* u) { universes.push_back(u); }
        size_t getUniverseCount() const { return universes.size(); }
    };
}