#pragma once
#include "Organism.hpp"
#include <vector>

namespace Umgebung {
    class Ecosystem {
    protected:
        std::vector<Organism*> organisms;
    public:
        Ecosystem() = default;
        virtual ~Ecosystem() {
            for (auto* o : organisms) delete o;
        }

        void addOrganism(Organism* o) { organisms.push_back(o); }
        size_t getOrganismCount() const { return organisms.size(); }
    };
}