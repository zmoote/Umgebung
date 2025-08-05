#pragma once
#include "Molecule.hpp"
#include <vector>

namespace Umgebung {
    class Cell {
    protected:
        std::vector<Molecule*> molecules;
    public:
        Cell() = default;
        virtual ~Cell() {
            for (auto* m : molecules) delete m;
        }

        void addMolecule(Molecule* m) { molecules.push_back(m); }
        size_t getMoleculeCount() const { return molecules.size(); }
    };
}