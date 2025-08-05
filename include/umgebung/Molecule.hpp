#pragma once
#include "Atom.hpp"
#include <vector>

namespace Umgebung {
    class Molecule {
    protected:
        std::vector<Atom*> atoms;
    public:
        Molecule() = default;
        virtual ~Molecule() {
            for (auto* a : atoms) delete a;
        }

        void addAtom(Atom* a) { atoms.push_back(a); }
        size_t getAtomCount() const { return atoms.size(); }
    };
}