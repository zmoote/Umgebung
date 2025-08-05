#pragma once
#include "Cell.hpp"
#include <vector>

namespace Umgebung {
    class Organism {
    protected:
        std::vector<Cell*> cells;
    public:
        Organism() = default;
        virtual ~Organism() {
            for (auto* c : cells) delete c;
        }

        void addCell(Cell* c) { cells.push_back(c); }
        size_t getCellCount() const { return cells.size(); }
    };
}