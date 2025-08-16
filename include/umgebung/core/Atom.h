#pragma once
#include "PhysicalObject.h"
#include <vector>

namespace Umgebung {
    class Atom : public PhysicalObject {
    public:
        Atom(const std::vector<float3>& quarkPositions);
        void update(float dt) override;
        void render() const override;

    private:
        std::vector<float3> quarks_;
        // device buffers (see CUDA section)
    };
}