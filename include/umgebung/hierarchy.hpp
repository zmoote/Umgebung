#pragma once

#include "umgebung/entity.hpp"
#include <iostream>

namespace umgebung {

    class Multiverse : public EntityBase {
    public:
        Multiverse(std::string n) 
            : EntityBase(std::move(n), {0.0, 0.0, 0.0, ScaleLevel::Multiverse}, 1e100) {}

        ScaleLevel getScale() const override { return ScaleLevel::Multiverse; }

        void load() override;
    };

    class Universe : public EntityBase {
    public:
        Universe(std::string n, Coordinate pos) 
            : EntityBase(std::move(n), pos, 1e26) {}

        ScaleLevel getScale() const override { return ScaleLevel::Universe; }

        void load() override;
    };

    class Galaxy : public EntityBase {
    public:
        Galaxy(std::string n, Coordinate pos) 
            : EntityBase(std::move(n), pos, 1e21) {}

        ScaleLevel getScale() const override { return ScaleLevel::Galaxy; }

        void load() override;
    };

    class StarSystem : public EntityBase {
    public:
        StarSystem(std::string n, Coordinate pos) 
            : EntityBase(std::move(n), pos, 1e13) {}

        ScaleLevel getScale() const override { return ScaleLevel::StarSystem; }

        void load() override;
    };

    class Planet : public EntityBase {
    public:
        Planet(std::string n, Coordinate pos, VibrationalState v = VibrationalState(3.0)) 
            : EntityBase(std::move(n), pos, 1e7, v) {}

        ScaleLevel getScale() const override { return ScaleLevel::Planet; }

        void load() override;
    };

    // A placeholder for reaching the absolute minimum scale, bridging into the existing CUDA engine
    class QuantumRegion : public EntityBase {
    public:
        QuantumRegion(std::string n, Coordinate pos)
            : EntityBase(std::move(n), pos, 1e-15) {}
            
        ScaleLevel getScale() const override { return ScaleLevel::Quantum; }
        
        void load() override;
    };

} // namespace umgebung
