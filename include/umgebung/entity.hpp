#pragma once

#include <memory>
#include <vector>
#include <string>
#include "umgebung/types.hpp"

namespace umgebung {

    // Abstract Base Class for any simulated entity
    class IEntity {
    public:
        virtual ~IEntity() = default;

        virtual void update(const Observer& observer) = 0;
        virtual void load() = 0;
        virtual void unload() = 0;
        
        virtual ScaleLevel getScale() const = 0;
        virtual const std::string& getName() const = 0;
    };

    // Deep Object-Oriented Framework Base
    class EntityBase : public IEntity {
    protected:
        std::string name;
        Coordinate position;
        VibrationalState vib_state;
        ConsciousnessState consciousness;
        
        bool is_loaded{false};
        std::vector<std::shared_ptr<IEntity>> children;

        // Threshold distance for an observer to trigger loading of children
        double observation_threshold;

    public:
        EntityBase(std::string n, Coordinate pos, double threshold,
                   VibrationalState v = VibrationalState(), 
                   ConsciousnessState c = ConsciousnessState())
            : name(std::move(n)), position(pos), observation_threshold(threshold),
              vib_state(v), consciousness(c) {}

        const std::string& getName() const override { return name; }
        
        // Procedural generation trigger
        void update(const Observer& observer) override {
            bool in_range = observer.isObserving(position, getScale(), observation_threshold);
            
            if (in_range && !is_loaded) {
                load();
            } else if (!in_range && is_loaded) {
                unload();
            }

            // Cascade update to loaded children
            if (is_loaded) {
                for (auto& child : children) {
                    child->update(observer);
                }
            }
        }

        void unload() override {
            // Free memory for regions no longer observed
            children.clear();
            is_loaded = false;
        }

        // load() remains pure virtual so subclasses define HOW they procedurally generate children
        virtual void load() = 0; 
    };

} // namespace umgebung
