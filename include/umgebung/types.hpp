#pragma once

#include <cstdint>
#include <string>

namespace umgebung {

    // Defines the scope or zoom-level of the simulation
    enum class ScaleLevel {
        Planck = 0,     // 10^-35 m (PSUs, Flower of Life)
        Quantum,        // 10^-15 m (Subatomic particles)
        Classical,      // 10^0 m (Human scale)
        Planet,         // 10^7 m (Planets)
        StarSystem,     // 10^13 m (Solar systems)
        Galaxy,         // 10^21 m (Galaxies)
        Universe,       // 10^26 m (Observable Universes)
        Multiverse      // 10^X m (Collections of Universes)
    };

    // Core attributes modeled beyond standard physics
    struct VibrationalState {
        double density;    // Dimensional density (e.g., 3.0 for 3D, 5.0 for 5D)
        double frequency;  // Base energetic frequency signature (Hz or abstract unit)
        
        VibrationalState(double d = 3.0, double f = 0.0) : density(d), frequency(f) {}
    };

    struct ConsciousnessState {
        bool is_conscious;
        double awareness_index; // 0.0 (inanimate) to 1.0+ (highly evolved)
        
        ConsciousnessState(bool c = false, double a = 0.0) : is_conscious(c), awareness_index(a) {}
    };

    // 3D/ND coordinate system capable of scaling
    struct Coordinate {
        double x, y, z;
        ScaleLevel scale;
    };

    // The Observer dictates what part of the simulation is instantiated
    class Observer {
    public:
        Coordinate position;
        ScaleLevel focus_scale; // What scale the observer is currently resolving

        Observer(Coordinate pos, ScaleLevel focus) : position(pos), focus_scale(focus) {}
        
        bool isObserving(const Coordinate& target_pos, ScaleLevel target_scale, double threshold_distance) const {
            // Simplified check: If observer is at a similar or smaller scale, and within distance
            if (focus_scale > target_scale) return false; // Too zoomed out to see this detail
            
            // In a real scenario, convert positions to a common scale to measure distance.
            // For now, we assume simple bounding box or distance triggers.
            double dx = position.x - target_pos.x;
            double dy = position.y - target_pos.y;
            double dz = position.z - target_pos.z;
            double dist_sq = dx*dx + dy*dy + dz*dz;
            
            return dist_sq <= (threshold_distance * threshold_distance);
        }
    };

} // namespace umgebung
