#ifndef UMGEBUNG_ENGINE_HPP
#define UMGEBUNG_ENGINE_HPP

#include <cmath>
#include <vector>

namespace umgebung {

    namespace constants {
        // Planck length in meters
        constexpr double PLANCK_LENGTH = 1.616255e-35;
        constexpr double PI = 3.14159265358979323846;
    }

    struct Point3D {
        double x, y, z;
    };

    class PSU {
    public:
        PSU() = default;
        PSU(double x, double y, double z) : center{x, y, z} {}

        // The fundamental PSU radius r = ℓ / 2
        static constexpr double radius() {
            return constants::PLANCK_LENGTH / 2.0;
        }

        static constexpr double volume() {
            double r = radius();
            return (4.0 / 3.0) * constants::PI * r * r * r;
        }

        Point3D getCenter() const { return center; }

    private:
        Point3D center{0.0, 0.0, 0.0};
    };

    class FlowerOfLife {
    public:
        FlowerOfLife() = default;

        // Generate a basic Flower of Life arrangement of PSUs
        // level specifies the number of layers/rings
        void generate(int levels);

        const std::vector<PSU>& getUnits() const { return units; }

    private:
        std::vector<PSU> units;
    };

} // namespace umgebung

#endif // UMGEBUNG_ENGINE_HPP
