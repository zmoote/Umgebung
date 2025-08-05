#pragma once

namespace Umgebung {
    constexpr double elementary_charge = 1.602176634e-19; // Coloumbs  (CODATA 2018)
    constexpr double MeV_c2_to_kg = 1.78266192e-30;      // kg per MeV/c²
    inline double MeVToKg(double mev) { return mev * MeV_c2_to_kg; }
}