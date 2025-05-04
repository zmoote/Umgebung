#pragma once
#include <vector>
#include "Layer.hpp"

namespace Umgebung {

    class LayerStack {
    public:
        void PushLayer(Layer* layer);
        void PopLayer(Layer* layer);

        std::vector<Layer*>::iterator begin() { return layers.begin(); }
        std::vector<Layer*>::iterator end() { return layers.end(); }

    private:
        std::vector<Layer*> layers;
    };

} // namespace Umgebung