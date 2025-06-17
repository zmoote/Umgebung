#pragma once
#include "Layer.hpp"
#include <vector>

namespace Umgebung {

    class LayerStack {
    public:
        LayerStack() = default;
        ~LayerStack();

        void PushLayer(Layer* layer);
        void PopLayer(Layer* layer);

        bool empty() const { return layers.empty(); }
        Layer* back() const { return layers.back(); }

        // Iterator support for range-based for loop
        auto begin() { return layers.begin(); }
        auto end() { return layers.end(); }

    private:
        std::vector<Layer*> layers;
    };

} // namespace Umgebung