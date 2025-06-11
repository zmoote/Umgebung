// --- Core/LayerStack.cpp ---
#include "LayerStack.hpp"
#include <algorithm>

namespace Umgebung {

    void LayerStack::PushLayer(Layer* layer) {
        layers.emplace_back(layer);
        layer->OnAttach();
    }

    void LayerStack::PopLayer(Layer* layer) {
        auto it = std::find(layers.begin(), layers.end(), layer);
        if (it != layers.end()) {
            (*it)->OnDetach();
            layers.erase(it);
        }
    }

} // namespace Umgebung