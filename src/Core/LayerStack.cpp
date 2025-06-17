#include "LayerStack.hpp"
#include <algorithm>

namespace Umgebung {

    LayerStack::~LayerStack() {
        for (Layer* layer : layers) {
            layer->OnDetach();
            delete layer;
        }
        layers.clear();
    }

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