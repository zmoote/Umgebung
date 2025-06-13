#pragma once

namespace Umgebung {

    class Layer {
    public:
        virtual ~Layer() = default;

        virtual void OnAttach() {}
        virtual void OnDetach() {}
        virtual void OnUpdate([[maybe_unused]] float deltaTime) {}
        virtual void OnRender() {}
        virtual void OnImGuiRender() {}
    };

} // namespace Umgebung