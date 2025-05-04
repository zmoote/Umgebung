#pragma once
#include "../Core/Layer.hpp"

namespace Umgebung {

    class Renderer;

    class GuiLayer : public Layer {
    public:
        void OnAttach(Renderer* renderer);
        void OnDetach() override;
        void OnImGuiRender() override;
    };

} // namespace Umgebung