#pragma once

namespace Umgebung {

    class Window;

    class Renderer {
    public:
        void Init(Window* window);
        void BeginFrame();
        void EndFrame();
        void Cleanup();
    };

} // namespace Umgebung