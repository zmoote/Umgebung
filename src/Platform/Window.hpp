#pragma once
#include <string>
struct GLFWwindow;

namespace Umgebung {

    class Window {
    public:
        bool ShouldClose() const;
        void Create(const std::string& title, uint32_t width, uint32_t height);
        void PollEvents();
        GLFWwindow* GetNativeWindow();
        void SetIcon(const std::string& path);
        bool WasResized() const;
        void ResetResizedFlag();

    private:
        GLFWwindow* window = nullptr;
        bool framebufferResized = false;
    };

} // namespace Umgebung