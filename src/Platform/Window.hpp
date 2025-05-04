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

    private:
        GLFWwindow* window = nullptr;
    };

} // namespace Umgebung