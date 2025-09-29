#pragma once

#include <string>
#include <memory>
#include <functional>

struct GLFWwindow;

namespace Umgebung
{
    namespace ui
    {
        class Window
        {
        public:
            using ResizeCallbackFn = std::function<void(int, int)>;

            Window(int width, int height, const std::string& title);

            ~Window();

            Window(const Window&) = delete;
            Window& operator=(const Window&) = delete;

            int init();

            bool shouldClose() const;

            void beginFrame();

            void endFrame();

            void beginImGuiFrame();

            void endImGuiFrame();

            void clear() const;

            void setResizeCallback(const ResizeCallbackFn& callback);

            GLFWwindow* getGLFWwindow() const { return m_window; }

        private:
            GLFWwindow* m_window = nullptr;
            int m_width;
            int m_height;
            std::string m_title;

            void onResize(int width, int height);

            ResizeCallbackFn resizeCallback_;

            static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
        };
    }
}