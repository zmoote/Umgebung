#pragma once

#include <string>
#include <memory>
#include <functional>

struct GLFWwindow;

namespace Umgebung
{
    namespace ui
    {
        /**
 * @file Window.hpp
 * @brief Contains the Window class.
 */
#pragma once

#include <string>
#include <memory>
#include <functional>

struct GLFWwindow;

namespace Umgebung
{
    namespace ui
    {
        /**
         * @brief A wrapper for the GLFW window.
         */
        class Window
        {
        public:
            using ResizeCallbackFn = std::function<void(int, int)>;

            /**
             * @brief Construct a new Window object.
             * 
             * @param width The width of the window.
             * @param height The height of the window.
             * @param title The title of the window.
             */
            Window(int width, int height, const std::string& title);

            /**
             * @brief Destroy the Window object.
             */
            ~Window();

            Window(const Window&) = delete;
            Window& operator=(const Window&) = delete;

            /**
             * @brief Initializes the window.
             * 
             * @return int 0 on success, -1 on failure.
             */
            int init();

            /**
             * @brief Returns whether the window should close.
             * 
             * @return true if the window should close, false otherwise.
             */
            bool shouldClose() const;

            /**
             * @brief Begins a new frame.
             */
            void beginFrame();

            /**
             * @brief Ends the current frame.
             */
            void endFrame();

            /**
             * @brief Begins a new ImGui frame.
             */
            void beginImGuiFrame();

            /**
             * @brief Ends the current ImGui frame.
             */
            void endImGuiFrame();

            /**
             * @brief Clears the window.
             */
            void clear() const;

            /**
             * @brief Set the Resize Callback object.
             * 
             * @param callback The callback function.
             */
            void setResizeCallback(const ResizeCallbackFn& callback);

            /**
             * @brief Get the GLFWwindow object.
             * 
             * @return GLFWwindow* 
             */
            GLFWwindow* getGLFWwindow() const { return m_window; }

        private:
            GLFWwindow* m_window = nullptr; ///< The GLFW window.
            int m_width;                    ///< The width of the window.
            int m_height;                   ///< The height of the window.
            std::string m_title;            ///< The title of the window.

            /**
             * @brief Called when the window is resized.
             * 
             * @param width The new width.
             * @param height The new height.
             */
            void onResize(int width, int height);

            ResizeCallbackFn resizeCallback_; ///< The resize callback function.

            /**
             * @brief The framebuffer size callback function.
             * 
             * @param window The GLFW window.
             * @param width The new width.
             * @param height The new height.
             */
            static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
        };
    }
}
    }
}