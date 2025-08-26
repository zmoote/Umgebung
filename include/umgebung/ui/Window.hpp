#pragma once

#include <string>
#include <memory>

// Forward declare GLFWwindow to avoid including the GLFW header in our public header
struct GLFWwindow;

namespace Umgebung
{
    namespace ui
    {
        /**
         * @class Window
         * @brief Manages the application window, OpenGL context, and ImGui integration.
         *
         * This class encapsulates all functionality related to the windowing system (GLFW),
         * OpenGL context loading (GLAD), and the immediate mode GUI library (ImGui).
         */
        class Window
        {
        public:
            /**
             * @brief Constructs a Window object with specified dimensions and title.
             * @param width The initial width of the window.
             * @param height The initial height of the window.
             * @param title The title to be displayed on the window's title bar.
             */
            Window(int width, int height, const std::string& title);

            /**
             * @brief Destructor that handles the cleanup of all resources.
             */
            ~Window();

            // Delete copy constructor and copy assignment operator
            Window(const Window&) = delete;
            Window& operator=(const Window&) = delete;

            /**
             * @brief Initializes GLFW, GLAD, and ImGui. Creates the window and sets up contexts.
             * @return 0 on success, -1 on failure.
             */
            int init();

            /**
             * @brief Checks if the window should close.
             * @return True if the window should close, false otherwise.
             */
            bool shouldClose() const;

            /**
             * @brief Prepares the window for a new frame of rendering.
             * This typically involves clearing the screen.
             */
            void beginFrame();

            /**
             * @brief Finalizes the frame by swapping buffers and polling for events.
             */
            void endFrame();

            /**
             * @brief Starts a new ImGui frame.
             */
            void beginImGuiFrame();

            /**
             * @brief Renders the ImGui draw data.
             */
            void endImGuiFrame();

        private:
            GLFWwindow* m_window = nullptr; // Raw pointer to the GLFW window
            int m_width;
            int m_height;
            std::string m_title;
        };
    } // namespace ui
} // namespace Umgebung