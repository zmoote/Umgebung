#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
#include <glm/glm.hpp>

namespace Umgebung::ui::imgui {

    /**
 * @file ViewportPanel.hpp
 * @brief Contains the ViewportPanel class.
 */
#pragma once

#include "umgebung/ui/imgui/Panel.hpp"
#include "umgebung/renderer/Framebuffer.hpp"
#include <glm/glm.hpp>

namespace Umgebung::ui::imgui {

    /**
     * @brief A class for the viewport panel.
     */
    class ViewportPanel : public Panel {
    public:
        /**
         * @brief Construct a new Viewport Panel object.
         * 
         * @param framebuffer The framebuffer to display.
         */
        explicit ViewportPanel(renderer::Framebuffer* framebuffer);

        /**
         * @brief Renders the viewport panel.
         */
        void onUIRender() override;

        /**
         * @brief Get the Size object.
         * 
         * @return glm::vec2 
         */
        glm::vec2 getSize() const { return size_; }

        /**
         * @brief Returns whether the panel is focused.
         * 
         * @return true if the panel is focused, false otherwise.
         */
        bool isFocused() const { return focused_; }

    private:
        renderer::Framebuffer* framebuffer_ = nullptr; ///< The framebuffer to display.
        glm::vec2 size_{ 0.0f, 0.0f };               ///< The size of the viewport.

        bool focused_ = false;                      ///< Whether the viewport is focused.
    };

}

}