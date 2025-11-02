#pragma once

#include <cstdint>

namespace Umgebung::renderer {

    /**
 * @file Framebuffer.hpp
 * @brief Contains the Framebuffer class.
 */
#pragma once

#include <cstdint>

namespace Umgebung::renderer {

    /**
     * @brief A class for creating and managing an OpenGL framebuffer.
     */
    class Framebuffer {
    public:
        /**
         * @brief Construct a new Framebuffer object.
         * 
         * @param width The width of the framebuffer.
         * @param height The height of the framebuffer.
         */
        Framebuffer(uint32_t width, uint32_t height);

        /**
         * @brief Destroy the Framebuffer object.
         */
        ~Framebuffer();

        /**
         * @brief Binds the framebuffer.
         */
        void bind();

        /**
         * @brief Unbinds the framebuffer.
         */
        void unbind();

        /**
         * @brief Resizes the framebuffer.
         * 
         * @param width The new width.
         * @param height The new height.
         */
        void resize(uint32_t width, uint32_t height);

        /**
         * @brief Get the Color Attachment ID object.
         * 
         * @return uint32_t 
         */
        uint32_t getColorAttachmentID() const { return colorAttachmentID_; }

        /**
         * @brief Get the Width object.
         * 
         * @return uint32_t 
         */
        uint32_t getWidth() const { return width_; }

        /**
         * @brief Get the Height object.
         * 
         * @return uint32_t 
         */
        uint32_t getHeight() const { return height_; }

    private:
        /**
         * @brief Invalidates and recreates the framebuffer.
         */
        void invalidate();

        uint32_t rendererID_ = 0;        ///< The renderer ID of the framebuffer.
        uint32_t colorAttachmentID_ = 0; ///< The color attachment ID of the framebuffer.
        uint32_t depthAttachmentID_ = 0; ///< The depth attachment ID of the framebuffer.
        uint32_t width_ = 0;             ///< The width of the framebuffer.
        uint32_t height_ = 0;            ///< The height of the framebuffer.
    };

}

}