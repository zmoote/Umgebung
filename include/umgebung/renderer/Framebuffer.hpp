#pragma once

#include <cstdint>

namespace Umgebung::renderer {

    class Framebuffer {
    public:
        Framebuffer(uint32_t width, uint32_t height);
        ~Framebuffer();

        void bind();
        void unbind();
        void resize(uint32_t width, uint32_t height);

        // --- Add these missing getters ---
        uint32_t getColorAttachmentID() const { return colorAttachmentID_; }
        uint32_t getWidth() const { return width_; }
        uint32_t getHeight() const { return height_; }

    private:
        void invalidate();

        uint32_t rendererID_ = 0;
        uint32_t colorAttachmentID_ = 0;
        uint32_t depthAttachmentID_ = 0;
        uint32_t width_ = 0;
        uint32_t height_ = 0;
    };

} // namespace Umgebung::renderer