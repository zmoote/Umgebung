#pragma once
#include <cstdint>

namespace Umgebung {
    namespace renderer {
        class Framebuffer {
        public:
            Framebuffer(uint32_t width, uint32_t height);
            ~Framebuffer();

            void bind() const;
            void unbind() const;

            void resize(uint32_t width, uint32_t height);
            uint32_t getColorAttachmentRendererID() const { return m_colorAttachment; }

        private:
            void invalidate();

            uint32_t m_rendererID = 0;
            uint32_t m_colorAttachment = 0;
            uint32_t m_depthAttachment = 0;
            uint32_t m_width, m_height;
        };
    }
}