#include "umgebung/renderer/Framebuffer.hpp"
#include <glad/glad.h>

namespace umgebung {
    namespace renderer {
        Framebuffer::Framebuffer(uint32_t width, uint32_t height)
            : m_width(width), m_height(height) {
            invalidate();
        }

        Framebuffer::~Framebuffer() {
            glDeleteFramebuffers(1, &m_rendererID);
            glDeleteTextures(1, &m_colorAttachment);
            glDeleteTextures(1, &m_depthAttachment);
        }

        void Framebuffer::invalidate() {
            if (m_rendererID) {
                glDeleteFramebuffers(1, &m_rendererID);
                glDeleteTextures(1, &m_colorAttachment);
                glDeleteTextures(1, &m_depthAttachment);
            }

            glCreateFramebuffers(1, &m_rendererID);
            glBindFramebuffer(GL_FRAMEBUFFER, m_rendererID);

            glCreateTextures(GL_TEXTURE_2D, 1, &m_colorAttachment);
            glBindTexture(GL_TEXTURE_2D, m_colorAttachment);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_colorAttachment, 0);

            glCreateTextures(GL_TEXTURE_2D, 1, &m_depthAttachment);
            glBindTexture(GL_TEXTURE_2D, m_depthAttachment);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, m_width, m_height);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, m_depthAttachment, 0);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        void Framebuffer::bind() const {
            glBindFramebuffer(GL_FRAMEBUFFER, m_rendererID);
            glViewport(0, 0, m_width, m_height);
        }

        void Framebuffer::unbind() const {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }

        void Framebuffer::resize(uint32_t width, uint32_t height) {
            m_width = width;
            m_height = height;
            invalidate();
        }
    }
}