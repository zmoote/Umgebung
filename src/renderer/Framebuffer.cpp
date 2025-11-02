/**
 * @file Framebuffer.cpp
 * @brief Implements the Framebuffer class.
 */
#include "umgebung/renderer/Framebuffer.hpp"
#include <glad/glad.h>

namespace Umgebung::renderer {

    Framebuffer::Framebuffer(uint32_t width, uint32_t height)
        : width_(width), height_(height) {
        invalidate();
    }

    Framebuffer::~Framebuffer() {
        glDeleteFramebuffers(1, &rendererID_);
        glDeleteTextures(1, &colorAttachmentID_);
        glDeleteTextures(1, &depthAttachmentID_);
    }

    void Framebuffer::invalidate() {
        if (rendererID_) {
            glDeleteFramebuffers(1, &rendererID_);
            glDeleteTextures(1, &colorAttachmentID_);
            glDeleteTextures(1, &depthAttachmentID_);
        }

        glCreateFramebuffers(1, &rendererID_);
        glBindFramebuffer(GL_FRAMEBUFFER, rendererID_);

        glCreateTextures(GL_TEXTURE_2D, 1, &colorAttachmentID_);
        glBindTexture(GL_TEXTURE_2D, colorAttachmentID_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorAttachmentID_, 0);

        glCreateTextures(GL_TEXTURE_2D, 1, &depthAttachmentID_);
        glBindTexture(GL_TEXTURE_2D, depthAttachmentID_);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, width_, height_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depthAttachmentID_, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Framebuffer::bind() {
        glBindFramebuffer(GL_FRAMEBUFFER, rendererID_);
        glViewport(0, 0, width_, height_);
    }

    void Framebuffer::unbind() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void Framebuffer::resize(uint32_t width, uint32_t height) {
        width_ = width;
        height_ = height;
        invalidate();
    }

}