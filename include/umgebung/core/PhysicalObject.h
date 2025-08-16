#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>

namespace Umgebung {
    class PhysicalObject {
    public:
        virtual ~PhysicalObject() = default;

        virtual void update(float dt) = 0;                 // physics + GPU kernels
        virtual void render() const = 0;                  // Irrlicht draw call

        glm::vec3 position() const { return pos_; }
        void setPosition(const glm::vec3& p) { pos_ = p; }

        glm::quat orientation() const { return ori_; }
        void setOrientation(const glm::quat& q) { ori_ = q; }

    protected:
        glm::vec3 pos_{ 0.f };
        glm::quat ori_{ glm::quat(1.f, 0.f, 0.f, 0.f) };
    };
}