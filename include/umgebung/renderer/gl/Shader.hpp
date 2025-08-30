#pragma once

#include <string>

// We need GLM for the matrix functions
#include <glm/glm.hpp>

namespace Umgebung {
    namespace renderer {
        namespace gl {

            class Shader {
            public:
                // The program ID, which we get from OpenGL
                unsigned int id;

                // 1. Constructor: reads and builds the shader
                Shader(const char* vertexPath, const char* fragmentPath);

                // 2. use/activate the shader
                void use() const;

                // 3. Utility uniform functions
                void setBool(const std::string& name, bool value) const;
                void setInt(const std::string& name, int value) const;
                void setFloat(const std::string& name, float value) const;
                void setMat4(const std::string& name, const glm::mat4& mat) const;

            private:
                // Utility function for checking shader compilation/linking errors.
                void checkCompileErrors(unsigned int shader, const std::string& type);
            };

        } // namespace gl
    } // namespace renderer
} // namespace umgebung