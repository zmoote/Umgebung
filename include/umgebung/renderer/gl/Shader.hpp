#pragma once

#include <string>
#include <unordered_map>
#include <glm/glm.hpp>

namespace Umgebung::renderer::gl {

    /**
 * @file Shader.hpp
 * @brief Contains the Shader class.
 */
#pragma once

#include <string>
#include <unordered_map>
#include <glm/glm.hpp>

namespace Umgebung::renderer::gl {

    /**
     * @brief A wrapper for an OpenGL shader program.
     */
    class Shader {
    public:
        /**
         * @brief Construct a new Shader object.
         * 
         * @param vertexPath The path to the vertex shader.
         * @param fragmentPath The path to the fragment shader.
         */
        Shader(const char* vertexPath, const char* fragmentPath);

        /**
         * @brief Destroy the Shader object.
         */
        ~Shader();

        /**
         * @brief Binds the shader.
         */
        void bind() const;

        /**
         * @brief Unbinds the shader.
         */
        void unbind() const;

        /**
         * @brief Sets a boolean uniform.
         * 
         * @param name The name of the uniform.
         * @param value The value of the uniform.
         */
        void setBool(const std::string& name, bool value);

        /**
         * @brief Sets an integer uniform.
         * 
         * @param name The name of the uniform.
         * @param value The value of the uniform.
         */
        void setInt(const std::string& name, int value);

        /**
         * @brief Sets a float uniform.
         * 
         * @param name The name of the uniform.
         * @param value The value of the uniform.
         */
        void setFloat(const std::string& name, float value);

        /**
         * @brief Sets a vec4 uniform.
         * 
         * @param name The name of the uniform.
         * @param value The value of the uniform.
         */
        void setVec4(const std::string& name, const glm::vec4& value);

        /**
         * @brief Sets a mat4 uniform.
         * 
         * @param name The name of the uniform.
         * @param mat The value of the uniform.
         */
        void setMat4(const std::string& name, const glm::mat4& mat);

    private:
        /**
         * @brief Checks for shader compilation errors.
         * 
         * @param shader The shader to check.
         * @param type The type of the shader.
         */
        void checkCompileErrors(unsigned int shader, const std::string& type);

        /**
         * @brief Get the Uniform Location object.
         * 
         * @param name The name of the uniform.
         * @return int 
         */
        int getUniformLocation(const std::string& name);

        unsigned int programID_; ///< The program ID of the shader.
        std::unordered_map<std::string, int> uniformLocationCache_; ///< The cache of uniform locations.
    };

}

}