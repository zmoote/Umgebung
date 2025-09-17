#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/util/LogMacros.hpp" // Or your preferred logging header

#include <fstream>
#include <sstream>
#include <iostream>
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

namespace Umgebung::renderer::gl {

    Shader::Shader(const char* vertexPath, const char* fragmentPath) {
        std::string vertexCode;
        std::string fragmentCode;
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;

        vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        try {
            vShaderFile.open(vertexPath);
            fShaderFile.open(fragmentPath);
            std::stringstream vShaderStream, fShaderStream;
            vShaderStream << vShaderFile.rdbuf();
            fShaderStream << fShaderFile.rdbuf();
            vShaderFile.close();
            fShaderFile.close();
            vertexCode = vShaderStream.str();
            fragmentCode = fShaderStream.str();
        }
        catch (std::ifstream::failure& e) {
            // UMGEBUNG_LOG_ERROR("SHADER::FILE_NOT_SUCCESSFULLY_READ: {}", e.what());
            std::cerr << "SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
        }

        const char* vShaderCode = vertexCode.c_str();
        const char* fShaderCode = fragmentCode.c_str();

        unsigned int vertex, fragment;

        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");

        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");

        programID_ = glCreateProgram();
        glAttachShader(programID_, vertex);
        glAttachShader(programID_, fragment);
        glLinkProgram(programID_);
        checkCompileErrors(programID_, "PROGRAM");

        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }

    Shader::~Shader() {
        glDeleteProgram(programID_);
    }

    void Shader::bind() const {
        glUseProgram(programID_);
    }

    void Shader::unbind() const {
        glUseProgram(0);
    }

    void Shader::setBool(const std::string& name, bool value) {
        glUniform1i(getUniformLocation(name), (int)value);
    }

    void Shader::setInt(const std::string& name, int value) {
        glUniform1i(getUniformLocation(name), value);
    }

    void Shader::setFloat(const std::string& name, float value) {
        glUniform1f(getUniformLocation(name), value);
    }

    void Shader::setVec4(const std::string& name, const glm::vec4& value) {
        glUniform4fv(getUniformLocation(name), 1, &value[0]);
    }

    void Shader::setMat4(const std::string& name, const glm::mat4& mat) {
        glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
    }

    int Shader::getUniformLocation(const std::string& name) {
        if (uniformLocationCache_.find(name) != uniformLocationCache_.end()) {
            return uniformLocationCache_[name];
        }

        int location = glGetUniformLocation(programID_, name.c_str());
        if (location == -1) {
            // UMGEBUNG_LOG_WARN("Uniform '{}' not found in shader!", name);
            std::cout << "Warning: Uniform '" << name << "' not found!" << std::endl;
        }

        uniformLocationCache_[name] = location;
        return location;
    }

    void Shader::checkCompileErrors(unsigned int shader, const std::string& type) {
        int success;
        char infoLog[1024];
        if (type != "PROGRAM") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                // UMGEBUNG_LOG_ERROR("SHADER_COMPILATION_ERROR of type: {}\\n{}", type, infoLog);
                std::cerr << "SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << std::endl;
            }
        }
        else {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                // UMGEBUNG_LOG_ERROR("PROGRAM_LINKING_ERROR of type: {}\\n{}", type, infoLog);
                std::cerr << "PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << std::endl;
            }
        }
    }

} // namespace Umgebung::renderer::gl