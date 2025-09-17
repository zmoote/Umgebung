#pragma once

#include <string>
#include <unordered_map>
#include <glm/glm.hpp>

namespace Umgebung::renderer::gl {

    class Shader {
    public:
        Shader(const char* vertexPath, const char* fragmentPath);
        ~Shader(); // Good practice to have a destructor to delete the program

        // --- Renamed and New Methods ---
        void bind() const;
        void unbind() const;

        // --- Existing Utility Uniform Functions ---
        void setBool(const std::string& name, bool value);
        void setInt(const std::string& name, int value);
        void setFloat(const std::string& name, float value);
        void setVec4(const std::string& name, const glm::vec4& value); // New
        void setMat4(const std::string& name, const glm::mat4& mat);

    private:
        void checkCompileErrors(unsigned int shader, const std::string& type);
        int getUniformLocation(const std::string& name);

        unsigned int programID_; // Changed from public 'id' to private 'programID_'
        std::unordered_map<std::string, int> uniformLocationCache_;
    };

} // namespace Umgebung::renderer::gl