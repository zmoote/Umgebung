#include "umgebung/renderer/gl/Shader.hpp"
#include "umgebung/util/LogMacros.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

namespace Umgebung {
	namespace renderer {
		namespace gl {

			Shader::Shader(const char* vertexPath, const char* fragmentPath)
			{
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
					UMGEBUNG_LOG_ERROR("SHADER::FILE_NOT_SUCCESSFULLY_READ: {}", e.what());
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

				id = glCreateProgram();
				glAttachShader(id, vertex);
				glAttachShader(id, fragment);
				glLinkProgram(id);
				checkCompileErrors(id, "PROGRAM");

				glDeleteShader(vertex);
				glDeleteShader(fragment);
			}

			void Shader::use() const
			{
				glUseProgram(id);
			}

			void Shader::setMat4(const std::string& name, const glm::mat4& mat) const
			{
				glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, glm::value_ptr(mat));
			}

			// --- THIS IS THE FIX ---
			void Shader::checkCompileErrors(unsigned int shader, const std::string& type)
			{
				int success;
				char infoLog[1024];
				if (type != "PROGRAM")
				{
					glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
					if (!success)
					{
						glGetShaderInfoLog(shader, 1024, NULL, infoLog);
						UMGEBUNG_LOG_ERROR("SHADER_COMPILATION_ERROR of type: {}\n{}\n -- --------------------------------------------------- -- ", type, infoLog);
					}
				}
				else
				{
					glGetProgramiv(shader, GL_LINK_STATUS, &success);
					if (!success)
					{
						glGetProgramInfoLog(shader, 1024, NULL, infoLog);
						UMGEBUNG_LOG_ERROR("PROGRAM_LINKING_ERROR of type: {}\n{}\n -- --------------------------------------------------- -- ", type, infoLog);
					}
				}
			}
		}
	}
}