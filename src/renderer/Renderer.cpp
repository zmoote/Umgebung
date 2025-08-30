#include "umgebung/renderer/Renderer.hpp"
#include <glad/glad.h>

namespace Umgebung {
	namespace renderer {
		void Renderer::init()
		{
			// 1. Define vertex data
			float vertices[] = {
				-0.5f, -0.5f, 0.0f, // left  
				 0.5f, -0.5f, 0.0f, // right 
				 0.0f,  0.5f, 0.0f  // top   
			};

			// --- Modern DSA Style ---

			// 2. Create VBO and upload data
			glCreateBuffers(1, &m_triangleVBO);
			glNamedBufferData(m_triangleVBO, sizeof(vertices), vertices, GL_STATIC_DRAW);

			// 3. Create VAO
			glCreateVertexArrays(1, &m_triangleVAO);

			// 4. Connect the VBO to the VAO's binding point 0
			glVertexArrayVertexBuffer(m_triangleVAO, 0, m_triangleVBO, 0, 3 * sizeof(float));

			// --- THIS IS THE FIX ---
			// 5. Enable the vertex attribute for our VAO
			glEnableVertexArrayAttrib(m_triangleVAO, 0);
			// -------------------------

			// 6. Set up the attribute format (the "recipe")
			glVertexArrayAttribFormat(m_triangleVAO, 0, 3, GL_FLOAT, GL_FALSE, 0);

			// 7. Link the attribute to the binding point
			glVertexArrayAttribBinding(m_triangleVAO, 0, 0);
		}

		void Renderer::draw()
		{
			glBindVertexArray(m_triangleVAO);
			glDrawArrays(GL_TRIANGLES, 0, 3);
		}
	}
}