#pragma once

namespace Umgebung {
	namespace renderer {
		class Renderer {
			public:
				void init();
				void draw();

			private:
				unsigned int m_triangleVAO;
				unsigned int m_triangleVBO;
		};
	}
}