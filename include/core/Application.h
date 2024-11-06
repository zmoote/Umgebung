// Application.h
#pragma once

namespace Umgebung {

	class Application {
	public:
		Application();
		~Application();

		void Run(); // Main loop
	
	private:
		void Init(); // Initialize systems
		void Shutdown(); // Clean up resources
		void Update(); // Update application state
		void Render(); // Render the scene
	};

}