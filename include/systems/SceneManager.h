#pragma once
#include <string>

namespace Umgebung {

	class SceneManager {
	public:
		SceneManager();
		~SceneManager();

		void LoadScene(const std::string& sceneName);
		void Update(float deltaTime);

	private:

	};

}