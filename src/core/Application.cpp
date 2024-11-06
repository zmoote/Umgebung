// Application.cpp
#include "../../include/core/Application.h"

namespace Umgebung {

	Application::Application() {
		Init();
	}

	Application::~Application() {
		Shutdown();
	}

	void Application::Init() {

	}

	void Application::Run() {
		while (true) {
			Update();
			Render();
		}
	}

	void Application::Update() {

	}

	void Application::Render() {

	}

	void Application::Shutdown() {

	}
}