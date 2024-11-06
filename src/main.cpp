#include "../include/core/Application.h"
#include "../include/core/Input.h"
#include "../include/core/Physics.h"
#include "../include/core/Renderer.h"
#include "../include/systems/SceneManager.h"
#include "../include/systems/UIManager.h"
#include "../include/utils/Logger.h"

int main()
{
	Umgebung::Application* Umgebung = new Umgebung::Application();
	Umgebung->Run();
	delete Umgebung;

	return 0;
}