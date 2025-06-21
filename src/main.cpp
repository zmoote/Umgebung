#include "vk_engine.h"

int main([[maybe_unused]]int argc, [[maybe_unused]]char* argv[])
{
	Umgebung::VulkanEngine engine;

	engine.init();

	engine.run();

	engine.cleanup();

	return 0;
}