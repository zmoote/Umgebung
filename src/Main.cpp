#include <irrlicht.h>
#include <iostream>

int main(int argc, char** argv)
{

	irr::IrrlichtDevice* device = irr::createDevice(irr::video::EDT_OPENGL, irr::core::dimension2d<irr::u32>(640, 480));
	
	if (!device) {
		return 1;
	}
	
	device->maximizeWindow();

	irr::video::IVideoDriver* driver = device->getVideoDriver();

	irr::scene::ISceneManager* smgr = device->getSceneManager();

	//irr::SKeyMap keyMap[4];
	//keyMap[0].Action = irr::EKA_MOVE_FORWARD;
	//keyMap[0].KeyCode = irr::KEY_KEY_W;

	//keyMap[1].Action = irr::EKA_MOVE_BACKWARD;
	//keyMap[1].KeyCode = irr::KEY_KEY_S;

	//keyMap[2].Action = irr::EKA_STRAFE_LEFT;
	//keyMap[2].KeyCode = irr::KEY_KEY_A;

	//keyMap[3].Action = irr::EKA_STRAFE_RIGHT;
	//keyMap[3].KeyCode = irr::KEY_KEY_D;

	//smgr->addCameraSceneNodeFPS(0, 100, 0.5f, -1, keyMap, 8);

	//device->getCursorControl()->setVisible(false);

	int lastFPS = -1;

	while (device->run())
	{
		if (device->isWindowActive())
		{
			driver->beginScene(true, true, irr::video::SColor(255, 200, 200, 200));
			smgr->drawAll();
			driver->endScene();

			int fps = driver->getFPS();

			if (lastFPS != fps)
			{
				irr::core::stringw str = L"Umgebung [";
				str += driver->getName();
				str += "] FPS:";
				str += fps;

				device->setWindowCaption(str.c_str());
				lastFPS = fps;
			}
		}
		else
			device->yield();
	}

	device->drop();
	return 0;
}