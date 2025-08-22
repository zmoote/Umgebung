#include <irrlicht.h>
#include <PxConfig.h>
#include <PxPhysicsAPI.h>
#include <iostream>
#include "umgebung/Multiverse.hpp"

int main(int argc, char** argv) {

    std::cout << "Creating Irrlicht Device..." << std::endl;

    irr::IrrlichtDevice* device = irr::createDevice(irr::video::EDT_OPENGL, irr::core::dimension2d<irr::u32>(800, 600));

    device->setResizable(true);
    device->maximizeWindow();

    std::cout << "Creating Irrlicht Driver..." << std::endl;

    irr::video::IVideoDriver* driver = device->getVideoDriver();

    std::cout << "Creating Irrlicht Scene Manager..." << std::endl;

    irr::scene::ISceneManager* smgr = device->getSceneManager();

    std::cout << "Starting PhysX up.." << std::endl;
    physx::PxDefaultAllocator allocator;
    physx::PxDefaultErrorCallback error_callback;
    std::cout << "Creating PhysX Foundation..." << std::endl;
    auto foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, error_callback);

    // Create a CUDA context manager
    std::cout << "Creating PhysX CUDA context manager..." << std::endl;
    physx::PxCudaContextManagerDesc cudaContextManagerDesc;
    auto cudaContextManager = PxCreateCudaContextManager(
        *foundation, cudaContextManagerDesc, PxGetProfilerCallback()
    );

    // Check if CUDA context manager is valid
    if (cudaContextManager && cudaContextManager->contextIsValid()) {
        std::cout << "GPU acceleration is available" << std::endl;
    }
    else {
        std::cout << "GPU acceleration is not available" << std::endl;
    }

    // Create a physics SDK object with GPU acceleration
    physx::PxTolerancesScale tolerancesScale;
    physx::PxSceneDesc sceneDesc(tolerancesScale);
    sceneDesc.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
    auto dispatcher = physx::PxDefaultCpuDispatcherCreate(4);
    sceneDesc.cpuDispatcher = dispatcher;
    sceneDesc.filterShader = physx::PxDefaultSimulationFilterShader;
    sceneDesc.cudaContextManager = cudaContextManager;
    sceneDesc.flags |= physx::PxSceneFlag::eENABLE_GPU_DYNAMICS;
    sceneDesc.broadPhaseType = physx::PxBroadPhaseType::eGPU;
    auto physics_sdk = PxCreatePhysics(
        PX_PHYSICS_VERSION, *foundation, tolerancesScale, true, nullptr
    );
    auto scene = physics_sdk->createScene(sceneDesc);

    std::cout << "PhysX set up" << std::endl;

    smgr->addCameraSceneNodeFPS();

    //physics_sdk->createDeformableSurface(cudaContextManager&);

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
    // Release resources
    PX_RELEASE(scene);
    PX_RELEASE(dispatcher);
    PX_RELEASE(physics_sdk);
    PX_RELEASE(cudaContextManager);
    PX_RELEASE(foundation);

    std::cout << "Shutting down.." << std::endl;
    device->drop();
    return 0;
}