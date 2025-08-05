cmake_minimum_required (VERSION 3.21)

set(VCPKG_TARGET_TRIPLET x64-windows)
set(VCPKG_CRT_LINKAGE dynamic)

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/submodules/vcpkg/scripts/buildsystems/vcpkg.cmake")

project ("Umgebung" LANGUAGES CXX CUDA VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_executable (${PROJECT_NAME} assets/icon/Umgebung.rc
                                src/Main.cpp
                                src/umgebung/AntiUpQuark.cpp
                                src/umgebung/Atom.cpp
                                src/umgebung/Boson.cpp
                                src/umgebung/Cell.cpp
                                src/umgebung/Consciousness.cpp
                                src/umgebung/Ecosystem.cpp
                                src/umgebung/Electron.cpp
                                src/umgebung/ElementaryParticle.cpp
                                src/umgebung/Galaxy.cpp
                                src/umgebung/Lepton.cpp
                                src/umgebung/Molecule.cpp
                                src/umgebung/Multiverse.cpp
                                src/umgebung/Neutron.cpp
                                src/umgebung/Organism.cpp
                                src/umgebung/Particle.cpp
                                src/umgebung/Phonon.cpp
                                src/umgebung/Photon.cpp
                                src/umgebung/Phryll.cpp
                                src/umgebung/PlanckSphericalUnit.cpp
                                src/umgebung/Planet.cpp
                                src/umgebung/Positron.cpp
                                src/umgebung/Proton.cpp
                                src/umgebung/QuantumVacuumFluctuation.cpp
                                src/umgebung/Quark.cpp
                                src/umgebung/SolarSystem.cpp
                                src/umgebung/Star.cpp
                                src/umgebung/SubatomicParticle.cpp
                                src/umgebung/SurfaceFeature.cpp
                                src/umgebung/Universe.cpp
                                src/umgebung/UpQuark.cpp
)

find_package(imgui CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(unofficial-omniverse-physx-sdk CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(winpixevent CONFIG REQUIRED)
find_package(Irrlicht CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE Irrlicht
                                              Microsoft::WinPixEventRuntime
                                              imgui::imgui
                                              nlohmann_json::nlohmann_json
                                              unofficial::omniverse-physx-sdk::sdk
                                              spdlog::spdlog
)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE
         "/wd4061" "/wd4365" "/wd4514" "/wd4571" "/wd4668" "/wd4710" "/wd4820" "/wd5039" "/wd5045"
         "/wd4265" "/wd4625" "/wd4626" "/wd4986" "/wd5204" "/wd5220")

    if(ENABLE_CODE_ANALYSIS)
        target_compile_options(${PROJECT_NAME} PRIVATE /analyze)
    endif()
endif()

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /Wall /EHsc /GR)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE ${COMPILER_DEFINES})
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILER_SWITCHES})
target_link_options(${PROJECT_NAME} PRIVATE ${LINKER_SWITCHES})

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>>,copy,true>
    $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}> $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMAND_EXPAND_LISTS
    )

# Optional: import the defined target to copy over the GPU acceleration libraries
# (3rd party provided by NVIDIA)
if(TARGET unofficial::omniverse-physx-sdk::gpu-library)
    if(UNIX)
        # Add rpath setting to find so libraries on unix based systems
        set_target_properties(${PROJECT_NAME} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "$ORIGIN"
        )
    endif()
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-library>
                    $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    if(WIN32)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        $<TARGET_FILE:unofficial::omniverse-physx-sdk::gpu-device-library>
                        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    endif()
else()
    message(WARNING "\GPU acceleration library target not defined
 - GPU acceleration will NOT be available!\
")
endif()