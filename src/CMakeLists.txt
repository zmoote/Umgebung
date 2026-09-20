# Core Library
add_library(umgebung_core
    core/engine.cu
    core/hierarchy.cpp
)
find_package(fmt CONFIG REQUIRED)
target_include_directories(umgebung_core PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(umgebung_core PUBLIC fmt::fmt)
set_target_properties(umgebung_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# CLI Interface
find_package(cxxopts CONFIG REQUIRED)
add_executable(umgebung-cli
    cli/main.cpp
)
target_link_libraries(umgebung-cli PRIVATE umgebung_core cxxopts::cxxopts fmt::fmt)

# GUI Interface
find_package(Qt5 COMPONENTS Core Widgets REQUIRED)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

add_executable(umgebung-gui WIN32
    gui/main.cpp
    gui/mainwindow.cpp
    gui/mainwindow.hpp
)
target_link_libraries(umgebung-gui PRIVATE umgebung_core Qt5::Core Qt5::Widgets)
