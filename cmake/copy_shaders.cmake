# This script ensures the shaders output directory exists and copies shaders.
file(MAKE_DIRECTORY "${CMAKE_ARGV3}")
foreach(shader IN LISTS ARGN)
    file(COPY "${shader}" DESTINATION "${CMAKE_ARGV3}")
endforeach()
