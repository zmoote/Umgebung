#version 460 core
layout (location = 0) in vec3 aPos;

// We are temporarily removing the matrices to simplify the program
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    // This ignores all 3D math and just passes the raw vertex position
    // to the screen. The triangle is defined in "clip space" which is -1.0 to 1.0.
    gl_Position = vec4(aPos, 1.0);
}