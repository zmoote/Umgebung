#version 460 core
layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    
    // Simple distance attenuation for point size
    float dist = length(view * model * vec4(aPos, 1.0));
    gl_PointSize = 3000.0 / dist; // Increased scale factor for better visibility
    
    // Clamp point size
    if (gl_PointSize < 4.0) gl_PointSize = 4.0; // Increased min size so they are always visible dots
    if (gl_PointSize > 64.0) gl_PointSize = 64.0;
}