#version 460 core
layout (location = 0) in vec3 aPos;

// Instanced Attributes
layout (location = 3) in mat4 aInstanceMatrix;
layout (location = 7) in vec4 aInstanceColor;
layout (location = 8) in float aInstanceDensity;
layout (location = 9) in float aInstancePhryllInfluence;
layout (location = 10) in float aInstanceSelected;
layout (location = 11) in float aInstanceIsManifesting;

out vec4 vInstanceColor;
out float vInstanceDensity;
out float vInstancePhryllInfluence;
out float vInstanceSelected;
out float vInstanceIsManifesting;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform bool uSelected;
uniform bool uIsInstanced;

void main()
{
    mat4 finalModel = uIsInstanced ? aInstanceMatrix : model;
    vInstanceColor = aInstanceColor;
    vInstanceDensity = aInstanceDensity;
    vInstancePhryllInfluence = aInstancePhryllInfluence;
    vInstanceSelected = aInstanceSelected;
    vInstanceIsManifesting = aInstanceIsManifesting;

    gl_Position = projection * view * finalModel * vec4(aPos, 1.0);
    
    // Better attenuation for cosmic distances
    vec4 viewPos = view * finalModel * vec4(aPos, 1.0);
    float dist = length(viewPos);
    
    // Base size that scales with distance but stays visible
    // We need a log-based or multi-stage scaling for 60 orders of magnitude
    float logDist = max(1.0, log(dist + 1.0) / log(10.0));
    float size = 50.0 / (logDist * 0.1 + 1.0);
    
    bool isSelected = uIsInstanced ? (aInstanceSelected > 0.5) : uSelected;

    if (isSelected) {
        size *= 2.0;
        if (size < 16.0) size = 16.0; 
    }
    
    gl_PointSize = size;
    
    // Clamp point size to ensure visibility
    if (gl_PointSize < 6.0) gl_PointSize = 6.0; 
    if (gl_PointSize > 128.0) gl_PointSize = 128.0;
}