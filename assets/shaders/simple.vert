#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

// Instanced Attributes
layout (location = 3) in mat4 aInstanceMatrix;
layout (location = 7) in vec4 aInstanceColor;
layout (location = 8) in float aInstanceDensity;
layout (location = 9) in float aInstancePhryllInfluence;
layout (location = 10) in float aInstanceSelected;
layout (location = 11) in float aInstanceIsManifesting;

out vec3 Normal;
out vec3 FragPos;
out vec2 TexCoords;
out vec4 vInstanceColor;
out float vInstanceDensity;
out float vInstancePhryllInfluence;
out float vInstanceSelected;
out float vInstanceIsManifesting;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform bool uIsInstanced;

void main()
{
    mat4 finalModel = uIsInstanced ? aInstanceMatrix : model;
    vInstanceColor = aInstanceColor;
    vInstanceDensity = aInstanceDensity;
    vInstancePhryllInfluence = aInstancePhryllInfluence;
    vInstanceSelected = aInstanceSelected;
    vInstanceIsManifesting = aInstanceIsManifesting;

    FragPos = vec3(finalModel * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(finalModel))) * aNormal;
    TexCoords = aTexCoords;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}