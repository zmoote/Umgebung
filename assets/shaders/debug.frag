#version 330 core
out vec4 FragColor;

in float vAlpha;

uniform vec4 color;

void main()
{
    FragColor = vec4(color.rgb, color.a * vAlpha);
}
