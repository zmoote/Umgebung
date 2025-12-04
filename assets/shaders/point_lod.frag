#version 460 core
out vec4 FragColor;

uniform vec4 uColor;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    if(dist > 0.5)
        discard;
        
    // Soft glow effect: alpha fades out from center
    // 0.0 at center -> 0.5 at edge
    // We want alpha 1.0 at center -> 0.0 at edge
    // Normalized dist (0 to 1) = dist * 2.0
    float alpha = 1.0 - (dist * 2.0);
    alpha = pow(alpha, 1.5); // Tweak power to adjust falloff hardness

    FragColor = vec4(uColor.rgb, uColor.a * alpha);
}