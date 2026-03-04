#version 460 core
out vec4 FragColor;

uniform vec4 uColor;
uniform float uDensity;
uniform float uTime;
uniform bool uSelected;
uniform bool uSourceView;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    if(dist > 0.5)
        discard;
        
    float alpha = 1.0 - (dist * 2.0);
    vec3 finalColor = uColor.rgb;
    float pulse = 1.0;

    if (uSourceView) {
        // Draw a Star-like geometric point (Fractal Source)
        // Pulse based on 3-6-9
        float sourcePulse = 0.5 + 0.5 * sin(uTime * 3.0) * sin(uTime * 6.0) * sin(uTime * 9.0);
        
        // Multi-pointed star shape
        float angle = atan(coord.y, coord.x);
        float starShape = 0.5 + 0.5 * cos(angle * 9.0); // 9 points
        float mask = smoothstep(0.4, 0.5, dist + starShape * 0.1);
        if (mask > 0.5) discard;

        finalColor = vec3(0.0, 1.0, 1.0); // Cyan
        alpha = (1.0 - dist * 2.0) * sourcePulse;
    } else {
        // Normal Particle Rendering
        float pulseSpeed = (uDensity - 2.0) * 2.0;
        pulse = 0.8 + 0.2 * sin(uTime * pulseSpeed);
        
        float densityPower = 2.0 - (uDensity * 0.1); 
        alpha = pow(alpha, clamp(densityPower, 0.5, 3.0)); 

        if (uDensity > 5.0) {
            finalColor += vec3(0.1, 0.2, 0.4) * sin(uTime * 0.5 + uDensity);
        }
    }

    if (uSelected) {
        FragColor = vec4(1.0, 1.0, 0.0, alpha * pulse);
    } else {
        FragColor = vec4(finalColor, uColor.a * alpha * pulse);
    }
}