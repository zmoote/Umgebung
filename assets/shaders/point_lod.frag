#version 460 core
out vec4 FragColor;

in vec4 vInstanceColor;
in float vInstanceDensity;
in float vInstancePhryllInfluence;
in float vInstanceSelected;
in float vInstanceIsManifesting;

uniform vec4 uColor;
uniform float uDensity;
uniform float uPhryllInfluence;
uniform float uTime;
uniform bool uSelected;
uniform bool uSourceView;
uniform bool uIsInstanced;
uniform bool uIsManifesting;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    if(dist > 0.5)
        discard;
        
    // Use instanced data if available, otherwise fallback to uniforms
    vec4 finalBaseColor = uIsInstanced ? vInstanceColor : uColor;
    float finalDensity = uIsInstanced ? vInstanceDensity : uDensity;
    float finalPhryllInfluence = uIsInstanced ? vInstancePhryllInfluence : uPhryllInfluence;
    bool isSelected = uIsInstanced ? (vInstanceSelected > 0.5) : uSelected;
    bool isManifesting = uIsInstanced ? (vInstanceIsManifesting > 0.5) : uIsManifesting;

    float alpha = 1.0 - (dist * 2.0);
    vec3 finalColor = finalBaseColor.rgb;
    float pulse = 1.0;

    // Pulse based on 3-6-9 frequency (approx 3Hz, 6Hz, 9Hz harmonics)
    float sourcePulse = 0.5 + 0.5 * sin(uTime * 3.0) * sin(uTime * 6.0) * sin(uTime * 9.0);

    // Phryll / Observer Effect: Points glow when observed
    float phryllGlow = finalPhryllInfluence * sourcePulse;
    finalColor += vec3(0.0, 0.8, 1.0) * phryllGlow;

    // Manifestation: Non-manifested points are transparent
    if (!isManifesting) {
        alpha *= (0.2 + finalPhryllInfluence * 0.8);
    }

    if (uSourceView) {
        // Draw a Star-like geometric point (Fractal Source)
        // Multi-pointed star shape
        float angle = atan(coord.y, coord.x);
        float starShape = 0.5 + 0.5 * cos(angle * 9.0); // 9 points
        float mask = smoothstep(0.4, 0.5, dist + starShape * 0.1);
        if (mask > 0.5) discard;

        finalColor = vec3(0.0, 1.0, 1.0); // Cyan
        alpha *= sourcePulse;
    } else {
        // Normal Particle Rendering
        float pulseSpeed = (finalDensity - 2.0) * 2.0;
        pulse = 0.8 + 0.2 * sin(uTime * pulseSpeed);
        
        float densityPower = 2.0 - (finalDensity * 0.1); 
        alpha = pow(alpha, clamp(densityPower, 0.5, 3.0)); 

        if (finalDensity > 5.0) {
            finalColor += vec3(0.1, 0.2, 0.4) * sin(uTime * 0.5 + finalDensity);
        }
    }

    if (isSelected) {
        FragColor = vec4(1.0, 1.0, 0.0, alpha * pulse);
    } else {
        FragColor = vec4(finalColor, finalBaseColor.a * alpha * pulse);
    }
}