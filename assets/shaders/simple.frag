#version 460 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoords;
in vec4 vInstanceColor;
in float vInstanceDensity;
in float vInstancePhryllInfluence;
in float vInstanceSelected;
in float vInstanceIsManifesting;

uniform vec4 uColor;
uniform vec3 uViewPos;
uniform float uDensity;
uniform float uPhryllInfluence;
uniform float uTime;
uniform bool uSelected;
uniform bool uSourceView;
uniform bool uIsInstanced;
uniform bool uIsManifesting;

void main()
{
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(uViewPos - FragPos);

    // Fresnel effect
    float fresnel = dot(norm, viewDir);
    fresnel = clamp(1.0 - fresnel, 0.0, 1.0);
    fresnel = pow(fresnel, 3.0); 

    // Use instanced data if available, otherwise fallback to uniforms
    vec4 finalBaseColor = uIsInstanced ? vInstanceColor : uColor;
    float finalDensity = uIsInstanced ? vInstanceDensity : uDensity;
    float finalPhryllInfluence = uIsInstanced ? vInstancePhryllInfluence : uPhryllInfluence;
    bool isSelected = uIsInstanced ? (vInstanceSelected > 0.5) : uSelected;
    bool isManifesting = uIsInstanced ? (vInstanceIsManifesting > 0.5) : uIsManifesting;

    // Pulse based on 3-6-9 frequency (approx 3Hz, 6Hz, 9Hz harmonics)
    float sourcePulse = 0.5 + 0.5 * sin(uTime * 3.0) * sin(uTime * 6.0) * sin(uTime * 9.0);
    
    vec3 baseColor = finalBaseColor.rgb;
    float finalAlpha = finalBaseColor.a;

    // Phryll / Observer Effect: Glow increases based on observation focus
    // scalarFieldSystem calculates this focus in phryllInfluence
    float phryllGlow = finalPhryllInfluence * sourcePulse;
    baseColor += vec3(0.0, 0.5, 1.0) * phryllGlow * fresnel;
    
    // Manifestation: Entities that aren't fully manifested are ghostly/transparent
    if (!isManifesting) {
        finalAlpha *= (0.1 + finalPhryllInfluence * 0.4); // Increases with observation
    }

    if (uSourceView) {
        // Draw a glowing grid (Sacred Geometry / Source Code)
        vec2 grid = abs(fract(TexCoords * 10.0 - 0.5) - 0.5) / fwidth(TexCoords * 10.0);
        float line = min(grid.x, grid.y);
        float gridPattern = 1.0 - min(line, 1.0);
        
        // Grid glows with "Source" cyan/white
        vec3 sourceColor = vec3(0.0, 0.8, 1.0);
        baseColor = mix(baseColor * 0.2, sourceColor, gridPattern * sourcePulse);
        baseColor += sourceColor * fresnel * sourcePulse;
        finalAlpha = mix(finalAlpha, 0.8, gridPattern);
    } else {
        // Normal Vibrational Rendering
        float pulseSpeed = (finalDensity - 2.0) * 1.5;
        float pulse = 0.9 + 0.1 * sin(uTime * pulseSpeed);
        
        if (finalDensity > 5.0) {
            float glowAmount = (finalDensity - 5.0) * 0.1;
            baseColor += vec3(0.2, 0.3, 0.6) * fresnel * glowAmount;
            baseColor *= pulse;
        }

        if (finalDensity > 7.0) {
            finalAlpha *= clamp(1.5 - (finalDensity * 0.1), 0.3, 1.0);
        }
    }

    if (isSelected) {
        FragColor = vec4(mix(baseColor, vec3(1.0, 1.0, 0.0), 0.5), finalAlpha);
    } else {
        FragColor = vec4(baseColor, finalAlpha);
    }
}