// Simple pixel shader for a colored cube
struct PSInput {
    float4 pos : SV_POSITION;
    float3 color : COLOR;
};

float4 main(PSInput input) : SV_TARGET {
    return float4(input.color, 1.0f);
}
