// Simple vertex shader for a colored cube
cbuffer MVP : register(b0)
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

struct VSInput {
    float3 pos : POSITION;
    float3 color : COLOR;
};

struct PSInput {
    float4 pos : SV_POSITION;
    float3 color : COLOR;
};

PSInput main(VSInput input) {
    PSInput output;
    float4 world = mul(float4(input.pos, 1.0f), model);
    float4 viewPos = mul(world, view);
    output.pos = mul(viewPos, proj);
    output.color = input.color;
    return output;
}
