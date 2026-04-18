// Example vertex shader
struct VertexInput
{
    float3 position : POSITION;
    float3 color : COLOR;
    float3 normal : NORMAL;
};

struct VertexOutput
{
    float4 position : SV_Position;
    float3 color : COLOR;
    float3 normal : NORMAL;
};

[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
    VertexOutput output;
    output.position = float4(input.position, 1.0);
    output.color = input.color;
    output.normal = input.normal;
    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    // A simple light direction
    float3 lightDir = normalize(float3(1.0, -1.0, -1.0));
    // Simple diffuse lighting
    float diff = max(dot(normalize(input.normal), lightDir), 0.0);
    float3 diffuse = diff * input.color;
    // Simple ambient light
    float3 ambient = 0.1 * input.color;
    return float4(diffuse + ambient, 1.0);
}
