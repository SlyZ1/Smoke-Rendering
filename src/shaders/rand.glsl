uint pcg_hash(uint v) {
    v = v * 747796405u + 2891336453u;
    uint word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (word >> 22u) ^ word;
}

uint initSeed(uvec2 pos, uint frame) {
    uint v = pos.x + pos.y * 4096u + frame * 1315423911u;
    return pcg_hash(v);
}

float rand(inout uint seed) {
    seed = pcg_hash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

float rand_bn(ivec2 pixel, int frame) {
    ivec2 size = textureSize(blueNoise, 0);

    // rotation / offset temporel
    ivec2 offset = ivec2(frame % size.x, (frame * 7) % size.y);

    ivec2 coord = (pixel + offset) % size;

    return texelFetch(blueNoise, coord, 0).r;
}

vec2 randomInCircle(inout uint seed) {
    float u = rand(seed);
    float v = rand(seed);

    float theta = 2.0 * PI * u;
    float r     = sqrt(v);

    float x = r * cos(theta);
    float y = r * sin(theta);

    return vec2(x, y);
}

vec2 randomInCircle_bn(float bnoise1, float bnoise2) {
    float u = bnoise1;
    float v = bnoise2;

    float theta = 2.0 * PI * u;
    float r     = sqrt(v);

    float x = r * cos(theta);
    float y = r * sin(theta);

    return vec2(x, y);
}