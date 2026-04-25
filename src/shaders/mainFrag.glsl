#version 430 core
out vec4 FragColor;
in vec4 vClipPos;

struct Camera {
    vec3 pos;
    vec3 lookDir;
};

struct AABB {
    vec3 min;
    vec3 max;
};

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct NoiseData {
    float bnoise;
    uint seed;
};

uniform Camera camera;
uniform vec2 texSize;
uniform uint frame;

uniform sampler3D densityTexture;

uniform float sigma_t;
uniform float sigma_s;
uniform float stepSize;

uniform vec3 backgroundColor;

uniform bool useNoise;
uniform sampler2D blueNoise;

uniform vec4 zhCoeffs;
uniform sampler1D aTable;
uniform sampler1D bTable;
uniform float maxDensityMagnitude; 

#define PI 3.14159265
#define PI_4_SQRT 3.544907702

float one[16] = float[16](PI_4_SQRT, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

#pragma include "./rand.glsl"
#pragma include "./intersections.glsl"
#pragma include "./sh_triple_product.glsl"
#pragma include "./sh_conv.glsl"
#pragma include "./sh.glsl"

#pragma FDECLARE
// RAND.GLSL
void initSeed(uvec2 pos, uint frame);
float rand(inout uint seed);
float rand_bn(ivec2 pixel, int frame);
float randomInCircle(inout uint seed);
float randomInCircle_bn(float bnoise1, float bnoise2);

// INTERSECTIONS.GLSL
float intersectAABB(Ray ray, AABB box);
bool isInAABB(vec3 point, AABB box);

// SH.GLSL
void sh_order3(vec3 d, out float sh[16]);

// SH_TRIPLE_PRODUCT.GLSL
void sh_triple_product3(float f[16], float g[16], out float result[16]);

// SH_CONV.GLSL
void sh_conv3(float f[16], float g[16], out float result[16]);
#pragma FEND

float beerLambert(float dx, float D){
    return exp(-sigma_t * dx * D);
}

float sampleScalarFlowDensity(vec3 pos, AABB box){
    vec3 uvw = (pos - box.min) / (box.max - box.min);
    return texture(densityTexture, uvw).r * 20;
}

vec3 J(Ray ray, AABB box, inout NoiseData nd){
    return vec3(1);
}

vec3 L_debug(Ray ray, AABB box, inout NoiseData nd){
    float dx = max(stepSize, 1e-4);
    float accumulated = 0.0;
    while(isInAABB(ray.origin, box)){
        accumulated += sampleScalarFlowDensity(ray.origin, box) * dx;
        ray.origin += ray.dir * dx;
    }
    return vec3(accumulated);
}

vec3 L(Ray ray, AABB box, inout NoiseData nd) {
    float dx = max(stepSize, 1e-4);
    if (useNoise){
        ray.origin += ray.dir * nd.bnoise * dx;
    }

    float t_x = 1;
    vec3 Lm = vec3(0);
    vec3 J = vec3(1);
    while(isInAABB(ray.origin, box)){
        vec3 pos = vec3(49 / 100.0, 7 / 170.0, 43 / 100.0);
        pos = (pos - vec3(0.5,0.5*1.7, 0.5)) * 2;
        if (length(ray.origin - pos) < 0.05){
            //return vec3(1,0,0);
        }
        float D = sampleScalarFlowDensity(ray.origin, box);
        //D = 0.5;
        t_x *= beerLambert(dx, D);
        Lm += t_x * sigma_t * D * J * dx;

        if (t_x < 0.01) break;

        ray.origin += ray.dir * dx;
    }
    vec3 Ld = t_x * backgroundColor;
    //return vec3(1 - t_x);
    return Lm + Ld;
}

vec4 intersect(Ray ray, inout NoiseData nd){
    AABB box = AABB(vec3(-1, -1.7, -1), vec3(1, 1.7, 1));
    float t = intersectAABB(ray, box);

    if (isInAABB(ray.origin, box))
        t = 0;

    if (t >= 0){
        ray.origin += ray.dir * (t + 1e-4);
        vec3 Lout = L(ray, box, nd);
        return vec4(Lout, 1);
    }
    //return vec4(0);
    return vec4(backgroundColor, 1);
}

Ray fovRay(vec2 pos, Ray ray){
    float fov = radians(mix(50, 90, 0 / 8.0));
    vec3 forward = normalize(camera.lookDir);

    vec3 worldUp = abs(forward.y) < 0.999
                 ? vec3(0,1,0)
                 : vec3(0,0,1);

    vec3 right = normalize(cross(forward, worldUp));
    vec3 up    = cross(right, forward);

    float tanHalfFov = tan(fov * 0.5);

    vec3 dir = forward + (right * pos.x + up * pos.y) * tanHalfFov;
    ray.dir = normalize(dir);
    return ray;
}

vec2 ratio(vec2 vec){
    return vec2(vec.x * texSize.x / texSize.y, vec.y);
}

void main()
{
    uint seed = initSeed(uvec2(gl_FragCoord.xy), frame);
    vec2 pos = ratio(vClipPos.xy);
    vec2 uv = (vClipPos.xy + vec2(1)) * 0.5;
    float bnoise1 = rand_bn(ivec2(ratio(uv) * texSize.y), int(frame));
    NoiseData nd = NoiseData(bnoise1, seed);

    Ray ray;
    ray.origin = camera.pos;
    ray.dir = camera.lookDir;
    ray = fovRay(pos/* + randomInCircle_bn(bnoise1, bnoise2) / texSize.y*/, ray);

    FragColor = intersect(ray, nd);
}