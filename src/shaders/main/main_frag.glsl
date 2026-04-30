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

struct RBF {
    vec4 c;
    float r;
    float w;
    vec2 pad;
};

layout(std430, binding = 0) buffer InputRBFs {
    RBF rbfs[];
};

layout(std430, binding = 1) buffer OutputJss {
    vec4 jss[];
};

uniform Camera camera;
uniform vec2 texSize;
uniform uint frame;

uniform sampler3D jTexture;
uniform sampler3D densityTexture;
uniform sampler3D densityTildeTexture;
uniform ivec3 densityShape;

uniform int numRBF;
uniform float sigma_t;
uniform float sigma_s;
uniform float stepSize;

uniform vec3 backgroundColor;

uniform bool useNoise;
uniform sampler2D blueNoise;

#define PI 3.14159265
#define PI_4_SQRT 3.544907702

float one[16] = float[16](PI_4_SQRT, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

#pragma include "./rand.glsl"
#pragma include "./intersections.glsl"
#pragma include "../sh/sh.glsl"

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
void sh(vec3 d, out float sh[16]);
float sh_dot(float v[16], float u[16]);
#pragma FEND

float beerLambert(float dx, float D){
    return exp(-sigma_t * dx * D);
}

vec3 aabbPos(vec3 pos, AABB box){
    vec3 newPos = (pos - box.min) / (box.max - box.min);
    newPos = vec3(newPos.z, newPos.y, newPos.x);
    newPos *= (densityShape - vec3(1.)) / densityShape.x;
    return newPos;
}

float sampleDensity(vec3 pos, AABB box, sampler3D density){
    vec3 uvw = (pos - box.min) / (box.max - box.min);
    return texture(density, uvw).r;
}

float compute_rbf(RBF rbf, vec3 x){
    float inside = length(x - rbf.c.xyz) / rbf.r;
    return rbf.w * exp(-inside*inside);
}

bool rbfTooFar(RBF rbf, vec3 x){
    return length(rbf.c.xyz - x) > 3 * rbf.r;
}

vec3 Jss(Ray ray, AABB box, inout NoiseData nd){
    vec3 u = aabbPos(ray.origin, box);
    float D = sampleDensity(ray.origin, box, densityTexture);
    float D_tilde = sampleDensity(ray.origin, box, densityTildeTexture);
    float factor = D / D_tilde;
    factor = clamp(factor, 0., 100.);
    if (D_tilde < 1e-10) factor = 0;

    vec3 result = vec3(0.);
    float y[16];
    if (false){
        sh(normalize(-ray.dir), y);
        for(int l = 0; l < numRBF; l++) {
            RBF rbf = rbfs[l];
            if (rbfTooFar(rbf, u)) continue;

            float rbf_val = compute_rbf(rbf, u);
            float jss_r[16];
            float jss_g[16];
            float jss_b[16];
            for(int i = 0; i < 16; i++) {
                vec3 coeff = jss[l * 16 + i].rgb;
                jss_r[i] = coeff.r;
                jss_g[i] = coeff.g;
                jss_b[i] = coeff.b;
            }
            vec3 scatter = vec3(sh_dot(jss_r, y), sh_dot(jss_g, y), sh_dot(jss_b, y));
            result += rbf_val * clamp(scatter, 0., 15.);
        }
    }
    else{
        vec3 uvw = (ray.origin - box.min) / (box.max - box.min);
        uvw = vec3(uvw.z, uvw.y, uvw.x);
        result = texture(jTexture, uvw).rgb;
    }
    result *= factor;
    return result;
}

float sampleDensityTilde(vec3 origin, AABB box){
    float result = 0.;
    vec3 u = aabbPos(origin, box);
    for(int l = 0; l < numRBF; l++) {
        RBF rbf = rbfs[l];
        if (rbfTooFar(rbf, u)) continue;

        float rbf_val = compute_rbf(rbf, u);
        result += rbf_val;
    }
    return result;
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
        float D = sampleDensity(ray.origin, box, densityTexture);
        //D = sampleDensityTilde(ray.origin, box);
        t_x *= beerLambert(dx, D);
        J = Jss(ray, box, nd);
        Lm += t_x * sigma_t * J * dx;

        if (t_x < 0.01) break;

        ray.origin += ray.dir * dx;
    }
    vec3 Ld = t_x * backgroundColor;
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
        //return vec4(0);
        return vec4(Lout, 1);
    }
    //return vec4(0);
    return vec4(backgroundColor, 1);
}

Ray fovRay(vec2 pos, Ray ray){
    float fov = radians(mix(50.0, 90.0, 0 / 8.0));
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
    ray.dir = camera.lookDir;
    ray.origin = camera.pos;
    ray = fovRay(pos/* + randomInCircle_bn(bnoise1, bnoise2) / texSize.y*/, ray);

    FragColor = intersect(ray, nd);
}