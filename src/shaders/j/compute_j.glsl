#version 430 core
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

struct AABB {
    vec3 min;
    vec3 max;
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
layout(std430, binding = 1) buffer InputJss {
    vec4 jss[];
};
/*layout(std430, binding = 2) buffer OutputJ {
    vec4 j_res[];
};*/
layout(rgba32f, binding = 0) writeonly uniform image3D jImage;


uniform sampler3D densityTexture;
uniform sampler3D densityTildeTexture;

uniform int numRBF;
uniform ivec3 densityShape;

uniform vec3 w0;

#pragma include "../sh/sh.glsl"

#pragma FDECLARE
void sh(vec3 d, out float sh[16]);
float sh_dot(float v[16], float u[16]);
#pragma FEND

vec3 aabbPos(vec3 pos, AABB box){
    vec3 newPos = (pos - box.min) / (box.max - box.min);
    newPos = vec3(newPos.z, newPos.y, newPos.x);
    newPos *= (densityShape - vec3(1.)) / densityShape.x;
    return newPos;
}

float compute_rbf(RBF rbf, vec3 x){
    float inside = length(x - rbf.c.xyz) / rbf.r;
    return rbf.w * exp(-inside*inside);
}

bool rbfTooFar(RBF rbf, vec3 x){
    return length(rbf.c.xyz - x) > 3 * rbf.r;
}

void main(){
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    uint z = gl_GlobalInvocationID.z;
    if (x >= densityShape.x || y >= densityShape.y || z >= densityShape.z)
        return;

    vec3 u = vec3(x, y, z) / densityShape.x;

    vec3 result = vec3(0);
    float y_sh[16];
    sh(normalize(-w0), y_sh);
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
        vec3 scatter = vec3(sh_dot(jss_r, y_sh), sh_dot(jss_g, y_sh), sh_dot(jss_b, y_sh));
        result += rbf_val * clamp(scatter, 0., 15.);
    }

    imageStore(jImage, ivec3(x, y, z), vec4(result, 0));
}