#version 430 core
layout(local_size_x = 64) in;

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

uniform int numRBF;
uniform float sigma_t;
uniform float omega;

uniform sampler1D depthTable;
uniform sampler1D aTable;
uniform sampler1D bTable;
uniform float maxDensityMagnitude; 
uniform float hgSh[16];
uniform float dirSh[16];
uniform vec3 backgroundColor;

uniform float lightIntensity;
uniform vec3 lightColor;

#define PI 3.14159265
#define PI_4_SQRT 3.544907702

float one[16] = float[16](PI_4_SQRT, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

#pragma include "../sh/sh_conv.glsl"
#pragma include "../sh/sh_triple_product.glsl"
#pragma include "../sh/sh.glsl"
#pragma include "../sh/sh_exp.glsl"

#pragma FDECLARE
// SH_TRIPLE_PRODUCT.GLSL
void sh_triple_product(float f[16], float g[16], out float result[16]);

// SH_CONV.GLSL
void sh_conv(float f[16], float g[16], out float result[16]);

// SH.GLSL
void rotate_zh_to_sh(vec4 zh, vec3 d, out float sh[16]);
void sh_mul(inout float v[16], float a);
void sh_add(inout float v[16], float u[16]);
void sh_assign(out float v[16], float u[16]);

// SH_EXP.GLSL
void sh_exp(float v[16], out float result[16]);
#pragma FEND

float compute_beta(float d){
    if (d > 1) {
        return asin(1/d);
    }
    return acos(d) + PI / 2;
}

void main(){
    uint h = gl_GlobalInvocationID.x;
    if (h >= numRBF) {
        return;
    }
    
    vec3 c_h = rbfs[h].c.xyz;
    float tau[16] = float[16](0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    for(int l = 0; l < numRBF; l++) {
        vec3 c_l = rbfs[l].c.xyz;
        float r_l = rbfs[l].r;
        float w_l = rbfs[l].w;

        vec3 dir = c_l - c_h;
        float d = length(dir);
        if (d < 1e-6) continue;
        float beta = compute_beta(d);
        float t = beta / PI;
        vec4 T = texture(depthTable, t);

        float T_h[16];
        if (d < 1e-6){
            T_h = float[16](T[0],0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
        }
        else{
            rotate_zh_to_sh(T, dir / d, T_h);
        }
        sh_mul(T_h, -0.77 * r_l * w_l);
        sh_add(tau, T_h);
    }
    sh_exp(tau, tau);

    float Lin_r[16], Lin_g[16], Lin_b[16];
    for (int i = 0; i < 16; i++){
        Lin_r[i] = 0.0;
        Lin_g[i] = 0.0;
        Lin_b[i] = 0.0;
    }
    vec3 light_color = lightColor * 100 * lightIntensity;
    sh_assign(Lin_r, dirSh); sh_assign(Lin_g, dirSh); sh_assign(Lin_b, dirSh);
    sh_mul(Lin_r, light_color.r); sh_mul(Lin_g, light_color.g); sh_mul(Lin_b, light_color.b);
    Lin_r[0] += backgroundColor.r * PI_4_SQRT;
    Lin_g[0] += backgroundColor.g * PI_4_SQRT;
    Lin_b[0] += backgroundColor.b * PI_4_SQRT;

    float Res_r[16], Res_g[16], Res_b[16];
    sh_triple_product(Lin_r, tau, Res_r);
    sh_triple_product(Lin_g, tau, Res_g);
    sh_triple_product(Lin_b, tau, Res_b);
    float tmp_r[16], tmp_g[16], tmp_b[16];
    sh_conv(Res_r, hgSh, tmp_r);
    sh_conv(Res_g, hgSh, tmp_g);
    sh_conv(Res_b, hgSh, tmp_b);

    for(int i = 0; i < 16; i++) {
        vec4 result = omega * vec4(tmp_r[i], tmp_g[i], tmp_b[i], 0.);
        jss[h * 16 + i] = result;
        //imageStore(jssImage, ivec2(h, i), result);
    }
}