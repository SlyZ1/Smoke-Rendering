void rotate_zh_to_sh(vec4 zh, vec3 d, out float sh[16]) {
    float x = d.x, y = d.y, z = d.z;

    float xx = x*x;
    float yy = y*y;
    float zz = z*z;

    sh[0] = zh[0];

    sh[1] = zh[1] * y;
    sh[2] = zh[1] * z;
    sh[3] = zh[1] * x;

    sh[4]  = zh[2] * 1.092548 * x * y;
    sh[5]  = zh[2] * 1.092548 * y * z;
    sh[6]  = zh[2] * 0.315392 * (3.0*zz - 1.0);
    sh[7]  = zh[2] * 1.092548 * x * z;
    sh[8]  = zh[2] * 0.546274 * (xx - yy);

    sh[9]  = zh[3] * 0.590044 * y * (3.0*xx - yy);
    sh[10] = zh[3] * 2.890611 * x * y * z;
    sh[11] = zh[3] * 0.457046 * y * (4.0*zz - xx - yy);
    sh[12] = zh[3] * 0.373176 * z * (2.0*zz - 3.0*xx - 3.0*yy);
    sh[13] = zh[3] * 0.457046 * x * (4.0*zz - xx - yy);
    sh[14] = zh[3] * 1.445306 * z * (xx - yy);
    sh[15] = zh[3] * 0.590044 * x * (xx - 3.0*yy);
}

void sh(vec3 d, out float sh[16])
{
    float x = d.x;
    float y = d.y;
    float z = d.z;

    float xx = x*x;
    float yy = y*y;
    float zz = z*z;

    sh[0] = 0.282095;

    sh[1] = 0.488603 * y;
    sh[2] = 0.488603 * z;
    sh[3] = 0.488603 * x;

    sh[4] = 1.092548 * x * y;
    sh[5] = 1.092548 * y * z;
    sh[6] = 0.315392 * (3.0*zz - 1.0);
    sh[7] = 1.092548 * x * z;
    sh[8] = 0.546274 * (xx - yy);

    sh[9]  = 0.590044 * y * (3.0*xx - yy);
    sh[10] = 2.890611 * x * y * z;
    sh[11] = 0.457046 * y * (5.0*zz - 1.0);
    sh[12] = 0.373176 * z * (5.0*zz - 3.0);
    sh[13] = 0.457046 * x * (5.0*zz - 1.0);
    sh[14] = 1.445306 * (xx - yy) * z;
    sh[15] = 0.590044 * x * (xx - 3.0*yy);
}

void sh_mul(inout float v[16], float a){
    for(int i = 0; i < 16; i++) {
        v[i] *= a;
    }
}

void sh_add(inout float v[16], float u[16]){
    for(int i = 0; i < 16; i++) {
        v[i] += u[i];
    }
}

float sh_norm(float v[16]){
    float result = 0;
    for(int i = 0; i < 16; i+=1){
        result += pow(v[i], 2);
    }
    return sqrt(result);
}

float sh_dot(float v[16], float u[16]){
    float result = 0;
    for(int i = 0; i < 16; i+=1){
        result += v[i] * u[i];
    }
    return result;
}

void sh_assign(out float v[16], float u[16]){
    for(int i = 0; i < 16; i++) {
        v[i] = u[i];
    }
}