void sh_order3(vec3 d, out float sh[16])
{
    float x = d.x;
    float y = d.y;
    float z = d.z;

    // l = 0
    sh[0] = 0.282095;

    // l = 1
    sh[1] = 0.488603 * y;
    sh[2] = 0.488603 * z;
    sh[3] = 0.488603 * x;

    // l = 2
    sh[4] = 1.092548 * x * y;
    sh[5] = 1.092548 * y * z;
    sh[6] = 0.315392 * (3.0*z*z - 1.0);
    sh[7] = 1.092548 * x * z;
    sh[8] = 0.546274 * (x*x - y*y);

    // l = 3
    sh[9]  = 0.590044 * y * (3.0*x*x - y*y);
    sh[10] = 2.890611 * x * y * z;
    sh[11] = 0.457046 * y * (5.0*z*z - 1.0);
    sh[12] = 0.373176 * z * (5.0*z*z - 3.0);
    sh[13] = 0.457046 * x * (5.0*z*z - 1.0);
    sh[14] = 1.445306 * (x*x - y*y) * z;
    sh[15] = 0.590044 * x * (x*x - 3.0*y*y);
}

void sh_mul6(inout float v[16], float a, out float result[16]){
    for(int i = 0; i < 16; i++) {
        result[i] = v[i] * a;
    }
}

void sh_add6(float v[16], float u[16], out float result[16]){
    for(int i = 0; i < 16; i++) {
        result[i] = v[i] + u[i];
    }
}

float sh_norm3(float v[16]){
    float result = 0;
    for(int i = 0; i < 16; i+=1){
        result += pow(v[i], 2);
    }
    return sqrt(result);
}

float sh_dot3(float v[16], float u[16]){
    float result = 0;
    for(int i = 0; i < 16; i+=1){
        result += v[i] * u[i];
    }
    return result;
}

void sh_exp3(float v[16], out float result[16]){
    float A = exp(v[0] / PI_4_SQRT);
    v[0] = 0;
    float v_mag = sh_norm3(v);
    float t = clamp(v_mag / maxDensityMagnitude, 0., 1.);
    t *= t;
    float a = texture(aTable, t).r;
    float b = texture(bTable, t).r;
    for(int i = 0; i < 16; i++) {
        result[i] = A * (a * one[i] + b * v[i]);
    }
}