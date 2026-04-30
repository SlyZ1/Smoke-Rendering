void sh_exp(float v[16], out float result[16]){
    float A = exp(v[0] / PI_4_SQRT);
    v[0] = 0;
    float v_mag = sh_norm(v);
    float t = clamp(v_mag / maxDensityMagnitude, 0., 1.);
    t *= t;
    float a = texture(aTable, t).r;
    float b = texture(bTable, t).r;
    for(int i = 0; i < 16; i++) {
        result[i] = A * (a * one[i] + b * v[i]);
    }
}