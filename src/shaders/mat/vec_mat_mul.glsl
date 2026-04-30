#version 430
layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer MatA { float A[]; };
layout(std430, binding = 1) buffer VecX { float x[]; };
layout(std430, binding = 2) buffer VecY { float y[]; };

uniform int N;

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= N) return;
    
    float sum = 0.0;
    for (int col = 0; col < N; col++)
        sum += A[row * N + col] * x[col];
    
    y[row] = sum;
}