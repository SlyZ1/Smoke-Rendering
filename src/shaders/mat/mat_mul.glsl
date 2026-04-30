#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer MatA { float A[]; };
layout(std430, binding = 1) buffer MatB { float B[]; };
layout(std430, binding = 2) buffer MatC { float C[]; };

uniform int N;

shared float tileA[16][16];
shared float tileB[16][16];

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    float sum = 0.0;
    
    for (int t = 0; t < (N + 15) / 16; t++) {
        tileA[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
            A[row * N + t * 16 + gl_LocalInvocationID.x];
        tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = 
            B[(t * 16 + gl_LocalInvocationID.y) * N + col];
        
        barrier();
        
        for (int k = 0; k < 16; k++)
            sum += tileA[gl_LocalInvocationID.y][k] * tileB[k][gl_LocalInvocationID.x];
        
        barrier();
    }
    
    if (row < N && col < N)
        C[row * N + col] = sum;
}