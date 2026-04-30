#include <iostream>
#include <fstream>
#include <sstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include "app.hpp"
#include "camera.hpp"
#include "shader_program.hpp"
#include "ui.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;

unsigned int frameCount = 0;
unsigned int VBO, VAO, EBO;
ShaderProgram mainProg;
ShaderProgram jssComputeProg;
ShaderProgram jComputeProg;

GLuint blueNoiseTexture = 0;
GLuint densityTexture = 0;
GLuint densityTildeTexture = 0;
GLuint depthTableTexture = 0;
GLuint aTableTexture = 0;
GLuint bTableTexture = 0;
GLuint jssBuffer = 0;
GLuint jTexture = 0;
GLuint rbfBuffer = 0;

const int BN_TEX = 0;
const int DENSITY_TEX = 1;
const int DENSITY_TILDE_TEX = 2;
const int DEPTH_TEX = 3;
const int A_TEX = 4;
const int B_TEX = 5;
const int J_TEX = 6;

float maxDensityMagnitude;
vector<float> hgShValues;
vector<float> dirShValues;
glm::ivec3 densityShape;

struct RBF {
    glm::vec4 c;
    float r;
    float w;
    glm::vec2 pad;
};
vector<RBF> rbfs;

Camera camera(0.02, 0.25);
App app = {};
UI ui;

int densityNumber = 0;

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 1;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

void loadBlueNoise(){
    int width, height, channels;
    unsigned char* data = stbi_load("src/blue_noise/LDR_RGBA_1024.png", &width, &height, &channels, 0);
    if (!data) {
        printf("Error loading blue noise texture\n");
        return;
    }

    GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
    glDeleteTextures(1, &blueNoiseTexture);
    glGenTextures(1, &blueNoiseTexture);
    glBindTexture(GL_TEXTURE_2D, blueNoiseTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
}

void loadScalarFlowDensity(){
    const int X = 100, Y = 178, Z = 100;
    densityShape = glm::ivec3(X, Y, Z);
    vector<float> density(X * Y * Z);

    string path = "src/densities/";
    ifstream f(path + "density.bin", std::ios::binary);
    f.read(reinterpret_cast<char*>(density.data()), density.size() * sizeof(float));

    glDeleteTextures(1, &densityTexture);
    glGenTextures(1, &densityTexture);
    glBindTexture(GL_TEXTURE_3D, densityTexture);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, X, Y, Z, 0, GL_RED, GL_FLOAT, density.data());

    ifstream fTilde(path + "density_tilde.bin", std::ios::binary);
    fTilde.read(reinterpret_cast<char*>(density.data()), density.size() * sizeof(float));

    glDeleteTextures(1, &densityTildeTexture);
    glGenTextures(1, &densityTildeTexture);
    glBindTexture(GL_TEXTURE_3D, densityTildeTexture);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, X, Y, Z, 0, GL_RED, GL_FLOAT, density.data());
}

void loadShData(){
    // ZONAL COEFFS
    const int nBeta = 256;
    vector<float> depthTable(nBeta * 4);
    string path = "src/preprocess/zh.bin";
    ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(depthTable.data()), depthTable.size() * sizeof(float));

    glDeleteTextures(1, &depthTableTexture);
    glGenTextures(1, &depthTableTexture);
    glBindTexture(GL_TEXTURE_1D, depthTableTexture);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, nBeta, 0, GL_RGBA, GL_FLOAT, depthTable.data());
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // EXP* DATA
    const int tableSize = 256;
    vector<float> expData(2 * tableSize + 1);
    string pathExp = "src/preprocess/exp_data.bin";
    ifstream fExp(pathExp, std::ios::binary);
    fExp.read(reinterpret_cast<char*>(expData.data()), expData.size() * sizeof(float));
    maxDensityMagnitude = expData[0];

    vector<float> aTable(expData.begin() + 1, expData.begin() + tableSize + 1);
    vector<float> bTable(expData.begin() + tableSize + 1, expData.end());

    glDeleteTextures(1, &aTableTexture);
    glGenTextures(1, &aTableTexture);
    glBindTexture(GL_TEXTURE_1D, aTableTexture);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, tableSize, 0, GL_RED, GL_FLOAT, aTable.data());
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glDeleteTextures(1, &bTableTexture);
    glGenTextures(1, &bTableTexture);
    glBindTexture(GL_TEXTURE_1D, bTableTexture);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, tableSize, 0, GL_RED, GL_FLOAT, bTable.data());
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // HG SH
    hgShValues = vector<float>(16);
    string pathHg = "src/preprocess/hg_sh.bin";
    ifstream fHg(pathHg, std::ios::binary);
    fHg.read(reinterpret_cast<char*>(hgShValues.data()), hgShValues.size() * sizeof(float));

    // DIR SH
    dirShValues = vector<float>(16);
    string pathDir = "src/preprocess/dir_sh.bin";
    ifstream fDir(pathDir, std::ios::binary);
    fDir.read(reinterpret_cast<char*>(dirShValues.data()), dirShValues.size() * sizeof(float));
}

void loadRbfs(){
    const int numRBF = 500;
    vector<float> rbfData(numRBF * 5);
    string path = "src/densities/rbfs.bin";
    ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(rbfData.data()), rbfData.size() * sizeof(float));
    rbfs.clear();
    for (int i = 0; i < numRBF; i++)
    {
        RBF rbf = {
            glm::vec4(rbfData[3 * i], rbfData[3 * i + 1], rbfData[3 * i + 2], 0),
            rbfData[3 * numRBF + i],
            rbfData[4 * numRBF + i],
            glm::vec2(0)
        };
        rbfs.push_back(rbf);
    }

    glDeleteBuffers(1, &rbfBuffer);
    glGenBuffers(1, &rbfBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, rbfBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, rbfs.size() * sizeof(RBF), rbfs.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, rbfBuffer);

    glDeleteBuffers(1, &jssBuffer);
    glGenBuffers(1, &jssBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, jssBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, rbfs.size() * 16 * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, jssBuffer);

    glDeleteTextures(1, &jTexture);
    glGenTextures(1, &jTexture);
    glBindTexture(GL_TEXTURE_3D, jTexture);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, densityShape.x, densityShape.y, densityShape.z, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindImageTexture(0, jTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
}

void initShaders(){
    mainProg.use();

    glUniform1i(glGetUniformLocation(mainProg.id(), "numRBF"), rbfs.size());
    glUniform3i(glGetUniformLocation(mainProg.id(), "densityShape"), densityShape.x, densityShape.y, densityShape.z);

    glActiveTexture(GL_TEXTURE0 + BN_TEX);
    glBindTexture(GL_TEXTURE_2D, blueNoiseTexture);
    glUniform1i(glGetUniformLocation(mainProg.id(), "blueNoise"), BN_TEX);

    glActiveTexture(GL_TEXTURE0 + DENSITY_TEX);
    glBindTexture(GL_TEXTURE_3D, densityTexture);
    glUniform1i(glGetUniformLocation(mainProg.id(), "densityTexture"), DENSITY_TEX);

    glActiveTexture(GL_TEXTURE0 + DENSITY_TILDE_TEX);
    glBindTexture(GL_TEXTURE_3D, densityTildeTexture);
    glUniform1i(glGetUniformLocation(mainProg.id(), "densityTildeTexture"), DENSITY_TILDE_TEX);

    glActiveTexture(GL_TEXTURE0 + J_TEX);
    glBindTexture(GL_TEXTURE_3D, jTexture);
    glUniform1i(glGetUniformLocation(mainProg.id(), "jTexture"), J_TEX);


    jssComputeProg.use();

    glUniform1i(glGetUniformLocation(jssComputeProg.id(), "numRBF"), rbfs.size());

    glActiveTexture(GL_TEXTURE0 + DEPTH_TEX);
    glBindTexture(GL_TEXTURE_1D, depthTableTexture);
    glUniform1i(glGetUniformLocation(jssComputeProg.id(), "depthTable"), DEPTH_TEX);

    glActiveTexture(GL_TEXTURE0 + A_TEX);
    glBindTexture(GL_TEXTURE_1D, aTableTexture);
    glUniform1i(glGetUniformLocation(jssComputeProg.id(), "aTable"), A_TEX);

    glActiveTexture(GL_TEXTURE0 + B_TEX);
    glBindTexture(GL_TEXTURE_1D, bTableTexture);
    glUniform1i(glGetUniformLocation(jssComputeProg.id(), "bTable"), B_TEX);

    glUniform1f(glGetUniformLocation(jssComputeProg.id(), "maxDensityMagnitude"), maxDensityMagnitude);

    glUniform1fv(glGetUniformLocation(jssComputeProg.id(), "hgSh"), 16, hgShValues.data());
    glUniform1fv(glGetUniformLocation(jssComputeProg.id(), "dirSh"), 16, dirShValues.data());


    jComputeProg.use();

    glUniform1i(glGetUniformLocation(jComputeProg.id(), "numRBF"), rbfs.size());
    glUniform3i(glGetUniformLocation(jComputeProg.id(), "densityShape"), densityShape.x, densityShape.y, densityShape.z);

    glActiveTexture(GL_TEXTURE0 + DENSITY_TEX);
    glBindTexture(GL_TEXTURE_3D, densityTexture);
    glUniform1i(glGetUniformLocation(jComputeProg.id(), "densityTexture"), DENSITY_TEX);

    glActiveTexture(GL_TEXTURE0 + DENSITY_TILDE_TEX);
    glBindTexture(GL_TEXTURE_3D, densityTildeTexture);
    glUniform1i(glGetUniformLocation(jComputeProg.id(), "densityTildeTexture"), DENSITY_TILDE_TEX);


}

void loadData(){
    loadBlueNoise();
    loadScalarFlowDensity();
    loadShData();
    loadRbfs();

    initShaders();
}

void init(){
    app.init(800, 600, "Smoke Simulation");
    app.toggleCursor(false);
    ui = UI(app);

    std::cerr << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cerr << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

    jComputeProg.create();
    jComputeProg.load(GL_COMPUTE_SHADER, "src/shaders/j/compute_j.glsl");
    jComputeProg.link();

    jssComputeProg.create();
    jssComputeProg.load(GL_COMPUTE_SHADER, "src/shaders/j/compute_jss.glsl");
    jssComputeProg.link();

    mainProg.create();
    mainProg.load(GL_VERTEX_SHADER, "src/shaders/main/main_vertex.glsl");
    mainProg.load(GL_FRAGMENT_SHADER, "src/shaders/main/main_frag.glsl");
    mainProg.link();

    vector<float> vertices = {
        1.f,  1.f, 0.0f,
        1.f, -1.f, 0.0f,
        -1.f, -1.f, 0.0f,
        -1.f,  1.f, 0.0f
    };
    vector<unsigned int> indices = {
        0, 1, 3,
        1, 2, 3
    };  
    tie(VBO, VAO, EBO) = ShaderProgram::addData(vertices, indices);
    ShaderProgram::linkData(3, sizeof(float), 0);

    camera.resetMousePos(app.mouseX(), app.mouseY());
    
    loadData();
}

void handleCamera(){
    if (!app.cursorIsHidden()){
        camera.hasStoppedMoving();
        camera.resetMousePos(app.mouseX(), app.mouseY());
        return;
    }

    camera.move(
        app.keyPressed(GLFW_KEY_W),
        app.keyPressed(GLFW_KEY_S),
        app.keyPressed(GLFW_KEY_D),
        app.keyPressed(GLFW_KEY_A),
        app.keyPressed(GLFW_KEY_SPACE),
        app.keyPressed(GLFW_KEY_LEFT_CONTROL),
        app.keyPressed(GLFW_KEY_LEFT_SHIFT)
    );
    camera.rotate(app.mouseX(), app.mouseY());
}

void computeJss(){
    jssComputeProg.use();

    ui.updateGPU(jssComputeProg.id());

    int n = (rbfs.size() + 63) / 64;
    jssComputeProg.dispatch(n, 1, 1);
    ShaderProgram::barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void computeJ(){
    jComputeProg.use();

    GLuint w0Loc = glGetUniformLocation(jComputeProg.id(), "w0");
    glUniform3f(w0Loc, camera.lookDir().x, camera.lookDir().y, camera.lookDir().z);

    jComputeProg.dispatch((densityShape.x + 7) / 8, (densityShape.y + 7) / 8, (densityShape.z + 7) / 8);
    ShaderProgram::barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void render(){
    mainProg.use();

    GLuint camPosLoc = glGetUniformLocation(mainProg.id(), "camera.pos");
    GLuint camLookLoc = glGetUniformLocation(mainProg.id(), "camera.lookDir");
    glUniform3f(camPosLoc, camera.position().x, camera.position().y, camera.position().z);
    glUniform3f(camLookLoc, camera.lookDir().x, camera.lookDir().y, camera.lookDir().z);

    GLuint texSizeLoc = glGetUniformLocation(mainProg.id(), "texSize");
    GLuint frameLoc = glGetUniformLocation(mainProg.id(), "frame");
    glUniform2f(texSizeLoc, app.width(), app.height());
    glUniform1ui(frameLoc, frameCount);

    ui.updateGPU(mainProg.id());

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void inputs(){
    if (app.keyPressed(GLFW_KEY_LEFT_SHIFT)){
        // Hot reload data
        if (app.keyPressedOnce(GLFW_KEY_R, frameCount)){
            loadData();
            cout << "Data reloaded." << endl;
        }
        return;
    }

    // Toggle cursor
    if (app.keyPressedOnce(GLFW_KEY_P, frameCount)){
        app.toggleCursor(app.cursorIsHidden());
        ui.toggle();
    }

    // Hot reload shaders
    if (app.keyPressedOnce(GLFW_KEY_R, frameCount)){
        mainProg.reload();
        jssComputeProg.reload();
        jComputeProg.reload();
        initShaders();
        cout << "Shaders reloaded." << endl;
    }
}

void end(){
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    mainProg.destroy();
    app.terminate();
}

int main(){
    init();
    while(!app.shouldClose())
    {
        app.startFrame(frameCount);
        handleCamera();

        ui.render();
        computeJss();
        computeJ();
        render();
        inputs();

        frameCount++;
        app.endFrame();
    }
    end();
    return EXIT_SUCCESS;
}