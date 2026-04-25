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
ShaderProgram shaderProg;

GLuint blueNoiseTexture;
GLuint scalarFlowDensityTexture;
GLuint aTableTexture;
GLuint bTableTexture;
float maxDensityMagnitude;
glm::vec4 zhCoeffs;

Camera camera(0.02, 0.25);
App app = {};
UI ui;

int densityNumber = 0;

void loadBlueNoise(){
    int width, height, channels;
    unsigned char* data = stbi_load("src/blue_noise/LDR_RGBA_1024.png", &width, &height, &channels, 0);
    if (!data) {
        printf("Error loading blue noise texture\n");
        return;
    }

    GLenum format = (channels == 4) ? GL_RGBA : GL_RGB;
    glGenTextures(1, &blueNoiseTexture);
    glBindTexture(GL_TEXTURE_2D, blueNoiseTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);
}

void loadScalarFlowDensity(const string& densityName){
    const int X = 100, Y = 178, Z = 100;
    vector<float> density(X * Y * Z);
    string path = "src/densities/";
    ifstream f(path + densityName, std::ios::binary);
    f.read(reinterpret_cast<char*>(density.data()), density.size() * sizeof(float));

    glGenTextures(1, &scalarFlowDensityTexture);
    glBindTexture(GL_TEXTURE_3D, scalarFlowDensityTexture);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, X, Y, Z, 0, GL_RED, GL_FLOAT, density.data());
}

void loadShData(){
    // ZONAL COEFFS
    string path = "src/preprocess/zh.bin";
    ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(&zhCoeffs), sizeof(glm::vec4));

    // EPX* DATA
    const int tableSize = 256;
    vector<float> expData(2*tableSize + 1);
    string path_exp = "src/preprocess/exp_data.bin";
    ifstream f_exp(path_exp, std::ios::binary);
    f_exp.read(reinterpret_cast<char*>(expData.data()), expData.size() * sizeof(float));
    maxDensityMagnitude = expData[0];

    vector<float> aTable(expData.begin() + 1, expData.begin() + tableSize + 1);
    vector<float> bTable(expData.begin() + tableSize + 1, expData.end());

    glGenTextures(1, &aTableTexture);
    glBindTexture(GL_TEXTURE_1D, aTableTexture);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, tableSize, 0, GL_RED, GL_FLOAT, aTable.data());
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &bTableTexture);
    glBindTexture(GL_TEXTURE_1D, bTableTexture);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, tableSize, 0, GL_RED, GL_FLOAT, bTable.data());
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void init(){
    app.init(800, 600, "Smoke Simulation");
    app.toggleCursor(false);
    ui = UI(app);

    shaderProg.create();
    shaderProg.load(GL_VERTEX_SHADER, "src/shaders/mainVertex.glsl");
    shaderProg.load(GL_FRAGMENT_SHADER, "src/shaders/mainFrag.glsl");
    shaderProg.link();

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
    
    loadBlueNoise();
    loadScalarFlowDensity("density.bin");
    loadShData();
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

void render(){
    shaderProg.use();

    GLuint camPosLoc = glGetUniformLocation(shaderProg.id(), "camera.pos");
    GLuint camLookLoc = glGetUniformLocation(shaderProg.id(), "camera.lookDir");
    glUniform3f(camPosLoc, camera.position().x, camera.position().y, camera.position().z);
    glUniform3f(camLookLoc, camera.lookDir().x, camera.lookDir().y, camera.lookDir().z);

    GLuint texSizeLoc = glGetUniformLocation(shaderProg.id(), "texSize");
    GLuint frameLoc = glGetUniformLocation(shaderProg.id(), "frame");
    glUniform2f(texSizeLoc, app.width(), app.height());
    glUniform1ui(frameLoc, frameCount);

    GLuint blueNoiseLoc = glGetUniformLocation(shaderProg.id(), "blueNoise");
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, blueNoiseTexture);
    glUniform1i(blueNoiseLoc, 0);

    GLuint densityTextureLoc = glGetUniformLocation(shaderProg.id(), "densityTexture");
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, scalarFlowDensityTexture);
    glUniform1i(densityTextureLoc, 1);

    GLuint aTableTextureLoc = glGetUniformLocation(shaderProg.id(), "aTable");
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_1D, aTableTexture);
    glUniform1i(aTableTextureLoc, 2);

    GLuint bTableTextureLoc = glGetUniformLocation(shaderProg.id(), "bTable");
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_1D, bTableTexture);
    glUniform1i(bTableTextureLoc, 3);

    GLuint maxDensityMagLoc = glGetUniformLocation(shaderProg.id(), "maxDensityMagnitude");
    glUniform1f(maxDensityMagLoc, maxDensityMagnitude);

    GLuint zhCoeffsLoc = glGetUniformLocation(shaderProg.id(), "zhCoeffs");
    glUniform4f(zhCoeffsLoc, zhCoeffs[0], zhCoeffs[1], zhCoeffs[2], zhCoeffs[3]);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void inputs(){
    // Toggle cursor
    if (app.keyPressedOnce(GLFW_KEY_P, frameCount)){
        app.toggleCursor(app.cursorIsHidden());
        ui.toggle();
    }

    // Hot reload shaders
    if (app.keyPressedOnce(GLFW_KEY_R, frameCount)){
        shaderProg.reload();
        cout << "Shaders reloaded" << endl;
    }

    if (app.keyPressedOnce(GLFW_KEY_TAB, frameCount)){
        switch (densityNumber)
        {
        case 0:
            loadScalarFlowDensity("density_tilde.bin");
            break;
        case 1:
            loadScalarFlowDensity("density.bin");
            break;
        }
        densityNumber = (densityNumber + 1) % 2;
    }
}

void end(){
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    shaderProg.destroy();
    app.terminate();
}

int main(){
    init();
    while(!app.shouldClose())
    {
        app.startFrame(frameCount);
        handleCamera();

        ui.render();
        ui.updateGPU(shaderProg.id());
        
        render();
        inputs();

        frameCount++;
        app.endFrame();
    }
    end();
    return EXIT_SUCCESS;
}