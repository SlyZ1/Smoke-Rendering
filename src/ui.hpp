#include "app.hpp"
#include <glm/glm.hpp>

using namespace glm;

class UI {
    private:
        bool m_show = false;
        App m_app = {};

        // App

        // Render
        vec3 m_lightColor = vec3(1.,1.,1.);
        float m_lightIntensity = 1.;
        float m_sigma_t = 30;
        float m_omega = 1.6;
        float m_stepSize = 0.05;
        bool m_useNoise = true;
        vec3 backgroundColor = vec3(0.655, 0.780, 0.906);

        // Simulation

        void renderApp();
        void renderRender();
        void renderSimulation();
        void Label(const char* label);
        void BeginTwoColumnLayout();
        void EndTwoColumnLayout();

    public:
        UI() = default;
        UI(App app) : m_app(app) { 
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(300, m_app.getIo()->DisplaySize.y));
        };

        void toggle();
        void render();
        void updateGPU(GLuint shaderProgramId);
};