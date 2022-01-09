#include "render.h"
#include <raylib.h>
#include <raymath.h>

#define RLIGHTS_IMPLEMENTATION

#include "rlights.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <iomanip>

Camera3D camera{0};
std::vector<Color> idx2Color{YELLOW, GREEN, RED};
bool isHeadless;
Texture2D texFloor, texWall, texWall2;
std::vector<Model> models;
Shader shader;
float ambient[4] = {0.1f, 0.1f, 0.1f, 1.0f};

void initRender(bool headless) {
    isHeadless = headless;
    if (!headless) InitWindow(1280, 1280, "Collision Detection");
    else InitWindow(600, 120, "Collision Detection");

    camera.position = Vector3{-135.0f, -135.0f, 35.0f};
    camera.target = Vector3{0.0f, 0.0f, -50.0f};
    camera.up = Vector3{0.0f, 0.0f, 1.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    auto imageBrick = LoadImage("./assets/bricks.png");
    texWall = LoadTextureFromImage(imageBrick);
    ImageRotateCW(&imageBrick);
    texWall2 = LoadTextureFromImage(imageBrick);
    texFloor = LoadTextureFromImage(LoadImage("./assets/birch_planks.png"));

    shader = LoadShader(TextFormat("./shaders/base_lighting.vs", 330),
                        TextFormat("./shaders/lighting.fs", 330));
    shader.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(shader, "viewPos");
    int ambientLoc = GetShaderLocation(shader, "ambient");
    SetShaderValue(shader, ambientLoc, ambient, SHADER_UNIFORM_VEC4);

    for (auto &ball: balls) {
        auto model = LoadModelFromMesh(GenMeshSphere(ball.r, 16, 16));
        model.materials[0].shader = shader;
        models.push_back(model);
    }

    Light light = CreateLight(LIGHT_POINT, (Vector3) {-60.0f, -60.0f, -12.0f}, Vector3Zero(), WHITE, shader);
    UpdateLightValues(shader, light);
}

void drawWalls() {
    DrawCubeTexture(texFloor, {0.0f, 0.0f, -80.0f}, 100.0f, 100.0f, 0.01f, WHITE);
    DrawCubeTexture(texWall, {0.0f, 50.0f, -40.0f}, 100.0f, 0.01f, 80.0f, WHITE);
    DrawCubeTexture(texWall2, {50.0f, 0.0f, -40.0f}, 0.01f, 100.0f, 80.0f, WHITE);
}

std::vector<int> fpses;

void MyDrawFPS(int posX, int posY) {
    Color color = LIME; // good fps
    int fps = GetFPS();
    fpses.push_back(fps);

    if (fps < 30 && fps >= 15) color = ORANGE;  // warning FPS
    else if (fps < 15) color = RED;    // bad FPS

    DrawText(TextFormat("%2i FPS", GetFPS()), posX, posY, 100, color);
}

bool shouldClose = false;

void render(std::vector<Ball> &balls) {
    if (WindowShouldClose() || shouldClose) {
        CloseWindow();
        printFPS();
        exit(0);
        return;
    }

    if (!isHeadless) {
        UpdateCamera(&camera);
        float cameraPos[3] = {camera.position.x, camera.position.y, camera.position.z};
        SetShaderValue(shader, shader.locs[SHADER_LOC_VECTOR_VIEW], cameraPos, SHADER_UNIFORM_VEC3);
    }
    BeginDrawing();
    ClearBackground(WHITE);
    if (!isHeadless) {
        BeginMode3D(camera);
        drawWalls();
        for (auto i = 0; i < balls.size(); i++) {
            auto &ball = balls[i];
            DrawModel(models[i], {ball.p.x, ball.p.y, ball.p.z}, 1.0f, idx2Color[ball.color]);
        }
//      DrawGrid(1000, 5.0f);
        EndMode3D();
    }
    MyDrawFPS(10, 10);
    EndDrawing();
}

void printFPS() {
    if (fpses.size() <= 30) return;
    std::vector t(fpses.begin() + 15, fpses.end() - 15); // 去掉开头和结尾15帧，以减小误差
    auto max = *std::max_element(t.begin(), t.end());
    auto min = *std::min_element(t.begin(), t.end());
    unsigned long long sum = 0;
    for (auto fps: t) {
        sum += fps;
    }
    std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2) << "FPS avg: " << (double) sum / t.size()
              << ", min: " << min << ", max: " << max << std::endl;
}
