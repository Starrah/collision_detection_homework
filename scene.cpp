#include "scene.h"

float GRAVITY = 98.0f; // 重力加速度，单位为dm/s^2

std::vector<Ball> balls;

Vec3 worldSizeMin{-50.0f, -50.0f, -80.0f};
Vec3 worldSizeMax{50.0f, 50.0f, 150.0f};
Vec3 generationMin{-40.0f, -40.0f, -40.0f};
Vec3 generationMax{40.0f, 40.0f, 0.0f};
