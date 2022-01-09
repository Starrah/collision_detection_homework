#ifndef COLLISION_DETECTION_SCENE_H
#define COLLISION_DETECTION_SCENE_H

#include <vector>

extern float GRAVITY; // 重力加速度，单位为dm/s^2
#define PI 3.14159265359f
#define FPS 30

struct Vec3 {
    float x, y, z; // 单位为dm
};

struct Ball {
    Vec3 p;
    float r;
    Vec3 v; // 速度，单位为dm/s
    int color = 0;
    float elastic = 1.0f; // 弹性系数，0.0~1.0
    float density = 1.0f; // 密度，单位为kg/dm^3
};

extern std::vector<Ball> balls;

extern Vec3 worldSizeMin;
extern Vec3 worldSizeMax;
extern Vec3 generationMin;
extern Vec3 generationMax;

#endif //COLLISION_DETECTION_SCENE_H
