#ifndef COLLISION_DETECTION_DBG_H
#define COLLISION_DETECTION_DBG_H

#include "scene.h"

typedef float f1000[1000];
typedef f1000 *pf1000;

pf1000 ddf(float *data, unsigned int len);

typedef unsigned int u1000[1000];
typedef u1000 *pu1000;

pu1000 ddu(unsigned int *data, unsigned int len);

float* dthf(float *data, unsigned int len);

unsigned int* dthu(unsigned int *data, unsigned int len);

inline void dbgData() {
//    GRAVITY = 0.0f;
    balls.clear();
    std::vector<Ball> data{
            {{-20.0f, 0.0f, -40.0f}, 2.0f, {10.0f,  0.0f, 0.0f}},
            {{20.0f,  0.0f, -40.0f}, 2.0f, {-10.0f, 0.0f, 0.0f}},
    };
    balls = data;
}

#endif //COLLISION_DETECTION_DBG_H
