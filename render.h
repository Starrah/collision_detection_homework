#ifndef COLLISION_DETECTION_RENDER_H
#define COLLISION_DETECTION_RENDER_H

#include <vector>
#include "scene.h"

void initRender(bool headless = false);

void render(std::vector<Ball> &balls);

extern bool shouldClose;

void printFPS();

#endif //COLLISION_DETECTION_RENDER_H
