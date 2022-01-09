#include "dataGenerator.h"
#include <random>

struct BallType {
    int portion;
    int color;
    float r;
};

std::vector<BallType> ballTypes{
        {6, 0, 2.0f},
        {3, 1, 3.0f},
        {1, 2, 5.0f}
};

void generateData(std::vector<Ball> &balls, int count) {
    std::vector<int> accPortion;
    int t = 0;
    for (auto ballType: ballTypes) {
        t += ballType.portion;
        accPortion.push_back(t);
    }

    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_int_distribution<int> randPortion(0, accPortion.back() - 1);
    std::uniform_real_distribution<float> randX(generationMin.x, generationMax.x);
    std::uniform_real_distribution<float> randY(generationMin.y, generationMax.y);
    std::uniform_real_distribution<float> randZ(generationMin.z, generationMax.z);
    std::normal_distribution<float> randVx(0.0f, 10.0f);
    std::normal_distribution<float> randVy(0.0f, 10.0f);
    std::normal_distribution<float> randVz(0.0f, 20.0f);
    std::uniform_real_distribution<float> randElastic(0.0f, 2.0f);

    for (int i = 0; i < count; ++i) {
        int ballTypeIdx, r = randPortion(e);
        for (ballTypeIdx = 0; ballTypeIdx < ballTypes.size(); ++ballTypeIdx) {
            if (r < accPortion[ballTypeIdx]) break;
        }
        auto &ballType = ballTypes[ballTypeIdx];

        Ball ball{{randX(e), randY(e), randZ(e)}, ballType.r, {randVx(e), randVy(e), randVz(e)}, ballType.color,
                  std::min(randElastic(e), 1.0f)};
        balls.push_back(ball);
    }
}
