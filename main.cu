#include <iostream>
#include "core.cuh"
#include "render.h"
#include "dataGenerator.h"
#include <getopt.h>
#include <signal.h>
#include <raylib.h>

void printDeviceInfo() {
    int dev = 0;
    cudaDeviceProp devProp{};
    if (cudaGetDeviceProperties(&devProp, dev) != cudaSuccess) {
        std::cout << "FATAL: No CUDA device found!" << std::endl;
        exit(1);
    }
    std::cout << "Using GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "Capability: " << devProp.major << "." << devProp.minor << std::endl;
    std::cout << "multiProcessorCount: " << devProp.multiProcessorCount << std::endl;
    std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "maxThreadsPerMultiProcessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "warpSize: " << devProp.warpSize << std::endl;
}

const struct option long_option[] = {
        {"num",      required_argument, nullptr, 'n'},
        {"frames",   required_argument, nullptr, 'f'},
        {"gravity",  required_argument, nullptr, 'g'},
        {"headless", no_argument,       nullptr, 'h'},
        {"help",     no_argument,       nullptr, 'e'}
};

const std::string helpStr = "Usage:\r\n"
                            "-n, --num N    \t\tSpecify the number of the balls (defaults to 1000)\r\n"
                            "-h, --headless \t\tHeadless Mode (not drawing the balls)\r\n"
                            "-g, --gravity  \t\tSpecify Gravity (m/s) (defaults to 9.8)\r\n"
                            "-f, --frames F \t\tExit after F frames (defaults to unlimited)\r\n";

void SIGINTHandler(int i) {
    shouldClose = true;
}

int main(int argc, char *argv[]) {
    signal(SIGINT, SIGINTHandler);
    int opt = 0;
    int ballNum = 1000;
    bool headless = false;
    int frames = 0;
    double g = 9.8;
    while ((opt = getopt_long(argc, argv, "n:h", long_option, nullptr)) != -1) {
        switch (opt) {
            case 'e':
                std::cout << helpStr;
                exit(0);
            case 'h':
                headless = true;
                break;
            case 'n':
                ballNum = atoi(optarg);
                break;
            case 'f':
                frames = atoi(optarg);
                break;
            case 'g':
                g = atof(optarg);
                break;
        }
    }
    GRAVITY = (float) (10.0 * g);
    printDeviceInfo();
    generateData(balls, ballNum);
    initRender(headless);

    int count = 0;
    auto lastUpdate = GetTime();
    while (count++ < frames || frames == 0) {
        auto now = GetTime();
        update(balls, (float) std::min(now - lastUpdate, 1.0 / 30.0));
        lastUpdate = now;
        render(balls);
    }
    return 0;
}
