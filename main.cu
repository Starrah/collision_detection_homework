#include <iostream>
#include "core.cuh"
#include "render.h"

void printDeviceInfo() {
    int dev = 0;
    cudaDeviceProp devProp{};
    cudaSharedMemConfig sharedMemConfig;
    if (cudaGetDeviceProperties(&devProp, dev) != cudaSuccess) return;
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "Capability " << devProp.major << "." << devProp.minor << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个EM的线程束线程数：" << devProp.warpSize << std::endl;
    std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / devProp.warpSize << std::endl;
    if (cudaDeviceGetSharedMemConfig(&sharedMemConfig) != cudaSuccess) return;
    std::cout << "SharedMemConfig：" << sharedMemConfig << std::endl;
}

int main() {
    // TODO 初始化balls

    initRender();

    while (true) {
        update(balls);
        render(balls);
    }
    return 0;
}
