计算机动画的算法与技术大作业——基于GPU的碰撞检测算法
===================================
## 简介
程序基于CUDA实现，需要在NVIDIA GPU上运行。

通过GPU编写了基于均等空间划分的两阶段(Board-Phase和Narrow-Phase)的碰撞检测算法。其中，两阶段**均在GPU上运行**，因此具有很快的速度。

## Get Started
### 依赖
- CUDA Toolkit (建议>=11.1)
- CMake >= 3.18
### 构建和运行
```shell
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug .. # 此步骤需要从Github上下载依赖，请自行解决科学上网问题
cmake --build .
cmake --install . # 这步只会把shader、资源文件等复制到当前目录下，并不会安装进系统。
./collision_detection
```
### 使用预构建版本
提供Linux和Windows平台上的amd64(x64)架构下的预编译版本，见bin文件夹下的各个子目录。
### 配置运行参数
若要调整程序运行的参数，请通过命令行参数指定。通过`./collision_detection --help`可以查看程序帮助。

如果直接不带参数运行，会使用默认参数（开启渲染、1000个小球）。

## 声明
> 请勿抄袭，否则后果自负。

> 请注意，至少在2021年，最终评分的要求是所谓的需要同时实现小球的碰撞和物体的碰撞，尽管作业文档中对此的说明存在极大的歧义和不准确。本代码因为错误理解了作业要求，只实现了小球的碰撞导致直接被扣了10分。

## 特点
- 两阶段(Board-Phase和Narrow-Phase)完全在GPU上运行，不仅是GPU碰撞检测器，更是GPU物理模拟器。
- 速度很快，即使在场景中有10000个小球的情况下，物理模拟的速度在测试环境下仍可达到200FPS以上。
- 支持不同尺寸、质量、弹性系数的小球。
- 使用[Raylib](https://www.raylib.com/) 渲染引擎，小球具有光照效果。

## 代码模块构成和逻辑关系
| 文件名            | 描述                                                                                                                                                      |
| ----------------- |---------------------------------------------------------------------------------------------------------------------------------------------------------|
| core.cu           | 核心的碰撞检测及物理仿真算法。core.cuh对外只暴露一个函数`void update(std::vector<Ball> &balls, float dt)`，接受场景中的球的列表和时间的变化量，利用球的列表，在GPU上进行并行的碰撞检测，和并行的基于碰撞检测的物理仿真，以对球的运动实行物理演算。 |
| render.cpp        | 使用[Raylib](https://www.raylib.com/) 渲染引擎，对场景中的球和墙壁进行渲染。其中，渲染球使用的是光照模型，该光照模型所用shader来源于[Raylib的参考实现](https://www.raylib.com/examples.html) 。             |
| main.cu           | 程序的入口点，执行初始化、打印硬件信息、生成随机小球参数、初始化渲染器等过程，然后进入主循环。主循环中循环调用core.cu的update函数更新小球位置，然后调用renderer.cpp中的render函数渲染当前帧的场景。                                       |
| scene.cpp         | 定义了Ball等基础数据结构、场景中球的列表、场景的边界和一些物理或数学常量等，供物理模拟core.cu和渲染器render.cpp使用。                                                                                   |
| dataGenerator.cpp | 生成随机的小球参数数据，包括随机生成小球位置、速度、种类（颜色、体积、弹性系数）等。                                                                                                              |

## 性能测试
### 测试环境
- OS：Ubuntu 20.04.3 LTS (GNU/Linux 5.11.0-38-generic)
- CPU：Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz (6核12线程)
- GPU：NVIDIA GeForce GTX 1080 Ti
- 内存：Kinston 16GB DDR4 2400 (8GB\*2)
### 测试结果
| 小球数量 | 渲染 | 平均FPS | 最小FPS | 最大FPS |
| -------- | ---- | ------- | ------- | ------- |
| 10       | 否   | 1310.94  |  303  |   2416      |
| 100      | 否   |  1319.46       |  264       |  2388       |
| 1000     |  否    |   847.01      |  25       |   1406      |
| 10000    |  否    |   226.35      |  21       |    328     |
| 10       | 是   | 1184.30   |  263       |  2010       |
| 100      | 是   |  719.89      |   221      | 1061        |
| 1000     |  是    |  170.72       | 23        |  193       |
| 10000    |  是    |  23.75      |    12     |   26      |

## 参考资料
- https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda
- https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
- https://docs.nvidia.com/cuda/thrust/index.html
- https://www.raylib.com/examples.html