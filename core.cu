#include "core.cuh"
#include "utils.h"
#include "dbg.h"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

__global__ void
getArrFloat(float *result, const void *data, unsigned int count, unsigned int sizeOf, unsigned int offsetOf) {
    auto thIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thIdx < count) result[thIdx] = *((float *) ((char *) data + thIdx * sizeOf + offsetOf));
}

__global__ void multiply(float *data, unsigned int count, float op2) {
    auto thIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thIdx < count) data[thIdx] *= op2;
}

#define getArrFloat_B 512
#define generateCellID_B 256
#define generateCollisionCells_B 256
#define generateCollisionCells_U 16
#define add8Impulse_B 512
#define calculateCollisionImpulses_B 256
#define physicalUpdate_B 256

#define MAX_GRID 1000
#define CELL_BITS 10
#define XSHIFT (2 * CELL_BITS)
#define YSHIFT CELL_BITS
#define CELL_MASK ((1 << CELL_BITS) - 1)
#define HOME_TYPE_MASK ((1 << XSHIFT) | (1 << YSHIFT) | 1)
#define CONTROL_SHIFT 21
#define OBJECTID_MASK ((1 << CONTROL_SHIFT) - 1)
#define COMMON_CELL_BITS_MASK 0xff000000

__device__ float flCe(const float value, const int b) {
    return (b == 0) ? value : ((b > 0) ? ceilf(value) : floorf(value));
}

__device__ float distanceSquared(const Vec3 p1, const Vec3 p2) {
    Vec3 d{p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

__device__ float vec3Norm(const Vec3 p) {
    return p.x * p.x + p.y * p.y + p.z * p.z;
}

__device__ unsigned int getCellID(const int ix, const int iy, const int iz) {
    return ((ix & CELL_MASK) << XSHIFT) | ((iy & CELL_MASK) << YSHIFT) | ((iz & CELL_MASK));
}

__host__ __device__ unsigned int getCellType(const unsigned int cellID) {
    unsigned int cellType = cellID & HOME_TYPE_MASK;
    cellType = ((cellType >> (XSHIFT - 2)) | (cellType >> (YSHIFT - 1)) | cellType) & 0x7;
    return cellType;
}

__host__ __device__ unsigned int getHomeType(const unsigned int objectID) {
    return (objectID >> CONTROL_SHIFT) & 0x7;
}

__global__ void generateCellID(Ball *devBalls, unsigned int count, const float *cellSize, unsigned int *cellIDs,
                               unsigned int *objectIDs, unsigned int *cellCounts) {
    auto thIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thIdx >= count) return;
    Vec3 cellPos{devBalls[thIdx].p.x / *cellSize, devBalls[thIdx].p.y / *cellSize, devBalls[thIdx].p.z / *cellSize};
    float cellR = devBalls[thIdx].r / *cellSize, cellR2 = cellR * cellR;
    Vec3 cellPosFloor{floorf(cellPos.x), floorf(cellPos.y), floorf(cellPos.z)};
    Vec3 cellPosRemain{cellPos.x - cellPosFloor.x, cellPos.y - cellPosFloor.y, cellPos.z - cellPosFloor.z};
    int ix = (int) cellPosFloor.x, iy = (int) cellPosFloor.y, iz = (int) cellPosFloor.z;
    unsigned int homeCellID = getCellID(ix, iy, iz);
    unsigned int homeCellType = getCellType(homeCellID);
    unsigned int occupationBits = 1 << homeCellType;

    // home cell的设置
    cellIDs[thIdx] = homeCellID;
    int cellCount = 1;

    // 其他相邻cell的遍历
    int bx = cellPosRemain.x < 0.5f ? -1 : 0, by = cellPosRemain.y < 0.5f ? -1 : 0,
            bz = cellPosRemain.z < 0.5f ? -1 : 0;
    // 至多与8个块相交，因此通过预处理，可以只遍历8次不遍历27次。
    for (int i = bx; i <= bx + 1; i++) {
        for (int j = by; j <= by + 1; j++) {
            for (int k = bz; k <= bz + 1; k++) {
                if (i == 0 && j == 0 && k == 0) continue;
                Vec3 tangentPos{flCe(cellPos.x, i), flCe(cellPos.y, j), flCe(cellPos.z, k)}; // 与该方向的（面/棱/角）相切时，球应具有的半径
                float d2 = distanceSquared(cellPos, tangentPos);
                if (d2 < cellR2) { // 相交
                    unsigned int cellId = getCellID(ix + i, iy + j, iz + k);
                    unsigned int cellType = getCellType(cellId);
                    occupationBits |= (1 << cellType);
                    cellIDs[thIdx + count * cellCount] = cellId;
                    cellCount++;
                }
            }
        }
    }

    // 到此，所有相邻空间求交完成，cellIDs设置完成，occupationBits也设置完成。
    cellCounts[thIdx] = cellCount;
    // objectId格式:8位占用mask，然后三位home类型，然后21位objectId
    unsigned int controlBits = (occupationBits << 3) | homeCellType;
    unsigned int objectID = (controlBits << CONTROL_SHIFT) | thIdx;
    for (int i = 0; i < cellCount; i++) {
        objectIDs[thIdx + count * i] = objectID;
    }
}

struct CollisionCell {
    unsigned int cellID;
    unsigned int offset;
    unsigned int home;
    unsigned int total;
};

__global__ void
generateCollisionCells(CollisionCell *result, unsigned int *findCounts, const unsigned int *cellIDs,
                       const unsigned int *objectIDs, unsigned int count, unsigned int U) {
    auto thIdx = blockIdx.x * blockDim.x + threadIdx.x;
    auto loopEnd = min((thIdx + 1) * U, count);
    auto findCount = 0;
    auto start = -1;
    bool isPhantom = true;
    unsigned int phantomStart;
    for (auto i = thIdx * U;; i++) {
        if (i >= count || cellIDs[i] != cellIDs[(int) i - 1]) { // 否则i=0时i-1会变成很大的数而不是-1，导致开场直接SIGSEGV
            if (start >= 0) {
                if (!isPhantom) phantomStart = i;
                auto home = phantomStart - start;
                auto total = i - start;
                if (home >= 1 && total >= 2) { // 只保留长度大于等于2的线段，太短的不要
                    result[thIdx * U + findCount] = {cellIDs[i - 1], (unsigned int) (start), home, total};
                    findCount++;
                }
            }
            start = (int) i;
            isPhantom = false;
            if (i >= loopEnd) break;
        }
        if (!isPhantom && getCellType(cellIDs[i]) != getHomeType(objectIDs[i])) {
            phantomStart = i;
            isPhantom = true;
        }
    }
    findCounts[thIdx] = findCount;
}

__global__ void pickByOffset(CollisionCell *result, const CollisionCell *data, unsigned int U,
                             const unsigned int *findCounts, const unsigned int *offset) {
    auto thIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (auto i = 0; i < findCounts[thIdx]; i++) {
        result[offset[thIdx] + i] = data[thIdx * U + i];
    }
}

__device__ void vec3AddTo(Vec3 *a, const Vec3 *b) {
    a->x += b->x;
    a->y += b->y;
    a->z += b->z;
}

__device__ Vec3 vec3Mul(const Vec3 a, float f) {
    return {a.x * f, a.y * f, a.z * f};
}

__device__ Vec3 vec3Sub(const Vec3 a, const Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ Vec3 vec3Normalized(const Vec3 p) {
    auto n = vec3Norm(p);
    return {p.x / n, p.y / n, p.z / n};
}

__device__ float vec3Dot(const Vec3 a, const Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vec3 vec3SubThenMul(const Vec3 a, const Vec3 b, float f) {
    return {(a.x - b.x) * f, (a.y - b.y) * f, (a.z - b.z) * f};
}

__device__ float getMass(const Ball ball) {
    return ball.r * ball.r * ball.r * 3 / 4 * ball.density;
}

__global__ void add8Impulse(Vec3 *impulses, unsigned int count) {
    auto thIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thIdx >= count) return;
    for (auto i = 1; i < 8; i++) {
        vec3AddTo(&impulses[thIdx], &impulses[thIdx + i * count]);
    }
}

__global__ void calculateCollisionImpulses(Vec3 *impulses, const CollisionCell *collisionCells, unsigned int count,
                                           const unsigned int *objectIDs, const Ball *devBalls,
                                           unsigned int ballCount) {
    auto thIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thIdx >= count) return;
    auto collisionCell = collisionCells[thIdx];
    auto cellType = getCellType(collisionCell.cellID);
    for (auto i = collisionCell.offset; i < collisionCell.offset + collisionCell.home; i++) {
        for (auto j = i + 1; j < collisionCell.offset + collisionCell.total; j++) {
            auto commonCellBits = (objectIDs[i] & objectIDs[j] & COMMON_CELL_BITS_MASK) >> (CONTROL_SHIFT + 3);
            auto homeType1 = getHomeType(objectIDs[i]), homeType2 = getHomeType(objectIDs[j]);
            if ((homeType1 < cellType && (((1 << homeType1) & commonCellBits) != 0)) ||
                (homeType2 < cellType && (((1 << homeType2) & commonCellBits) != 0)))
                continue;
            auto id1 = objectIDs[i] & OBJECTID_MASK, id2 = objectIDs[j] & OBJECTID_MASK;
            auto sumR = devBalls[id1].r + devBalls[id2].r;
            if (distanceSquared(devBalls[id1].p, devBalls[id2].p) <= sumR * sumR) { // 撞了
                auto m1 = getMass(devBalls[id1]), m2 = getMass(devBalls[id2]);
                auto e = (devBalls[id1].elastic + devBalls[id2].elastic) / 2, ea1dm1am2 = (e + 1) / (m1 + m2);
                auto centerLine = vec3Normalized(vec3Sub(devBalls[id2].p, devBalls[id1].p)); // 从球1球心指向球2球心的单位向量
                auto vCLLen1 = vec3Dot(devBalls[id1].v, centerLine), vCLLen2 = vec3Dot(devBalls[id1].v,
                                                                                       centerLine); // 沿球心连线方向的速度分量
                auto dvCLLen1 = (vCLLen2 - vCLLen1) * (ea1dm1am2 * m2), dvCLLen2 =
                        (vCLLen1 - vCLLen2) * (ea1dm1am2 * m1);
                auto dv1 = vec3Mul(centerLine, dvCLLen1), dv2 = vec3Mul(centerLine, dvCLLen2);
                vec3AddTo(&impulses[cellType * ballCount + id1], &dv1);
                vec3AddTo(&impulses[cellType * ballCount + id2], &dv2);
            }
        }
    }
}

#define IMPULSE_MAX 200

__device__ void clamp(float *value, float minVal, float maxVal) {
    if (*value < minVal) *value = minVal;
    else if (*value > maxVal) *value = maxVal;
}

__device__ void clamp(Vec3 *value, float minVal, float maxVal) {
    clamp(&value->x, minVal, maxVal);
    clamp(&value->y, minVal, maxVal);
    clamp(&value->z, minVal, maxVal);
}

__global__ void
physicalUpdate(Ball *devBalls, unsigned int count, Vec3 *impulses, float dt, std::pair<Vec3, Vec3> worldSize,
               float GRAVITY) {
    // 应用冲量之和
    auto thIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thIdx >= count) return;
    // clamp冲量数值，以防鬼畜
    clamp(&impulses[thIdx], -IMPULSE_MAX, IMPULSE_MAX);
    vec3AddTo(&devBalls[thIdx].v, &impulses[thIdx]);
    // 应用重力加速度
    Vec3 gravityImpulseDted = vec3Mul({0.0f, 0.0f, -GRAVITY}, dt);;
    vec3AddTo(&devBalls[thIdx].v, &gravityImpulseDted);
    // 计算与墙壁碰撞、反转速度
    if ((devBalls[thIdx].p.x - devBalls[thIdx].r <= worldSize.first.x && devBalls[thIdx].v.x < 0) ||
        (devBalls[thIdx].p.x + devBalls[thIdx].r >= worldSize.second.x && devBalls[thIdx].v.x > 0)) {
        // v'=v-(1+e)v=-ev,e=(e_wall+e_ball)/2,e_wall=1
        devBalls[thIdx].v.x = -(0.5f + devBalls[thIdx].elastic) / 2.0f * devBalls[thIdx].v.x;
    }
    if ((devBalls[thIdx].p.y - devBalls[thIdx].r <= worldSize.first.y && devBalls[thIdx].v.y < 0) ||
        (devBalls[thIdx].p.y + devBalls[thIdx].r >= worldSize.second.y && devBalls[thIdx].v.y > 0)) {
        // v'=v-(1+e)v=-ev,e=(e_wall+e_ball)/2,e_wall=1
        devBalls[thIdx].v.y = -(0.5f + devBalls[thIdx].elastic) / 2.0f * devBalls[thIdx].v.y;
    }
    if ((devBalls[thIdx].p.z - devBalls[thIdx].r <= worldSize.first.z && devBalls[thIdx].v.z < 0) ||
        (devBalls[thIdx].p.z + devBalls[thIdx].r >= worldSize.second.z && devBalls[thIdx].v.z > 0)) {
        // v'=v-(1+e)v=-ev,e=(e_wall+e_ball)/2,e_wall=0.5
        devBalls[thIdx].v.z = -(0.5f + devBalls[thIdx].elastic) / 2.0f * devBalls[thIdx].v.z;
        if (fabsf(devBalls[thIdx].v.z) <= 7.0f) devBalls[thIdx].v.z = 0;
    }
    // 进行delta时间以调整位置
    Vec3 movementDted = vec3Mul(devBalls[thIdx].v, dt);
    vec3AddTo(&devBalls[thIdx].p, &movementDted);
}

void update(std::vector<Ball> &balls, float dt) {
    auto ballNum = balls.size();
    // balls数据复制到显存上
    Ball *devBalls;
    cudaMalloc(&devBalls, ballNum * sizeof(Ball));
    cudaMemcpy(devBalls, balls.data(), ballNum * sizeof(Ball), cudaMemcpyHostToDevice);

    // 求cellSize
    float *rList, *devCellSize; // 半径列表
    cudaMalloc(&rList, (ballNum + 1) * sizeof(float));
    cudaMalloc(&devCellSize, sizeof(float));
    {
        dim3 block(getArrFloat_B), grid(DIVIDE_CEIL(ballNum, block.x));
        getArrFloat<<<grid, block>>>(rList, devBalls, ballNum, sizeof(Ball), offsetof(Ball, r));
    }
    // 把最小的网格大小要求也放到数组里一起求max
    float maxWorldSize = std::max(std::max(worldSizeMax.x - worldSizeMin.x, worldSizeMax.y - worldSizeMin.y),
                                  worldSizeMax.z - worldSizeMin.z);
    float minGridSize = maxWorldSize / MAX_GRID / 3.0f;
    cudaMemcpy(rList + ballNum, &minGridSize, sizeof(float), cudaMemcpyHostToDevice);
    devCellSize = thrust::raw_pointer_cast(
            thrust::max_element(thrust::device_pointer_cast(rList), thrust::device_pointer_cast(rList + ballNum + 1)));
    multiply<<<dim3(1), dim3(1)>>>(devCellSize, 1, 3.0f);
    cudaFree(rList);

    // 生成cellID和objectID数组
    unsigned int *cellIDs_raw, *cellIDs, *objectIDs, *cellCounts;
    cudaMalloc(&cellIDs_raw, (ballNum * 8 + 2) * sizeof(unsigned int));
    cudaMemset(cellIDs_raw, 0xff, (ballNum * 8 + 2) * sizeof(unsigned int));
    cellIDs = cellIDs_raw + 2;
    cudaMalloc(&objectIDs, ballNum * 8 * sizeof(unsigned int));
    cudaMalloc(&cellCounts, ballNum * sizeof(unsigned int));
    {
        dim3 block(generateCellID_B), grid(DIVIDE_CEIL(ballNum, block.x));
        generateCellID<<<grid, block>>>(devBalls, ballNum, devCellSize, cellIDs, objectIDs, cellCounts);
    }
    thrust::inclusive_scan(thrust::device_pointer_cast(cellCounts), thrust::device_pointer_cast(cellCounts + ballNum),
                           thrust::device_pointer_cast(cellCounts));
    unsigned int *dev_totalCellCount = cellCounts + ballNum - 1, totalCellCount;
    cudaMemcpy(&totalCellCount, dev_totalCellCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    thrust::stable_sort_by_key(thrust::device_pointer_cast(cellIDs),
                               thrust::device_pointer_cast(cellIDs + ballNum * 8),
                               thrust::device_pointer_cast(objectIDs));

    CollisionCell *buf, *collisionCells;
    unsigned int *findCounts, *findCountsSummed;
    cudaMalloc(&buf, totalCellCount * sizeof(CollisionCell));
    cudaMalloc(&collisionCells, totalCellCount * sizeof(CollisionCell));
    unsigned int ths = DIVIDE_CEIL(totalCellCount, generateCollisionCells_U);
    unsigned int B = generateCollisionCells_B, G = DIVIDE_CEIL(ths, B);
    cudaMalloc(&findCounts, G * B * sizeof(unsigned int));
    cudaMalloc(&findCountsSummed, G * B * sizeof(unsigned int));

    {
        generateCollisionCells<<<dim3(G), dim3(B)>>>(buf, findCounts, cellIDs, objectIDs, totalCellCount,
                                                     generateCollisionCells_U);
    }

    thrust::exclusive_scan(thrust::device_pointer_cast(findCounts), thrust::device_pointer_cast(findCounts + G * B),
                           thrust::device_pointer_cast(findCountsSummed));

    unsigned int collisionCellCount, temp;
    cudaMemcpy(&collisionCellCount, findCountsSummed + G * B - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp, findCounts + G * B - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    collisionCellCount += temp;

    {
        pickByOffset<<<dim3(G), dim3(B)>>>(collisionCells, buf, generateCollisionCells_U, findCounts, findCountsSummed);
    }

    cudaFree(findCountsSummed);
    cudaFree(findCounts);
    cudaFree(buf);
    // 到这里，collisionCells存的是所有需要进行碰撞检测的cell

    Vec3 *impulses;
    cudaMalloc(&impulses, ballNum * 8 * sizeof(Vec3));
    cudaMemset(impulses, 0x00, ballNum * 8 * sizeof(Vec3)); // 因为IEEE 754的0.0f就是0x00000000

    {
        dim3 block(calculateCollisionImpulses_B), grid(DIVIDE_CEIL(collisionCellCount, block.x));
        auto a = ddu((unsigned int *) collisionCells, 4 * collisionCellCount);
        auto b = ddu(objectIDs, totalCellCount);
        calculateCollisionImpulses<<<grid, block>>>(impulses, collisionCells, collisionCellCount, objectIDs, devBalls,
                                                    ballNum);
    }

    {
        dim3 block(add8Impulse_B), grid(DIVIDE_CEIL(ballNum, block.x));
        add8Impulse<<<grid, block>>>(impulses, ballNum);
    }

    {
        dim3 block(physicalUpdate_B), grid(DIVIDE_CEIL(ballNum, block.x));
        physicalUpdate<<<grid, block>>>(devBalls, ballNum, impulses, dt, std::make_pair(worldSizeMin, worldSizeMax),
                                        GRAVITY);
    }

    cudaFree(impulses);
    cudaFree(collisionCells);
    cudaFree(cellIDs_raw);
    cudaFree(objectIDs);
    cudaFree(cellCounts);
    cudaFree(devCellSize);

    auto errcode4 = cudaDeviceSynchronize();
    if (errcode4 != cudaSuccess) {
        exit(1);
    }

    cudaMemcpy(balls.data(), devBalls, ballNum * sizeof(Ball), cudaMemcpyDeviceToHost);

    cudaFree(devBalls);
}

