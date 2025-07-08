# sustcsc_11

## 项目简介
该文件夹包含基于 CUDA 的通用矩阵乘法 (GEMM) 实现，利用 NVIDIA Tensor Core 的 Warp Matrix Multiply-Accumulate (WMMA) 技术进行加速。代码旨在展示如何使用 CUDA 编程和 WMMA API 来优化矩阵计算性能。

## 文件结构
- **`hgemm_buffer.cu`**: 使用 WMMA 技术实现的矩阵乘法核心代码，支持双缓冲优化。
- **`hgemm.cu`**: 包含 WMMA 核心计算逻辑的实现。
- **`README.md`**: 当前文件，介绍项目内容和运行方法。

## 主要功能
- Warp 级别矩阵乘加速。
- 双缓冲优化以提高数据传输效率。
- 支持 CUDA 统一内存管理。

## 运行方法
1. 确保安装了 CUDA 环境。
2. 编译代码：
    ```bash
    nvcc -o hgemm hgemm.cu -std=c++11 -arch=sm_70 -lcublas -lcudart
    ```
    请根据您的 GPU 架构调整 `sm_70`。
3. 运行程序：
    ```bash
    ./hgemm M N K
    nsys profile --stats=true --trace=cuda,nvtx --cuda-memory-usage=true ./hgemm 8192 8192 8192
    ```

## 注意事项
- 确保您的 GPU 支持 Tensor Core（如 NVIDIA Volta 或更高版本架构）。
- 代码仅用于学习和研究目的，可能不是最优实现。

## 参考
- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [WMMA API 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
