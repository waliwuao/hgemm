#!/bin/bash
#SBATCH --job-name=hgemm_benchmark
#SBATCH --partition=8v100-32
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=hgemm_%j.out
#SBATCH --error=hgemm_%j.err

# 环境配置
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 工作目录
WORK_DIR="/work/share/sustcsc-submit/sustcsc_11/hgemm/src"
OUTPUT_DIR="$WORK_DIR/nsys_results"
mkdir -p "$OUTPUT_DIR"

# 进入工作目录
cd "$WORK_DIR" || exit

# 编译代码
echo "开始编译 Hgemm..."
nvcc -o hgemm hgemm.cu -std=c++11 -arch=sm_70 -lcublas -lcudart
if [ $? -ne 0 ]; then
    echo "编译失败！"
    exit 1
fi
echo "编译完成！"

# 矩阵维度测试用例
TEST_CASES=(
    "768 768 768"
    "128 1024 2048"
    "128 2048 8192"
    "512 3072 1024"
    "512 4096 8192"
    "3136 576 64"
    "4096 4096 4096"
    "1024 16384 16384"
    "4096 16384 14336"
    "32768 32768 32768"
)

# Nsight Systems 配置
NSYS_ARGS="--stats=true --trace=cuda,nvtx --cuda-memory-usage=true"

# 循环执行测试
echo "开始执行 ${#TEST_CASES[@]} 个测试用例..."
for dims in "${TEST_CASES[@]}"; do
    # 解析维度参数
    read -r m n k <<< "$dims"
    
    # 构建输出文件名
    OUTPUT_FILE="${OUTPUT_DIR}/hgemm_m${m}_n${n}_k${k}"
    
    # 打印测试信息
    echo "===================================="
    echo "测试矩阵维度: M=$m, N=$n, K=$k"
    echo "输出文件: ${OUTPUT_FILE}.qdrep"
    echo "开始时间: $(date)"
    
    # 执行性能分析
    nsys profile $NSYS_ARGS -o "$OUTPUT_FILE" ./hgemm $m $n $k
    
    # 检查执行状态
    if [ $? -eq 0 ]; then
        echo "✓ 测试完成"
    else
        echo "✗ 测试失败"
    fi
    
    echo "结束时间: $(date)"
done

echo "===================================="
echo "所有测试完成！"
echo "结果保存在: $OUTPUT_DIR"
