#!/bin/bash
# 在 GPU 机器上运行此脚本安装 Seed-VC 本地环境
# 要求: NVIDIA GPU (P40/V100), CUDA 11.8+, Python 3.10+

set -e

echo "=== 安装 Seed-VC 本地环境 ==="

# 克隆 Seed-VC
if [ ! -d "Seed-VC" ]; then
    git clone https://github.com/Plachta/Seed-VC.git
    cd Seed-VC
else
    cd Seed-VC
    git pull
fi

# 安装依赖
pip install -r requirements.txt


echo ""
echo "=== 安装完成 ==="
echo "模型会在首次运行时自动下载到 ~/.cache/huggingface/"
echo "请确保设置了 HF_TOKEN 环境变量"
