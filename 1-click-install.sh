#!/bin/bash

echo "===== CosyVoice 一键安装脚本 ====="
echo "当前日期: 2025年04月03日"
echo "该脚本将安装 CosyVoice 及其所有依赖项"
echo "=============================="

# 检查操作系统
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "错误: 此脚本目前只支持 macOS。"
    exit 1
fi

# 检查并安装 Homebrew
if ! command -v brew &> /dev/null; then
    echo "正在安装 Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # 添加 Homebrew 到 PATH
    if [[ -f ~/.zshrc ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -f ~/.bash_profile ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "Homebrew 已安装，继续执行..."
fi

# 检查并安装 Git
if ! command -v git &> /dev/null; then
    echo "正在安装 Git..."
    brew install git
else
    echo "Git 已安装，继续执行..."
fi

# 检查并安装 Git LFS
if ! command -v git-lfs &> /dev/null; then
    echo "正在安装 Git LFS..."
    brew install git-lfs
    git lfs install
else
    echo "Git LFS 已安装，继续执行..."
fi

# 检查并安装 Miniconda
if ! command -v conda &> /dev/null; then
    echo "正在安装 Miniconda..."
    brew install --cask miniconda
    
    # 初始化 conda
    if [[ -f ~/.zshrc ]]; then
        conda init zsh
        source ~/.zshrc
    elif [[ -f ~/.bash_profile ]]; then
        conda init bash
        source ~/.bash_profile
    fi
else
    echo "Conda 已安装，继续执行..."
fi

# 创建工作目录
WORK_DIR="$HOME/CosyVoice_Project"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
echo "工作目录: $WORK_DIR"

# 克隆主仓库
echo "正在克隆 CosyVoice 仓库..."
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# 创建并激活 conda 环境
echo "正在创建 conda 环境..."
conda create -n cosyvoice -y python=3.10

# 激活环境（需要特殊处理，因为 conda activate 在脚本中不能直接工作）
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate cosyvoice

# 安装依赖
echo "正在安装 pynini..."
conda install -y -c conda-forge pynini==2.1.5

echo "正在安装其他依赖..."
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

echo "正在升级 modelscope..."
pip install --upgrade modelscope

# 下载预训练模型
echo "正在下载预训练模型..."
cd "$WORK_DIR/CosyVoice"
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B

# 克隆测试仓库
echo "正在克隆测试仓库..."
cd "$WORK_DIR"
git clone --recursive https://github.com/joeseesun/CosyVoice-test

echo "===== 安装完成 ====="
echo "请使用以下命令激活环境并开始使用 CosyVoice:"
echo "cd $WORK_DIR"
echo "conda activate cosyvoice"
echo "===== 祝您使用愉快! ====="
