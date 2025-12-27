# AI-Tripod (ModelVerse) 部署指南

## 📋 项目概述

AI-Tripod 是一个基于 ModelVerse 的 LLM 模型一体化推训平台，提供模型管理、推理部署、训练微调和性能评估功能。

---

## 🖥️ 系统要求

### 硬件要求
| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4核 | 8核+ |
| 内存 | 16GB | 32GB+ |
| GPU | NVIDIA GPU (8GB显存) | NVIDIA GPU (16GB+显存) |
| 存储 | 50GB SSD | 200GB+ SSD |

### 软件要求
| 软件 | 版本要求 |
|------|----------|
| 操作系统 | Linux (Ubuntu 20.04+) 或 WSL2 |
| Python | 3.10+ |
| NVIDIA Driver | 525+ |
| CUDA | 12.1+ |

> ⚠️ **注意**: vLLM 仅支持 Linux 环境，Windows 用户请使用 WSL2。

---

## 🚀 快速部署

### 1. 克隆项目

```bash
git clone https://github.com/Guiwith/ModelVerse.git
cd ModelVerse
```

### 2. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

#### 方式一：使用完整依赖清单（推荐，确保版本一致）

```bash
# 使用已验证的完整依赖清单 (285个包)
pip install -r requirements-all.txt
```

> 说明：如果安装过程中出现 `protobuf` 与 `opentelemetry-*` 或 `oumi` 的版本冲突，通常是因为某些组件对 `protobuf` 的主版本要求不同。
> 当前推荐使用 `protobuf>=6.32,<7`（满足 oumi），并允许 pip 选择与之兼容的 `opentelemetry-*` 版本。

#### 方式二：使用 PyTorch 官方源安装 (CUDA 12.8)

```bash
# 安装 PyTorch (CUDA 12.8)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 安装核心依赖
pip install -r modelverse/requirements.txt

# 安装 oumi (训练框架)
pip install "oumi[gpu]==0.6.0"

# 安装 vLLM (推理引擎) - 必须使用与 PyTorch 兼容的版本
pip install "vllm==0.10.2"

# protobuf (oumi 需要)
pip install "protobuf>=6.32,<7"
```

#### 方式二：使用国内镜像源

```bash
# 使用清华镜像
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -r modelverse/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install "oumi[gpu]==0.6.0" -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install "vllm==0.10.2" -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install "protobuf>=6.32,<7" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 配置环境变量（可选，推荐国内用户）

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### 5. 启动服务

```bash
cd modelverse
python main.py
# 或使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8888
```

### 6. 访问应用

打开浏览器访问: http://localhost:8888

**默认账户**:
- 用户名: `admin`
- 密码: `admin123`

---

## 📦 核心依赖版本

以下是经过测试的兼容版本组合：

### 核心框架
| 包名 | 版本 | 说明 |
|------|------|------|
| torch | 2.8.0 | PyTorch 深度学习框架 |
| torchvision | 0.23.0 | PyTorch 视觉库 |
| torchaudio | 2.8.0 | PyTorch 音频库 |
| vllm | 0.10.2 | 高性能 LLM 推理引擎 |
| oumi | 0.6.0 | 模型训练框架 |
| transformers | 4.57.3 | HuggingFace Transformers |

### Web 框架
| 包名 | 版本 | 说明 |
|------|------|------|
| fastapi | 0.127.0 | 现代 Web API 框架 |
| uvicorn | 0.35.0 | ASGI 服务器 |
| pydantic | 2.12.5 | 数据验证 |

### 其他重要依赖
| 包名 | 版本 | 说明 |
|------|------|------|
| huggingface-hub | 0.36.0 | HuggingFace Hub 客户端 |
| accelerate | 1.12.0 | 分布式训练加速 |
| datasets | 4.4.2 | 数据集管理 |
| evaluate | 0.4.6 | 模型评估 |
| protobuf | >=6.32,<7 | Protocol Buffers |

---

## 📁 项目结构

```
ModelVerse/
├── modelverse/                 # 主应用目录
│   ├── main.py                # FastAPI 主入口
│   ├── models.py              # Pydantic 数据模型
│   ├── auth.py                # 用户认证
│   ├── database.py            # 数据库操作
│   ├── inference_utils.py     # vLLM 推理工具
│   ├── training_utils.py      # oumi 训练工具
│   ├── evaluation_utils.py    # 模型评估工具
│   ├── huggingface_utils.py   # HuggingFace 工具
│   ├── requirements.txt       # Python 依赖
│   ├── static/                # 前端静态文件
│   │   ├── index.html
│   │   ├── css/
│   │   └── js/
│   ├── models/                # 下载的模型存储目录
│   ├── datasets/              # 下载的数据集存储目录
│   ├── trainedmodels/         # 训练后的模型存储目录
│   ├── logs/                  # 日志目录
│   ├── training_configs/      # 训练配置目录
│   ├── evaluation_configs/    # 评估配置目录
│   └── evaluation_results/    # 评估结果目录
├── assets/                    # 项目资源文件
├── README.md                  # 项目说明
├── DEPLOYMENT.md              # 部署指南（本文件）
└── chat.py                    # 命令行聊天脚本
```

---

## ⚠️ 版本兼容性说明

### 重要：PyTorch、vLLM 和 oumi 版本必须兼容

| 组合 | PyTorch | vLLM | oumi | 状态 |
|------|---------|------|------|------|
| ✅ 推荐 | 2.8.0 | 0.10.2 | 0.6.0 | 已测试通过 |
| ❌ 不兼容 | 2.8.0 | 0.8.x | 0.6.0 | vLLM 需要 torch 2.6 |
| ❌ 不兼容 | 2.6.0 | 0.8.x | 0.6.0 | oumi 需要 torch>=2.6 |

### 版本约束
- **oumi 0.6.0**: 需要 `torch>=2.6,<2.9.0`
- **vLLM 0.10.2**: 需要 `torch==2.8.0`
- **PyTorch 2.8.0**: 需要 CUDA 12.1+

---

## 🔧 常见问题

### 1. vLLM 安装失败

```bash
# 确保使用正确的 PyTorch 版本
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.10.2
```

### 2. CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvidia-smi
python -c "import torch; print(torch.version.cuda)"

# 确保 NVIDIA 驱动版本 >= 525
```

### 3. protobuf 版本冲突

```bash
# oumi 需要 protobuf >= 6.32
pip install "protobuf>=6.32,<7"
```

### 4. HuggingFace 下载慢

```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 5. WSL2 环境配置

```bash
# 确保 WSL2 可以访问 GPU
nvidia-smi

# 如果失败，需要安装 NVIDIA CUDA on WSL
# 参考: https://docs.nvidia.com/cuda/wsl-user-guide/
```

---

## 🌐 端口说明

| 服务 | 默认端口 | 说明 |
|------|----------|------|
| Web 应用 | 8888 | 主应用端口 |
| vLLM 推理 | 8000-8099 | 推理服务端口范围 |

---

## 📝 数据库

项目使用 SQLite 数据库，首次启动时会自动创建：

- 数据库文件: `modelverse/modelverse.db`
- 自动创建默认管理员账户

---

## 🔒 默认账户

| 用户名 | 密码 | 角色 |
|--------|------|------|
| admin | admin123 | 管理员 |

> ⚠️ 生产环境请务必修改默认密码！

---

## 📊 功能说明

### 1. 资源管理
- 从 HuggingFace 下载模型和数据集
- 支持国内镜像加速
- 自动管理本地存储

### 2. 推理服务
- 基于 vLLM 的高性能推理
- OpenAI 兼容 API
- 内置聊天界面

### 3. 模型训练
- 基于 oumi 框架
- 支持 SFT/LoRA 等训练方式
- 实时训练日志

### 4. 模型评估
- 多种评估基准
- 自动评估报告

---

## 🛡️ 生产部署建议

### 1. 使用进程管理器

```bash
# 使用 nohup
nohup uvicorn main:app --host 0.0.0.0 --port 8888 > server.log 2>&1 &

# 或使用 systemd 服务
```

### 2. 反向代理 (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8888;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### 3. 环境变量配置

```bash
# 创建 .env 文件
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENABLE_HF_TRANSFER=0
```

---

## 📞 支持

如有问题，请提交 Issue: https://github.com/Guiwith/ModelVerse/issues

---

## 📄 许可证

本项目遵循开源原则，完全开源。核心依赖 oumi 框架。
