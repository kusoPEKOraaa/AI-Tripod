# AI_Tripod_backend — Django 后端骨架

此仓库为小组协作搭建的 Django 后端骨架，参考 LLaMA-Factory 的后端设计思路，提供：

- 一个轻量的 Django 项目骨架（`ai_tripod_backend`）。
- API 子应用（基于 Django REST framework 风格的接口骨架）。
- 训练模块骨架（`train`），包含运行入口 `run_train.py` 与管理命令示例。
- WebUI 模块骨架（`webui`），可与 Gradio 集成。
- `ChatModel` 抽象：统一模型加载与推理接口，便于后续替换具体引擎。

快速开始（使用 Conda 推荐流程）
-------------------------------

推荐使用 Conda 管理 Python 环境，以便更方便地安装 PyTorch / CUDA 相关包并避免依赖冲突。

1) 创建并激活 Conda 环境（推荐 Python 3.10）：

```bash
# 使用 environment.yml 创建
conda env create -f environment.yml
conda activate ai_tripod_backend
```

2) 安装 PyTorch（按你的硬件选择官方安装命令）

```bash
# CPU-only 示例
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# CUDA 示例 (按 CUDA 版本替换 cu116/cu118 等)：
# pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
```

3) 安装其它可选依赖：

```bash
pip install -r requirements.txt
```

4) 数据库迁移（第一次运行）并启动开发服务器：

```bash
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

5) 使用 ASGI/uvicorn（推荐用于生产测试或与反向代理配合）：

```bash
uvicorn ai_tripod_backend.asgi:application --host 0.0.0.0 --port 8000
```

6) 启动 WebUI (Gradio) demo（本仓库包含一个最小 Gradio demo，调用本地 `ChatModel`）：

```bash
python run_webui.py
```

7) 运行训练入口（示例）：

```bash
# 优先使用 Django 管理命令
python manage.py run_train --config path/to/config.yaml

# 或直接使用脚本（会在 Django 不可用时 fallback）
python run_train.py
```

项目结构与说明
--------------

- `ai_tripod_backend/` - Django 项目配置（settings, urls, asgi, wsgi）。
- `ai_tripod_backend/api/` - API 子应用，包含 `views.py`、`urls.py` 等。
- `ai_tripod_backend/train/` - 训练子应用，包含训练 runner 与管理命令示例。
- `ai_tripod_backend/webui/` - Web UI 子应用，和 Gradio 集成入口。
- `ai_tripod_backend/chat.py` - `ChatModel` 抽象，模拟 LLaMA-Factory 中的模型抽象层。

后续工作建议
------------

- 将 `ChatModel` 与真实的 Transformers / vLLM / modelscope 集成。
- 在 `api` 中添加鉴权、速率限制与监控中间件。
- 为 `train` 补充实际的训练逻辑（参考 LLaMA-Factory 的 `train` 目录）。
- 在 CI 中添加 lint、测试与 basic smoke tests。
